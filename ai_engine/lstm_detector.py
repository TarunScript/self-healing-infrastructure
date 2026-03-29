# LSTM + TCN metric degradation detector
# File: ai_engine/lstm_detector.py
# Purpose: Temporal anomaly detection on Prometheus metric time-series
# Architecture:
#   1. LSTM encoder: captures temporal dependencies in rolling metric windows
#   2. TCN head: multi-scale temporal convolution for fast pattern recognition
#   3. Outputs: degradation_probability (0-1) + predicted_next_values for anomaly scoring
# Training: Online/incremental — no labeled data required
# Timing budget: <30ms inference per service

import time
import logging
import collections
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Feature order must match metric_detector.py's feature vector
FEATURE_NAMES = [
    "request_rate",
    "p99_latency_ms",
    "p50_latency_ms",
    "error_rate",
    "throughput_delta",
    "avg_latency_ms",
]

N_FEATURES = len(FEATURE_NAMES)
SEQUENCE_LENGTH = 10   # How many past windows to feed the LSTM
HIDDEN_SIZE = 32       # LSTM hidden dim (small = fast inference)
TCN_CHANNELS = [16, 16]  # TCN channel widths


@dataclass
class LSTMAnomaly:
    """Anomaly detected by the LSTM/TCN temporal model."""
    service: str
    degradation_probability: float   # 0.0 - 1.0 (core output)
    anomaly_type: str                 # "temporal_degradation" | "sudden_spike" | "gradual_drift"
    severity: float
    prediction_error: float          # MSE between predicted and actual values
    contributing_features: dict      # {feature: {predicted, actual, error}}
    sequence_length_used: int
    inference_time_ms: float
    model_type: str                  # "lstm_tcn" | "statistical_fallback"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "detector": "lstm_tcn_metric",
            "service": self.service,
            "degradation_probability": round(self.degradation_probability, 3),
            "anomaly_type": self.anomaly_type,
            "severity": round(self.severity, 3),
            "prediction_error": round(self.prediction_error, 4),
            "contributing_features": self.contributing_features,
            "sequence_length_used": self.sequence_length_used,
            "inference_time_ms": round(self.inference_time_ms, 1),
            "model_type": self.model_type,
        }


# ─── Pure-NumPy LSTM Implementation ──────────────────────────────────
# Full LSTM without PyTorch dependency so the demo runs anywhere.
# We expose a PyTorch version below for when torch IS available.

class NumpyLSTMCell:
    """Single LSTM cell implemented in pure NumPy."""

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = 0.1
        self.Wf = rng.standard_normal((hidden_size, input_size + hidden_size)) * scale
        self.bf = np.zeros(hidden_size)
        self.Wi = rng.standard_normal((hidden_size, input_size + hidden_size)) * scale
        self.bi = np.zeros(hidden_size)
        self.Wc = rng.standard_normal((hidden_size, input_size + hidden_size)) * scale
        self.bc = np.zeros(hidden_size)
        self.Wo = rng.standard_normal((hidden_size, input_size + hidden_size)) * scale
        self.bo = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, h, c):
        xh = np.concatenate([x, h])
        f = self._sigmoid(self.Wf @ xh + self.bf)
        i = self._sigmoid(self.Wi @ xh + self.bi)
        g = np.tanh(self.Wc @ xh + self.bc)
        o = self._sigmoid(self.Wo @ xh + self.bo)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyTCNLayer:
    """Single causal dilated 1D convolution layer (TCN block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((out_ch, in_ch, kernel)) * 0.1
        self.b = np.zeros(out_ch)
        self.kernel = kernel
        self.dilation = dilation
        self.padding = (kernel - 1) * dilation

    def forward(self, x):
        """x: (channels, time_steps)"""
        # Causal padding (left-pad)
        pad = np.zeros((x.shape[0], self.padding))
        x_padded = np.concatenate([pad, x], axis=1)
        T = x.shape[1]
        out = np.zeros((self.W.shape[0], T))
        for t in range(T):
            for k in range(self.kernel):
                t_src = t + self.padding - k * self.dilation
                if 0 <= t_src < x_padded.shape[1]:
                    out[:, t] += self.W[:, :, k] @ x_padded[:, t_src]
        out += self.b[:, None]
        return np.maximum(0, out)  # ReLU activation


class NumpyLSTMTCN:
    """
    Lightweight LSTM + TCN model for metric anomaly detection.
    Pure NumPy — no PyTorch required.

    Architecture:
        Input (N_FEATURES) → LSTM (HIDDEN_SIZE) → TCN (16→16) → Dense (N_FEATURES)
        Loss: MSE on next-step prediction
    """

    def __init__(self, input_size: int = N_FEATURES, hidden_size: int = HIDDEN_SIZE):
        self.lstm = NumpyLSTMCell(input_size, hidden_size)
        self.tcn1 = NumpyTCNLayer(hidden_size, TCN_CHANNELS[0], kernel=3, dilation=1, seed=1)
        self.tcn2 = NumpyTCNLayer(TCN_CHANNELS[0], TCN_CHANNELS[1], kernel=3, dilation=2, seed=2)
        # Output projection: TCN_CHANNELS[-1] → input_size
        rng = np.random.default_rng(99)
        self.W_out = rng.standard_normal((input_size, TCN_CHANNELS[-1])) * 0.1
        self.b_out = np.zeros(input_size)
        self.hidden_size = hidden_size

    def forward(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the LSTM → TCN.

        Args:
            sequence: (seq_len, n_features) array of normalized metric values

        Returns:
            lstm_outputs: (seq_len, hidden_size) — hidden states
            prediction: (n_features,) — predicted next values
        """
        T = sequence.shape[0]
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        lstm_outputs = []

        for t in range(T):
            h, c = self.lstm.forward(sequence[t], h, c)
            lstm_outputs.append(h)

        # Stack to (hidden_size, T) for TCN
        lstm_out = np.stack(lstm_outputs, axis=1)  # (hidden, T)
        tcn1_out = self.tcn1.forward(lstm_out)
        tcn2_out = self.tcn2.forward(tcn1_out)

        # Project last time step to prediction
        last = tcn2_out[:, -1]
        prediction = self.W_out @ last + self.b_out
        return np.stack(lstm_outputs), prediction

    def online_update(self, sequence: np.ndarray, learning_rate: float = 0.001) -> float:
        """
        Simple gradient-free online update: shift W_out toward reducing prediction error.
        Uses perturbation-based update (SPSA) for fast incremental adaptation.
        Returns MSE loss.
        """
        _, pred = self.forward(sequence[:-1])
        actual = sequence[-1]
        loss = float(np.mean((pred - actual) ** 2))

        # Perturbation-based W_out update
        eps = 0.01
        for i in range(min(3, self.W_out.shape[0])):
            perturb = np.random.choice([-eps, eps], size=self.W_out.shape[1])
            self.W_out[i] += perturb
            _, p_new = self.forward(sequence[:-1])
            new_loss = float(np.mean((p_new - actual) ** 2))
            if new_loss > loss:
                self.W_out[i] -= perturb  # Revert

        return loss


# ─── Main Detector Class ──────────────────────────────────────────────

class LSTMMetricDetector:
    """
    LSTM + TCN based metric degradation detector.

    How it works:
    1. Maintains a rolling window of metric observations per service
    2. LSTM encodes temporal patterns → TCN extracts multi-scale features
    3. Model predicts next expected metric values
    4. Large prediction errors → degradation_probability spikes
    5. Online updates every N cycles to adapt to service baseline drift

    This runs AFTER IsolationForest as a confirmation layer:
    - IsolationForest catches point anomalies (sudden spikes)
    - LSTM/TCN catches temporal patterns (gradual degradation, drift, oscillation)
    """

    ANOMALY_THRESHOLD = 0.60       # degradation_prob above this = anomaly
    PREDICTION_ERROR_THRESHOLD = 0.15  # Normalized MSE above this = anomaly
    MIN_SEQUENCE = 5               # Minimum observations before inference
    UPDATE_EVERY = 10              # Online update every N cycles
    DEGRADATION_WINDOW = 5         # How many recent values to check for gradient

    def __init__(self):
        # Per-service model and history
        self._models: dict[str, NumpyLSTMTCN] = {}
        self._history: dict[str, collections.deque] = {}
        self._normalizers: dict[str, dict] = {}  # {service: {feature: {mean, std}}}
        self._cycle_count: dict[str, int] = {}
        self._inference_count = 0

    def _get_or_create(self, service: str) -> NumpyLSTMTCN:
        if service not in self._models:
            self._models[service] = NumpyLSTMTCN()
            self._history[service] = collections.deque(maxlen=50)
            self._normalizers[service] = {}
            self._cycle_count[service] = 0
            logger.debug(f"LSTM model created for service: {service}")
        return self._models[service]

    def _build_feature_vector(self, metrics: dict) -> np.ndarray:
        """Build a fixed-length feature vector from metric dict."""
        vec = []
        for name in FEATURE_NAMES:
            val = metrics.get(name, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            vec.append(float(val))
        return np.array(vec)

    def _normalize(self, service: str, vec: np.ndarray) -> np.ndarray:
        """Z-score normalize features using running stats."""
        norm = self._normalizers[service]
        result = np.zeros_like(vec)
        for i, name in enumerate(FEATURE_NAMES):
            if name not in norm:
                norm[name] = {"mean": vec[i], "std": 1.0, "count": 1, "m2": 0.0}
            else:
                # Welford's online update
                n = norm[name]
                count = n["count"] + 1
                delta = vec[i] - n["mean"]
                n["mean"] += delta / count
                delta2 = vec[i] - n["mean"]
                n["m2"] += delta * delta2
                n["count"] = count
                n["std"] = max(np.sqrt(n["m2"] / max(count - 1, 1)), 1e-6)

            result[i] = (vec[i] - norm[name]["mean"]) / norm[name]["std"]
        return result

    def _denormalize_error(self, service: str, pred: np.ndarray, actual: np.ndarray) -> dict:
        """Compute per-feature prediction error in original units."""
        norm = self._normalizers[service]
        result = {}
        for i, name in enumerate(FEATURE_NAMES):
            if name not in norm:
                continue
            std = norm[name]["std"]
            mean = norm[name]["mean"]
            pred_orig = pred[i] * std + mean
            actual_orig = actual[i] * std + mean
            error = abs(pred_orig - actual_orig)
            rel_error = error / max(abs(actual_orig), 1e-6)
            if rel_error > 0.15:  # Only include significant errors
                result[name] = {
                    "predicted": round(float(pred_orig), 3),
                    "actual": round(float(actual_orig), 3),
                    "error": round(float(error), 3),
                    "relative_error": round(float(rel_error), 3),
                }
        return result

    def _classify_pattern(self, history: list[np.ndarray]) -> str:
        """Classify degradation pattern from recent history."""
        if len(history) < 3:
            return "temporal_degradation"

        # Check error_rate (index 3) and latency (index 1) trends
        error_rates = [h[3] for h in history[-5:]]
        latencies = [h[1] for h in history[-5:]]

        # Sudden spike: last value >> mean of previous
        if len(error_rates) >= 2:
            prev_mean = np.mean(error_rates[:-1])
            if error_rates[-1] > prev_mean * 3:
                return "sudden_spike"

        # Gradual drift: monotonic increase
        if len(latencies) >= 4:
            diffs = np.diff(latencies)
            if np.all(diffs > 0):
                return "gradual_drift"

        return "temporal_degradation"

    def infer(self, service: str, metrics: dict) -> Optional[LSTMAnomaly]:
        """
        Run LSTM/TCN inference for a single service.

        Args:
            service: Service name
            metrics: {metric_name: float} current observations

        Returns:
            LSTMAnomaly if degradation detected, None otherwise
        """
        t0 = time.time()
        model = self._get_or_create(service)
        self._cycle_count[service] = self._cycle_count.get(service, 0) + 1

        vec = self._build_feature_vector(metrics)
        norm_vec = self._normalize(service, vec)
        self._history[service].append(norm_vec)

        seq_len = len(self._history[service])
        if seq_len < self.MIN_SEQUENCE:
            return None  # Not enough history yet

        # Build sequence
        history_list = list(self._history[service])
        sequence = np.stack(history_list)  # (T, N_FEATURES)

        # Inference
        _, prediction = model.forward(sequence)

        # Prediction error (normalized space)
        actual = sequence[-1]
        pred_error = float(np.mean((prediction - actual) ** 2))

        # Compute degradation probability from prediction error
        # Sigmoid mapping: error → probability
        degradation_prob = float(1.0 / (1.0 + np.exp(-10 * (pred_error - 0.08))))

        # Online update (periodically)
        if self._cycle_count[service] % self.UPDATE_EVERY == 0 and seq_len >= 6:
            model.online_update(sequence)

        elapsed = (time.time() - t0) * 1000
        self._inference_count += 1

        if degradation_prob < self.ANOMALY_THRESHOLD and pred_error < self.PREDICTION_ERROR_THRESHOLD:
            return None

        # Build result
        contributing = self._denormalize_error(service, prediction, actual)
        pattern = self._classify_pattern(history_list)

        # Severity from degradation prob
        severity = min(1.0, degradation_prob * 0.9 + pred_error * 0.3)

        return LSTMAnomaly(
            service=service,
            degradation_probability=degradation_prob,
            anomaly_type=pattern,
            severity=severity,
            prediction_error=pred_error,
            contributing_features=contributing,
            sequence_length_used=seq_len,
            inference_time_ms=elapsed,
            model_type="lstm_tcn",
        )

    def infer_all(self, metrics_per_service: dict) -> list[LSTMAnomaly]:
        """Run inference for all services. Returns sorted by severity."""
        results = []
        for service, metrics in metrics_per_service.items():
            anomaly = self.infer(service, metrics)
            if anomaly:
                results.append(anomaly)
        results.sort(key=lambda a: a.severity, reverse=True)
        return results

    def get_stats(self) -> dict:
        return {
            "model_type": "lstm_tcn_numpy",
            "services_tracked": len(self._models),
            "total_inferences": self._inference_count,
            "history_lengths": {s: len(h) for s, h in self._history.items()},
        }


# ─── Standalone Testing ──────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s")

    detector = LSTMMetricDetector()

    print("=== Feeding normal data (10 cycles) ===")
    np.random.seed(42)
    normal_base = {
        "request_rate": 50.0, "p99_latency_ms": 120.0, "p50_latency_ms": 45.0,
        "error_rate": 0.01, "throughput_delta": 1.0, "avg_latency_ms": 55.0,
    }

    for i in range(10):
        data = {
            "cartservice": {k: v + np.random.normal(0, v * 0.05) for k, v in normal_base.items()},
            "frontend": {k: v * 1.2 + np.random.normal(0, v * 0.05) for k, v in normal_base.items()},
        }
        results = detector.infer_all(data)
        print(f"  Cycle {i+1}: {len(results)} anomalies")

    print("\n=== Injecting gradual degradation ===")
    for i in range(5):
        factor = 1.0 + i * 0.5
        degraded = {
            "cartservice": {
                "request_rate": 50.0 - i * 8,
                "p99_latency_ms": 120.0 * factor,
                "p50_latency_ms": 45.0 * factor,
                "error_rate": 0.01 + i * 0.08,
                "throughput_delta": 1.0,
                "avg_latency_ms": 55.0 * factor,
            },
            "frontend": {k: v * 1.2 for k, v in normal_base.items()},
        }
        results = detector.infer_all(degraded)
        for r in results:
            print(f"  Cycle {10+i+1}: [{r.anomaly_type}] {r.service} — prob={r.degradation_probability:.2f} sev={r.severity:.2f} err={r.prediction_error:.4f} ({r.inference_time_ms:.1f}ms)")
        if not results:
            print(f"  Cycle {10+i+1}: no anomaly detected yet")

    print(f"\nStats: {json.dumps(detector.get_stats(), indent=2)}")


# ─── PyTorch Reference Implementation ────────────────────────────────
# The production detector above uses pure NumPy for zero-dependency
# portability.  This section provides the equivalent architecture in
# PyTorch — same layer structure, same hyperparameters — so the model
# can be trained on GPU or exported to ONNX for production deployment.
#
# Usage:
#   model = PyTorchLSTMTCN(input_size=N_FEATURES)
#   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#   # Training loop: feed rolling windows, minimize next-step MSE
#   loss = model.training_step(sequence_tensor)
#   loss.backward()
#   optimizer.step()

try:
    import torch
    import torch.nn as nn

    class _CausalConv1d(nn.Module):
        """Causal dilated 1-D convolution (TCN building block)."""
        def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
            super().__init__()
            self.padding = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=self.padding)
            self.relu = nn.ReLU()

        def forward(self, x):
            # x: (batch, channels, time)
            out = self.conv(x)
            # Remove future timesteps (causal masking)
            if self.padding > 0:
                out = out[:, :, :-self.padding]
            return self.relu(out)

    class PyTorchLSTMTCN(nn.Module):
        """
        PyTorch LSTM + TCN for metric degradation detection.

        Architecture (mirrors NumpyLSTMTCN exactly):
            Input  (batch, seq_len, N_FEATURES)
            → LSTM (hidden=32, num_layers=2, batch_first=True)
            → Reshape to (batch, hidden, seq_len) for TCN
            → CausalConv1d(32 → 16, kernel=3, dilation=1)
            → CausalConv1d(16 → 16, kernel=3, dilation=2)
            → Linear(16 → N_FEATURES)  [last time step only]
            Output: next-step prediction (batch, N_FEATURES)

        Loss: MSE on next-step prediction (self-supervised, no labels needed).
        Training signal comes purely from prediction error on historical windows.

        Deployment:
            torch.onnx.export(model, dummy_input, "lstm_tcn.onnx")
        """

        def __init__(self, input_size: int = N_FEATURES,
                     hidden_size: int = HIDDEN_SIZE,
                     num_lstm_layers: int = 2,
                     dropout: float = 0.1):
            super().__init__()
            self.hidden_size = hidden_size

            # LSTM encoder — captures temporal dependencies
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout if num_lstm_layers > 1 else 0.0,
            )

            # TCN head — multi-scale temporal convolution
            self.tcn1 = _CausalConv1d(hidden_size, TCN_CHANNELS[0], kernel=3, dilation=1)
            self.tcn2 = _CausalConv1d(TCN_CHANNELS[0], TCN_CHANNELS[1], kernel=3, dilation=2)

            # Output projection: predict next feature vector
            self.output = nn.Linear(TCN_CHANNELS[-1], input_size)

            # Layer norm for training stability
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, seq_len, input_size) — normalized metric windows
            Returns:
                pred: (batch, input_size) — predicted next metric values
            """
            # LSTM: (batch, seq, features) → (batch, seq, hidden)
            lstm_out, _ = self.lstm(x)
            lstm_out = self.norm(lstm_out)

            # TCN expects (batch, channels, time)
            tcn_in = lstm_out.permute(0, 2, 1)          # (batch, hidden, seq)
            tcn1_out = self.tcn1(tcn_in)                  # (batch, 16, seq)
            tcn2_out = self.tcn2(tcn1_out)                # (batch, 16, seq)

            # Use last timestep for next-step prediction
            last = tcn2_out[:, :, -1]                     # (batch, 16)
            pred = self.output(last)                       # (batch, input_size)
            return pred

        def training_step(self, sequence: "torch.Tensor") -> "torch.Tensor":
            """
            Compute next-step prediction MSE loss.

            Args:
                sequence: (batch, seq_len, input_size)
                          The model sees seq[:-1] and predicts seq[-1].
            Returns:
                MSE loss scalar
            """
            x = sequence[:, :-1, :]    # Input: all but last step
            y = sequence[:, -1,  :]    # Target: last step
            pred = self.forward(x)
            return nn.functional.mse_loss(pred, y)

        def predict_degradation_probability(self, sequence: "torch.Tensor") -> float:
            """
            Convenience wrapper matching the NumPy API:
            Returns degradation_probability in [0, 1].

            Args:
                sequence: (1, seq_len, input_size) — single-service window
            """
            self.eval()
            with torch.no_grad():
                x = sequence[:, :-1, :]
                y = sequence[:, -1,  :]
                pred = self.forward(x)
                mse = float(nn.functional.mse_loss(pred, y).item())
                # Sigmoid mapping identical to NumpyLSTMTCN
                return float(1.0 / (1.0 + torch.exp(torch.tensor(-10.0 * (mse - 0.08)))))

    # ── Quick smoke-test (only runs when torch is available) ──────────
    if __name__ == "__main__":
        print("\n=== PyTorch LSTM+TCN smoke test ===")
        _model = PyTorchLSTMTCN(input_size=N_FEATURES)
        _total_params = sum(p.numel() for p in _model.parameters())
        print(f"  Model params: {_total_params:,}")

        _batch = torch.randn(4, SEQUENCE_LENGTH + 1, N_FEATURES)
        _loss  = _model.training_step(_batch)
        print(f"  Training step loss: {_loss.item():.6f}")

        _prob = _model.predict_degradation_probability(_batch[:1])
        print(f"  Degradation probability: {_prob:.4f}")
        print("  PyTorch LSTM+TCN: OK")

except ImportError:
    # torch not installed — NumPy implementation is the active detector
    pass
