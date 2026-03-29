# Metric anomaly detection using Isolation Forest on Prometheus time-series data
# File: ai_engine/metric_detector.py
# Performance target: <50ms inference per detection cycle
# Approach:
#   1. Query Prometheus for key metrics (latency, error rate, CPU, memory)
#   2. Build feature vectors per service from recent metric windows
#   3. Isolation Forest flags outliers (unsupervised, no labeled data needed)
#   4. Return structured anomaly results with contributing metric evidence

import time
import random
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class MetricAnomaly:
    """A detected metric anomaly with evidence for the explainability engine."""
    service: str
    anomaly_type: str              # "latency_spike" | "error_rate" | "throughput_anomaly" | "multi_signal"
    severity: float                # 0.0 - 1.0
    anomaly_score: float           # Raw Isolation Forest score (-1 to 0 range, more negative = more anomalous)
    contributing_metrics: dict     # {metric_name: {current: x, baseline: y, deviation: z}}
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "detector": "isolation_forest_metric",
            "service": self.service,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "anomaly_score": round(self.anomaly_score, 4),
            "contributing_metrics": self.contributing_metrics,
            "timestamp": self.timestamp,
        }


# Prometheus queries for key signals
# Each returns a per-service instant value
METRIC_QUERIES = {
    "request_rate": {
        "query": 'sum by (service_name) (rate(calls_total{service_name!=""}[1m]))',
        "unit": "req/s",
    },
    "p99_latency_ms": {
        "query": 'histogram_quantile(0.99, sum by (service_name, le) (rate(duration_milliseconds_bucket{service_name!=""}[1m])))',
        "unit": "ms",
    },
    "p50_latency_ms": {
        "query": 'histogram_quantile(0.50, sum by (service_name, le) (rate(duration_milliseconds_bucket{service_name!=""}[1m])))',
        "unit": "ms",
    },
    "error_rate": {
        "query": 'sum by (service_name) (rate(calls_total{status_code="STATUS_CODE_ERROR",service_name!=""}[1m])) / sum by (service_name) (rate(calls_total{service_name!=""}[1m]))',
        "unit": "ratio",
    },
    "throughput_delta": {
        "query": 'sum by (service_name) (rate(calls_total{service_name!=""}[1m])) / sum by (service_name) (rate(calls_total{service_name!=""}[5m]))',
        "unit": "ratio",
    },
    "avg_latency_ms": {
        "query": 'sum by (service_name) (rate(duration_milliseconds_sum{service_name!=""}[1m])) / sum by (service_name) (rate(duration_milliseconds_count{service_name!=""}[1m]))',
        "unit": "ms",
    },
}

# Fallback queries for alternative metric naming conventions
FALLBACK_QUERIES = {
    "request_rate": {
        "query": 'sum by (service_name) (rate(duration_milliseconds_count{service_name!=""}[1m]))',
        "unit": "req/s",
    },
    "error_rate": {
        "query": 'sum by (service_name) (rate(duration_milliseconds_count{status_code=~"5..",service_name!=""}[1m])) / sum by (service_name) (rate(duration_milliseconds_count{service_name!=""}[1m]))',
        "unit": "ratio",
    },
}


class MetricAnomalyDetector:
    """
    Isolation Forest-based metric anomaly detector.

    How it works:
    1. Queries Prometheus for latency, error rate, CPU, memory per service
    2. Builds a feature matrix [services × metrics] from current values
    3. Maintains a rolling history buffer for training the Isolation Forest
    4. Scores each service — outliers (score < threshold) are flagged
    5. Identifies which specific metrics contribute most to the anomaly

    Timing budget: <50ms for scoring (Prometheus query is separate)
    """

    # Minimum data points before the model starts scoring
    MIN_TRAINING_SAMPLES = 10
    # Isolation Forest contamination parameter (expected anomaly fraction)
    CONTAMINATION = 0.1
    # Threshold for considering a metric as contributing to an anomaly
    DEVIATION_THRESHOLD = 2.0  # Standard deviations

    def __init__(self, prometheus_url: str = "http://prometheus:9090",
                 history_size: int = 100):
        """
        Args:
            prometheus_url: Prometheus HTTP API endpoint
            history_size: Max rows in training buffer per service
        """
        self.prometheus_url = prometheus_url
        self.history_size = history_size

        # Per-service feature history: {service: [[feat1, feat2, ...], ...]}
        self._history: dict[str, list[list[float]]] = defaultdict(list)
        # Per-service trained models
        self._models: dict[str, IsolationForest] = {}
        # Per-service baseline stats for deviation analysis
        self._baselines: dict[str, dict[str, dict]] = {}  # {service: {metric: {mean, std}}}
        # Feature names in order
        self._feature_names: list[str] = list(METRIC_QUERIES.keys())
        # Cycle counter for periodic retraining
        self._cycle_count: int = 0
        self._retrain_interval: int = 5  # Retrain every N cycles

        # LSTM/TCN temporal detector — catches gradual drift that IsolationForest misses
        try:
            from lstm_detector import LSTMMetricDetector
            self._lstm = LSTMMetricDetector()
            logger.info("LSTM/TCN temporal detector initialized alongside IsolationForest")
        except Exception as e:
            self._lstm = None
            logger.warning(f"LSTM detector unavailable ({e}) — IsolationForest-only mode")

    def _query_prometheus(self, query: str) -> dict[str, float]:
        """
        Execute an instant query against Prometheus.

        Returns: {service_name: value}
        """
        try:
            resp = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=1,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return {}

        results = {}
        for item in data.get("data", {}).get("result", []):
            service = item.get("metric", {}).get("service_name", "")
            if not service:
                continue
            try:
                value = float(item["value"][1])
                if not (np.isnan(value) or np.isinf(value)):
                    results[service] = value
            except (IndexError, ValueError, TypeError):
                continue

        return results

    def _generate_simulated_metrics(self) -> dict[str, dict[str, float]]:
        """
        Generate realistic simulated metrics when Prometheus is unavailable.
        Cycles through 3 failure scenarios matching the log detector.
        """
        now = time.time()
        cycle = int(now / 30) % 3  # Rotate scenario every 30 seconds

        # Base normal metrics for all services
        services = ["frontend", "productcatalogservice", "currencyservice",
                    "cartservice", "checkoutservice", "recommendationservice",
                    "shippingservice", "emailservice", "paymentservice"]

        metrics = {}
        for svc in services:
            noise = lambda v: v + random.gauss(0, v * 0.08)
            metrics[svc] = {
                "request_rate": noise(45.0),
                "p99_latency_ms": noise(110.0),
                "p50_latency_ms": noise(40.0),
                "error_rate": noise(0.01),
                "throughput_delta": noise(1.0),
                "avg_latency_ms": noise(55.0),
            }

        # Inject anomalous metrics based on current scenario
        if cycle == 0:
            # Scenario 1: cartservice crash — Redis unavailable
            metrics["cartservice"] = {
                "request_rate": random.uniform(2.0, 8.0),
                "p99_latency_ms": random.uniform(3000, 8000),
                "p50_latency_ms": random.uniform(1500, 4000),
                "error_rate": random.uniform(0.6, 0.95),
                "throughput_delta": random.uniform(0.1, 0.3),
                "avg_latency_ms": random.uniform(2000, 5000),
            }
            metrics["frontend"]["error_rate"] = random.uniform(0.15, 0.35)
            metrics["frontend"]["p99_latency_ms"] = random.uniform(500, 1500)

        elif cycle == 1:
            # Scenario 2: productcatalogservice CPU spike
            metrics["productcatalogservice"] = {
                "request_rate": random.uniform(3.0, 10.0),
                "p99_latency_ms": random.uniform(2000, 6000),
                "p50_latency_ms": random.uniform(800, 2500),
                "error_rate": random.uniform(0.3, 0.6),
                "throughput_delta": random.uniform(0.15, 0.35),
                "avg_latency_ms": random.uniform(1200, 3500),
            }
            metrics["recommendationservice"]["p99_latency_ms"] = random.uniform(400, 1200)
            metrics["recommendationservice"]["error_rate"] = random.uniform(0.1, 0.25)

        elif cycle == 2:
            # Scenario 3: checkoutservice OOM
            metrics["checkoutservice"] = {
                "request_rate": random.uniform(1.0, 5.0),
                "p99_latency_ms": random.uniform(5000, 12000),
                "p50_latency_ms": random.uniform(2000, 6000),
                "error_rate": random.uniform(0.7, 0.98),
                "throughput_delta": random.uniform(0.05, 0.2),
                "avg_latency_ms": random.uniform(4000, 8000),
            }
            metrics["frontend"]["error_rate"] = random.uniform(0.2, 0.4)
            metrics["frontend"]["p99_latency_ms"] = random.uniform(600, 2000)

        logger.info(f"Generated simulated metrics for {len(metrics)} services (scenario {cycle + 1}/3)")
        return metrics

    def fetch_metrics(self) -> dict[str, dict[str, float]]:
        """
        Fetch all metric values from Prometheus, organized by service.
        Falls back to simulated metrics when Prometheus is unreachable.

        Returns: {service: {metric_name: value}}
        """
        t_start = time.time()

        # Quick probe: try a single simple query to check if Prometheus is up
        try:
            resp = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": "up"},
                timeout=1,
            )
            resp.raise_for_status()
            prometheus_up = True
        except requests.RequestException:
            prometheus_up = False

        if not prometheus_up:
            logger.info("Prometheus unreachable, using simulated metrics")
            return self._generate_simulated_metrics()

        service_metrics: dict[str, dict[str, float]] = defaultdict(dict)

        for metric_name, config in METRIC_QUERIES.items():
            values = self._query_prometheus(config["query"])

            # Try fallback if primary query returns no results
            if not values and metric_name in FALLBACK_QUERIES:
                values = self._query_prometheus(FALLBACK_QUERIES[metric_name]["query"])

            for service, value in values.items():
                service_metrics[service][metric_name] = value

        # If Prometheus returned nothing, use simulation
        if not service_metrics:
            logger.info("Prometheus returned no data, using simulated metrics")
            return self._generate_simulated_metrics()

        elapsed = time.time() - t_start
        logger.debug(f"Fetched metrics for {len(service_metrics)} services in {elapsed:.3f}s")
        return dict(service_metrics)

    def _build_feature_vector(self, metrics: dict[str, float]) -> list[float]:
        """Convert a service's metrics dict into a fixed-length feature vector."""
        return [metrics.get(name, 0.0) for name in self._feature_names]

    def _update_history(self, service: str, features: list[float]) -> None:
        """Append features to service's rolling history buffer."""
        history = self._history[service]
        history.append(features)
        # Trim to bounded size
        if len(history) > self.history_size:
            self._history[service] = history[-self.history_size:]

    def _update_baselines(self, service: str) -> None:
        """Compute mean/std per metric for deviation analysis."""
        history = self._history[service]
        if len(history) < 3:
            return

        arr = np.array(history)
        baselines = {}
        for i, name in enumerate(self._feature_names):
            col = arr[:, i]
            baselines[name] = {
                "mean": float(np.mean(col)),
                "std": float(max(np.std(col), 1e-6)),  # Avoid division by zero
            }
        self._baselines[service] = baselines

    def _train_model(self, service: str) -> None:
        """Train/retrain Isolation Forest for a service."""
        history = self._history[service]
        if len(history) < self.MIN_TRAINING_SAMPLES:
            return

        X = np.array(history)
        model = IsolationForest(
            n_estimators=50,          # Fewer trees for speed
            max_samples=min(len(X), 50),
            contamination=self.CONTAMINATION,
            random_state=42,
            n_jobs=1,                 # Single-threaded for predictability
        )
        model.fit(X)
        self._models[service] = model
        logger.debug(f"Trained Isolation Forest for {service} with {len(X)} samples")

    def _classify_anomaly(self, service: str, features: list[float],
                         metrics: dict[str, float]) -> Optional[tuple[str, dict]]:
        """
        Determine anomaly type and identify contributing metrics.

        Returns: (anomaly_type, contributing_metrics) or None
        """
        baselines = self._baselines.get(service, {})
        if not baselines:
            return None

        deviations = {}
        for i, name in enumerate(self._feature_names):
            baseline = baselines.get(name)
            if not baseline:
                continue
            current = features[i]
            z_score = (current - baseline["mean"]) / baseline["std"]
            if abs(z_score) > self.DEVIATION_THRESHOLD:
                deviations[name] = {
                    "current": round(current, 4),
                    "baseline_mean": round(baseline["mean"], 4),
                    "baseline_std": round(baseline["std"], 4),
                    "z_score": round(z_score, 2),
                    "unit": METRIC_QUERIES.get(name, {}).get("unit", ""),
                }

        if not deviations:
            return None

        # Classify by dominant signal
        deviation_names = set(deviations.keys())
        if "error_rate" in deviation_names:
            anomaly_type = "error_rate"
        elif deviation_names & {"p99_latency_ms", "p50_latency_ms", "avg_latency_ms"}:
            anomaly_type = "latency_spike"
        elif "throughput_delta" in deviation_names:
            anomaly_type = "throughput_anomaly"
        else:
            anomaly_type = "multi_signal"

        return anomaly_type, deviations

    def detect(self, metrics: Optional[dict[str, dict[str, float]]] = None) -> list[MetricAnomaly]:
        """
        Run anomaly detection on current metrics.

        Args:
            metrics: Pre-fetched {service: {metric: value}}, or None to fetch from Prometheus

        Returns:
            List of MetricAnomaly objects, sorted by severity descending

        Performance: <50ms for scoring (excludes Prometheus fetch)
        """
        t_start = time.time()
        self._cycle_count += 1

        if metrics is None:
            metrics = self.fetch_metrics()

        if not metrics:
            return []

        anomalies = []
        scoring_start = time.time()

        for service, service_metrics in metrics.items():
            features = self._build_feature_vector(service_metrics)
            self._update_history(service, features)
            self._update_baselines(service)

            # Retrain periodically
            if self._cycle_count % self._retrain_interval == 0:
                self._train_model(service)

            # Score with Isolation Forest (if model exists)
            model = self._models.get(service)
            if model is None:
                # Not enough data yet — train if we have enough
                if len(self._history[service]) >= self.MIN_TRAINING_SAMPLES:
                    self._train_model(service)
                    model = self._models.get(service)
                if model is None:
                    continue

            X = np.array([features])
            score = float(model.score_samples(X)[0])
            prediction = int(model.predict(X)[0])

            # prediction == -1 means anomaly
            if prediction == -1:
                classification = self._classify_anomaly(service, features, service_metrics)
                if classification is None:
                    continue

                anomaly_type, contributing_metrics = classification

                # Severity: map score from [-0.5, 0] to [0.5, 1.0]
                # More negative score = more anomalous
                severity = min(1.0, max(0.3, 0.5 + abs(score) * 1.5))

                # Boost severity for error_rate anomalies
                if anomaly_type == "error_rate":
                    severity = min(1.0, severity + 0.15)

                anomalies.append(MetricAnomaly(
                    service=service,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    anomaly_score=score,
                    contributing_metrics=contributing_metrics,
                ))

        scoring_ms = (time.time() - scoring_start) * 1000
        total_ms = (time.time() - t_start) * 1000

        logger.info(
            f"MetricDetector: scored {len(metrics)} services, "
            f"found {len(anomalies)} anomalies "
            f"(scoring: {scoring_ms:.1f}ms, total: {total_ms:.1f}ms)"
        )

        # ── LSTM/TCN temporal detection (2nd layer) ─────────────────────────
        # Runs alongside IsolationForest to catch gradual drift & temporal patterns.
        # LSTM anomalies for services NOT caught by IsolationForest are added.
        # For services caught by both, the max severity is used.
        if self._lstm is not None and metrics:
            lstm_t0 = time.time()
            # Build per-service metric dicts for LSTM (uses raw values, not feature vectors)
            lstm_anomalies = self._lstm.infer_all(metrics)
            existing_services = {a.service for a in anomalies}
            for la in lstm_anomalies:
                if la.service not in existing_services:
                    # LSTM caught something IsolationForest missed — add as new anomaly
                    anomalies.append(MetricAnomaly(
                        service=la.service,
                        anomaly_type=la.anomaly_type,
                        severity=la.severity,
                        anomaly_score=-la.degradation_probability,
                        contributing_metrics={
                            f: {"lstm_error": v}
                            for f, v in la.contributing_features.items()
                        },
                    ))
                else:
                    # Boost existing anomaly severity with LSTM confirmation
                    for a in anomalies:
                        if a.service == la.service:
                            a.severity = min(1.0, max(a.severity, la.severity * 0.9))
                            break
            lstm_ms = (time.time() - lstm_t0) * 1000
            logger.info(f"  LSTM/TCN: {len(lstm_anomalies)} temporal anomalies in {lstm_ms:.1f}ms")

        # Sort by severity
        anomalies.sort(key=lambda a: a.severity, reverse=True)
        return anomalies

    def get_stats(self) -> dict:
        """Return detector statistics including LSTM/TCN layer."""
        lstm_stats = self._lstm.get_stats() if self._lstm else {}
        return {
            "cycle_count": self._cycle_count,
            "services_tracked": len(self._history),
            "models_trained": len(self._models),
            "history_sizes": {s: len(h) for s, h in self._history.items()},
            "lstm_services_tracked": lstm_stats.get("services_tracked", 0),
            "lstm_total_inferences": lstm_stats.get("total_inferences", 0),
            "lstm_model_type": lstm_stats.get("model_type", "none"),
        }


# ─── Standalone Testing ──────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import os

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s")

    prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    detector = MetricAnomalyDetector(prometheus_url=prometheus_url)

    # Test with synthetic metrics — simulate a service with normal behavior
    # then inject anomalous metrics
    print("\n=== Training with synthetic normal data ===")
    np.random.seed(42)
    normal_metrics_base = {
        "request_rate": 50.0, "p99_latency_ms": 120.0, "p50_latency_ms": 45.0,
        "error_rate": 0.01, "cpu_percent": 25.0, "memory_percent": 40.0,
    }

    # Feed normal data to build baseline
    for i in range(15):
        synthetic = {
            "frontend": {
                k: v + np.random.normal(0, v * 0.1)
                for k, v in normal_metrics_base.items()
            },
            "cartservice": {
                k: v * 0.8 + np.random.normal(0, v * 0.1)
                for k, v in normal_metrics_base.items()
            },
        }
        anomalies = detector.detect(synthetic)

    # Now inject anomalous metrics
    print("\n=== Injecting anomalous metrics ===")
    anomalous_metrics = {
        "frontend": {
            "request_rate": 50.0, "p99_latency_ms": 120.0, "p50_latency_ms": 45.0,
            "error_rate": 0.01, "cpu_percent": 25.0, "memory_percent": 40.0,
        },
        "cartservice": {
            "request_rate": 5.0,       # Dropped!
            "p99_latency_ms": 2500.0,  # 20x spike!
            "p50_latency_ms": 800.0,   # Spiked!
            "error_rate": 0.45,        # 45% errors!
            "cpu_percent": 95.0,       # CPU saturated!
            "memory_percent": 88.0,    # Memory high!
        },
    }

    t0 = time.time()
    anomalies = detector.detect(anomalous_metrics)
    elapsed = (time.time() - t0) * 1000
    print(f"\nDetection completed in {elapsed:.1f}ms")
    print(f"Found {len(anomalies)} anomalies:\n")

    for a in anomalies:
        print(json.dumps(a.to_dict(), indent=2))
        print()

    print(f"Stats: {json.dumps(detector.get_stats(), indent=2)}")

    # Test with Prometheus (if available)
    print("\n=== Testing with Prometheus ===")
    try:
        resp = requests.get(f"{prometheus_url}/api/v1/query", params={"query": "up"}, timeout=2)
        resp.raise_for_status()
        anomalies = detector.detect()
        print(f"Found {len(anomalies)} anomalies from Prometheus")
        for a in anomalies:
            print(json.dumps(a.to_dict(), indent=2))
    except Exception as e:
        print(f"Prometheus not available: {e}")

# Usage:
# python metric_detector.py                                    # Run standalone test
# PROMETHEUS_URL=http://localhost:9090 python metric_detector.py  # Test with local Prometheus
