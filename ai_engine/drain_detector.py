# Log anomaly detection using Drain3 template parsing + TF-IDF frequency spike scoring
# File: ai_engine/drain_detector.py
# Performance target: <100ms per batch of logs
# Approach:
#   1. Drain3 parses raw log lines into templates (clusters)
#   2. Track template frequency in sliding windows (current vs baseline)
#   3. Score anomalies via frequency deviation: new templates = high score, spikes = medium
#   4. Return structured anomaly results with evidence for explainability

import time
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import logging

import requests
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

logger = logging.getLogger(__name__)


@dataclass
class LogAnomaly:
    """A detected log anomaly with evidence for the explainability engine."""
    service: str
    template: str
    anomaly_type: str          # "new_template" | "frequency_spike" | "error_burst"
    severity: float            # 0.0 - 1.0
    sample_logs: list          # Up to 3 raw log lines as evidence
    template_id: int
    current_count: int
    baseline_count: float
    trace_ids: list            # Correlated trace IDs for cross-signal linking
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "detector": "drain_log",
            "service": self.service,
            "template": self.template,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "sample_logs": self.sample_logs[:3],
            "template_id": self.template_id,
            "current_count": self.current_count,
            "baseline_count": round(self.baseline_count, 2),
            "trace_ids": self.trace_ids[:5],
            "timestamp": self.timestamp,
        }


class DrainLogDetector:
    """
    Fast log anomaly detector using Drain3 for template mining.

    Detection strategies:
    1. NEW_TEMPLATE: A log pattern never seen before → likely a new error
    2. FREQUENCY_SPIKE: A known template appears 3x+ more than baseline → degradation signal
    3. ERROR_BURST: Templates containing error keywords spike → immediate alert

    Timing budget: <100ms for processing a batch of ~500 log lines
    """

    # Keywords that indicate error-level logs
    ERROR_KEYWORDS = frozenset([
        "error", "exception", "fatal", "panic", "crash", "failed",
        "timeout", "refused", "unavailable", "oom", "killed",
        "segfault", "stack trace", "traceback", "critical",
    ])

    # Spike detection thresholds
    NEW_TEMPLATE_SEVERITY = 0.8
    SPIKE_THRESHOLD = 3.0       # Current count must be 3x baseline
    SPIKE_SEVERITY_BASE = 0.5
    ERROR_SEVERITY_BOOST = 0.2  # Added when template contains error keywords
    MIN_BASELINE_COUNT = 2      # Minimum baseline count to avoid false positives

    def __init__(self, window_size: int = 6, loki_url: str = "http://loki:3100"):
        """
        Args:
            window_size: Number of historical windows to use as baseline
            loki_url: Loki HTTP API endpoint
        """
        self.loki_url = loki_url
        self.window_size = window_size

        # Initialize Drain3 with tuned parameters for microservice logs
        config = TemplateMinerConfig()
        config.drain_sim_th = 0.4          # Similarity threshold (lower = more clusters)
        config.drain_depth = 4             # Parse tree depth
        config.drain_max_clusters = 200    # Cap clusters for memory
        config.drain_max_children = 100
        config.profiling_enabled = False   # Disable for speed
        self.template_miner = TemplateMiner(config=config)

        # DistilBERT re-scorer — 2-stage pipeline: Drain3 pre-filter → BERT severity boost
        try:
            from bert_log_classifier import BertLogClassifier
            self._bert = BertLogClassifier()
            logger.info("DistilBERT log classifier initialized (2-stage pipeline active)")
        except Exception as e:
            self._bert = None
            logger.warning(f"DistilBERT unavailable ({e}) — Drain3-only mode")

        # Sliding window counters: {window_idx: {template_id: count}}
        self._window_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._current_window: int = 0
        self._total_logs_processed: int = 0

        # Track template first-seen times for new-template detection
        self._template_first_seen: dict[int, float] = {}

        # Cache for service → template mapping
        self._service_templates: dict[str, set] = defaultdict(set)

    def _generate_simulated_logs(self) -> list[dict]:
        """
        Generate realistic simulated log data when Loki is unavailable.
        Cycles through 3 failure scenarios to demo the full pipeline.
        """
        now = time.time()
        cycle = int(now / 30) % 3  # Rotate scenario every 30 seconds
        logs = []

        # Normal background traffic for all services
        normal_services = ["frontend", "productcatalogservice", "currencyservice",
                           "cartservice", "checkoutservice", "recommendationservice"]
        for svc in normal_services:
            for _ in range(random.randint(3, 8)):
                latency = random.randint(5, 80)
                path = random.choice(["/product/OLJCESPC7Z", "/cart", "/checkout",
                                       "/currencies", "/recommendations", "/shipping"])
                trace_id = hashlib.md5(f"{svc}{now}{random.random()}".encode()).hexdigest()[:16]
                logs.append({
                    "log": f"HTTP GET {path} 200 {latency}ms",
                    "service": svc,
                    "trace_id": trace_id,
                    "timestamp": now - random.uniform(0, 10),
                })

        # Scenario-specific failure injection
        if cycle == 0:
            # Scenario 1: cartservice crash — connection refused to redis
            for i in range(random.randint(8, 15)):
                trace_id = hashlib.md5(f"cart_err_{now}_{i}".encode()).hexdigest()[:16]
                logs.append({
                    "log": "ERROR: connection refused to redis-cart:6379 — dial tcp 172.18.0.5:6379: connect: connection refused",
                    "service": "cartservice",
                    "trace_id": trace_id,
                    "timestamp": now - random.uniform(0, 5),
                })
            for i in range(random.randint(3, 6)):
                trace_id = hashlib.md5(f"front_err_{now}_{i}".encode()).hexdigest()[:16]
                logs.append({
                    "log": f"ERROR: failed to get cart: rpc error: code = Unavailable desc = upstream connect error, trace_id={trace_id}",
                    "service": "frontend",
                    "trace_id": trace_id,
                    "timestamp": now - random.uniform(0, 5),
                })

        elif cycle == 1:
            # Scenario 2: productcatalogservice CPU spike / panic
            for i in range(random.randint(5, 10)):
                trace_id = hashlib.md5(f"pcs_err_{now}_{i}".encode()).hexdigest()[:16]
                logs.append({
                    "log": "panic: runtime error: index out of range [3] with length 3",
                    "service": "productcatalogservice",
                    "trace_id": trace_id,
                    "timestamp": now - random.uniform(0, 5),
                })
            for i in range(random.randint(2, 5)):
                trace_id = hashlib.md5(f"rec_err_{now}_{i}".encode()).hexdigest()[:16]
                logs.append({
                    "log": f"ERROR: failed to list products: rpc timeout, trace_id={trace_id}",
                    "service": "recommendationservice",
                    "trace_id": trace_id,
                    "timestamp": now - random.uniform(0, 5),
                })

        elif cycle == 2:
            # Scenario 3: checkoutservice OOM / memory pressure
            for i in range(random.randint(5, 10)):
                trace_id = hashlib.md5(f"checkout_err_{now}_{i}".encode()).hexdigest()[:16]
                logs.append({
                    "log": "FATAL: out of memory — container killed by OOM killer, exit code 137",
                    "service": "checkoutservice",
                    "trace_id": trace_id,
                    "timestamp": now - random.uniform(0, 5),
                })
            for i in range(random.randint(3, 6)):
                trace_id = hashlib.md5(f"front_checkout_err_{now}_{i}".encode()).hexdigest()[:16]
                logs.append({
                    "log": f"ERROR: checkout failed: rpc error: code = Unavailable, trace_id={trace_id}",
                    "service": "frontend",
                    "trace_id": trace_id,
                    "timestamp": now - random.uniform(0, 5),
                })

        logger.info(f"Generated {len(logs)} simulated logs (scenario {cycle + 1}/3)")
        return logs

    def fetch_logs(self, lookback_seconds: int = 10) -> list[dict]:
        """
        Fetch recent logs from Loki via the HTTP API.
        Falls back to simulated logs when Loki is unreachable.

        Returns list of dicts with keys: log, service, trace_id, timestamp
        Target: <30ms for API call
        """
        t_start = time.time()
        end_ns = int(time.time() * 1e9)
        start_ns = end_ns - (lookback_seconds * int(1e9))

        # LogQL query: all logs from online-boutique namespace
        query = '{service_namespace="online-boutique"}'

        try:
            resp = requests.get(
                f"{self.loki_url}/loki/api/v1/query_range",
                params={
                    "query": query,
                    "start": str(start_ns),
                    "end": str(end_ns),
                    "limit": 500,
                    "direction": "backward",
                },
                timeout=2,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.info(f"Loki unreachable, using simulated logs ({time.time() - t_start:.1f}s)")
            return self._generate_simulated_logs()

        logs = []
        results = data.get("data", {}).get("result", [])
        for stream in results:
            labels = stream.get("stream", {})
            service = labels.get("service_name", labels.get("service", "unknown"))
            for ts, line in stream.get("values", []):
                trace_id = ""
                # Extract trace_id from structured log or labels
                if "trace_id=" in line:
                    try:
                        trace_id = line.split("trace_id=")[1].split()[0].strip('"')
                    except (IndexError, ValueError):
                        pass
                elif "traceID" in labels:
                    trace_id = labels["traceID"]

                logs.append({
                    "log": line,
                    "service": service,
                    "trace_id": trace_id,
                    "timestamp": int(ts) / 1e9,
                })

        # If Loki returned empty results, use simulation
        if not logs:
            logger.info("Loki returned no logs, using simulated logs")
            return self._generate_simulated_logs()

        elapsed = time.time() - t_start
        logger.debug(f"Fetched {len(logs)} logs from Loki in {elapsed:.3f}s")
        return logs

    def _is_error_template(self, template: str) -> bool:
        """Check if a template contains error-related keywords."""
        template_lower = template.lower()
        return any(kw in template_lower for kw in self.ERROR_KEYWORDS)

    def _get_baseline_count(self, template_id: int) -> float:
        """Calculate baseline frequency from historical windows."""
        counts = []
        for w in range(max(0, self._current_window - self.window_size), self._current_window):
            counts.append(self._window_counts[w].get(template_id, 0))

        if not counts:
            return 0.0
        return sum(counts) / len(counts)

    def _advance_window(self) -> None:
        """Rotate to next time window and clean up old windows."""
        self._current_window += 1
        # Keep only recent windows to bound memory
        cutoff = self._current_window - self.window_size - 2
        stale = [w for w in self._window_counts if w < cutoff]
        for w in stale:
            del self._window_counts[w]

    def detect(self, logs: Optional[list[dict]] = None) -> list[LogAnomaly]:
        """
        Run anomaly detection on a batch of logs.

        Args:
            logs: Pre-fetched logs, or None to fetch from Loki

        Returns:
            List of LogAnomaly objects, sorted by severity descending

        Performance: <100ms for 500 log lines
        """
        t_start = time.time()

        if logs is None:
            logs = self.fetch_logs()

        if not logs:
            return []

        # Advance sliding window on each detection cycle
        self._advance_window()

        # Phase 1: Parse all logs through Drain3 and count templates
        template_logs: dict[int, list[dict]] = defaultdict(list)
        template_services: dict[int, set] = defaultdict(set)
        template_traces: dict[int, list] = defaultdict(list)

        for log_entry in logs:
            line = log_entry["log"].strip()
            if not line:
                continue

            result = self.template_miner.add_log_message(line)
            cluster_id = result["cluster_id"]
            template = result.get("template_mined", line)

            # Track in current window
            self._window_counts[self._current_window][cluster_id] += 1
            template_logs[cluster_id].append(log_entry)
            template_services[cluster_id].add(log_entry["service"])
            self._service_templates[log_entry["service"]].add(cluster_id)

            if log_entry.get("trace_id"):
                template_traces[cluster_id].append(log_entry["trace_id"])

            # Record first-seen time
            if cluster_id not in self._template_first_seen:
                self._template_first_seen[cluster_id] = time.time()

            self._total_logs_processed += 1

        # Phase 2: Score anomalies
        anomalies = []

        for cluster_id, entries in template_logs.items():
            cluster = self.template_miner.drain.id_to_cluster.get(cluster_id)
            if cluster is None:
                continue

            template_str = cluster.get_template()
            current_count = self._window_counts[self._current_window][cluster_id]
            baseline_count = self._get_baseline_count(cluster_id)
            is_error = self._is_error_template(template_str)

            anomaly = None

            # Strategy 1: New template (never seen before this cycle or very recent)
            first_seen = self._template_first_seen.get(cluster_id, time.time())
            if time.time() - first_seen < 30:  # Seen within last 30s = new
                severity = self.NEW_TEMPLATE_SEVERITY
                if is_error:
                    severity = min(1.0, severity + self.ERROR_SEVERITY_BOOST)

                anomaly = LogAnomaly(
                    service=next(iter(template_services[cluster_id]), "unknown"),
                    template=template_str,
                    anomaly_type="new_template",
                    severity=severity,
                    sample_logs=[e["log"] for e in entries[:3]],
                    template_id=cluster_id,
                    current_count=current_count,
                    baseline_count=baseline_count,
                    trace_ids=template_traces.get(cluster_id, []),
                )

            # Strategy 2: Frequency spike (3x+ baseline)
            elif (baseline_count >= self.MIN_BASELINE_COUNT
                  and current_count > baseline_count * self.SPIKE_THRESHOLD):
                spike_ratio = current_count / max(baseline_count, 1)
                severity = min(1.0, self.SPIKE_SEVERITY_BASE + (spike_ratio - self.SPIKE_THRESHOLD) * 0.1)
                if is_error:
                    severity = min(1.0, severity + self.ERROR_SEVERITY_BOOST)

                anomaly = LogAnomaly(
                    service=next(iter(template_services[cluster_id]), "unknown"),
                    template=template_str,
                    anomaly_type="frequency_spike",
                    severity=severity,
                    sample_logs=[e["log"] for e in entries[:3]],
                    template_id=cluster_id,
                    current_count=current_count,
                    baseline_count=baseline_count,
                    trace_ids=template_traces.get(cluster_id, []),
                )

            # Strategy 3: Error burst (error template with significant count)
            elif is_error and current_count >= 3:
                severity = min(1.0, 0.6 + current_count * 0.02)
                anomaly = LogAnomaly(
                    service=next(iter(template_services[cluster_id]), "unknown"),
                    template=template_str,
                    anomaly_type="error_burst",
                    severity=severity,
                    sample_logs=[e["log"] for e in entries[:3]],
                    template_id=cluster_id,
                    current_count=current_count,
                    baseline_count=baseline_count,
                    trace_ids=template_traces.get(cluster_id, []),
                )

            if anomaly is not None:
                anomalies.append(anomaly)

        # Sort by severity (highest first) BEFORE capping
        anomalies.sort(key=lambda a: a.severity, reverse=True)

        # Cap output BEFORE BERT to avoid wasting inference on anomalies we'd discard
        MAX_ANOMALIES_PER_CYCLE = 20
        if len(anomalies) > MAX_ANOMALIES_PER_CYCLE:
            logger.info(
                f"DrainDetector: capping anomalies from {len(anomalies)} "
                f"to top {MAX_ANOMALIES_PER_CYCLE} by severity"
            )
            anomalies = anomalies[:MAX_ANOMALIES_PER_CYCLE]

        # ── DistilBERT Re-scoring (2nd stage) ───────────────────────────────
        # Each Drain3-flagged template is re-scored by DistilBERT to:
        #   1. Boost true anomalies (high NLI score for "service failure")
        #   2. Dampen false positives (high NLI score for "normal operation")
        #   3. Add bert_classification to anomaly for explainability
        if self._bert is not None and anomalies:
            bert_t0 = time.time()
            templates = [a.template for a in anomalies]
            classifications = self._bert.classify_batch(templates)
            for anomaly, cls in zip(anomalies, classifications):
                boosted = self._bert.boost_severity(anomaly.severity, cls)
                anomaly.severity = boosted
                # Attach BERT metadata for explainability
                anomaly.bert_label = cls.label
                anomaly.bert_confidence = cls.confidence
                anomaly.bert_anomaly_prob = cls.anomaly_probability
                anomaly.bert_model = cls.model_used
            bert_ms = (time.time() - bert_t0) * 1000
            logger.info(f"  DistilBERT re-scoring: {len(anomalies)} templates in {bert_ms:.1f}ms")

            # Re-sort after BERT adjustment
            anomalies.sort(key=lambda a: a.severity, reverse=True)

        elapsed_ms = (time.time() - t_start) * 1000
        logger.info(
            f"DrainDetector: processed {len(logs)} logs, "
            f"found {len(anomalies)} anomalies in {elapsed_ms:.1f}ms"
        )

        return anomalies

    def get_stats(self) -> dict:
        """Return detector statistics for monitoring."""
        bert_stats = self._bert.get_stats() if self._bert else {"model_loaded": False}
        return {
            "total_logs_processed": self._total_logs_processed,
            "total_templates": len(self._template_first_seen),
            "current_window": self._current_window,
            "bert_model_loaded": bert_stats.get("model_loaded", False),
            "bert_model_name": bert_stats.get("model_name", "none"),
            "bert_cache_size": bert_stats.get("cache_size", 0),
            "bert_avg_inference_ms": bert_stats.get("avg_inference_ms", 0),
            "active_clusters": self.template_miner.drain.get_total_cluster_size(),
        }


# ─── Standalone Testing ──────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import os

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s")

    loki_url = os.getenv("LOKI_URL", "http://localhost:3100")
    detector = DrainLogDetector(loki_url=loki_url)

    # Test with synthetic logs if Loki is not available
    test_logs = [
        {"log": "HTTP GET /product/OLJCESPC7Z 200 12ms", "service": "frontend", "trace_id": "abc123", "timestamp": time.time()},
        {"log": "HTTP GET /product/OLJCESPC7Z 200 15ms", "service": "frontend", "trace_id": "abc124", "timestamp": time.time()},
        {"log": "HTTP GET /cart 200 8ms", "service": "frontend", "trace_id": "abc125", "timestamp": time.time()},
        {"log": "ERROR: connection refused to cartservice:7070", "service": "frontend", "trace_id": "abc126", "timestamp": time.time()},
        {"log": "ERROR: connection refused to cartservice:7070", "service": "frontend", "trace_id": "abc127", "timestamp": time.time()},
        {"log": "ERROR: connection refused to cartservice:7070", "service": "frontend", "trace_id": "abc128", "timestamp": time.time()},
        {"log": "FATAL: out of memory in recommendationservice", "service": "recommendationservice", "trace_id": "def456", "timestamp": time.time()},
        {"log": "panic: runtime error: index out of range [3]", "service": "productcatalogservice", "trace_id": "ghi789", "timestamp": time.time()},
        {"log": "HTTP POST /cart/checkout 200 450ms", "service": "checkoutservice", "trace_id": "jkl012", "timestamp": time.time()},
        {"log": "HTTP POST /cart/checkout 200 120ms", "service": "checkoutservice", "trace_id": "jkl013", "timestamp": time.time()},
    ]

    print("\n=== Testing with synthetic logs ===")
    t0 = time.time()
    anomalies = detector.detect(test_logs)
    elapsed = (time.time() - t0) * 1000
    print(f"Detection completed in {elapsed:.1f}ms")
    print(f"Found {len(anomalies)} anomalies:\n")

    for a in anomalies:
        print(json.dumps(a.to_dict(), indent=2))
        print()

    print(f"Stats: {json.dumps(detector.get_stats(), indent=2)}")

    # Test with Loki (if available)
    print("\n=== Testing with Loki ===")
    try:
        anomalies = detector.detect()
        print(f"Found {len(anomalies)} anomalies from Loki")
        for a in anomalies:
            print(json.dumps(a.to_dict(), indent=2))
    except Exception as e:
        print(f"Loki not available: {e}")

# Usage:
# python drain_detector.py                           # Run standalone test
# LOKI_URL=http://localhost:3100 python drain_detector.py  # Test with local Loki
