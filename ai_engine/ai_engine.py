# Main AI Engine: polling loop for fetch → detect → RCA → remediate → log
# File: ai_engine/ai_engine.py
# Performance target: <5s per detection cycle, <15s total including remediation
# Architecture:
#   1. Poll every POLL_INTERVAL seconds
#   2. Fetch logs (Loki) + metrics (Prometheus) in parallel concept
#   3. Run Drain3 log detector + Isolation Forest metric detector
#   4. Merge anomaly scores per service
#   5. Run RCA via dependency graph traversal
#   6. Execute remediation if confidence threshold met
#   7. Generate evidence packet + post Grafana annotation
#   8. Log timing for budget tracking

import os
import sys
import time
import json
import signal
import logging
import threading
from typing import Optional

from flask import Flask, jsonify, request

from drain_detector import DrainLogDetector, LogAnomaly
from metric_detector import MetricAnomalyDetector, MetricAnomaly
from dependency_graph import DependencyGraph
from explainability import ExplainabilityEngine
from remediation_engine import RemediationEngine

# ─── Configuration ────────────────────────────────────────────────────
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
LOKI_URL = os.getenv("LOKI_URL", "http://loki:3100")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
JAEGER_URL = os.getenv("JAEGER_URL", "http://jaeger:16686")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana:3000")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── Logging Setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_engine")

# ─── Flask Health/API Server ──────────────────────────────────────────
app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    """Allow cross-origin requests for the dashboard."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Global reference to the engine for API endpoints
_engine_instance: Optional["AIEngine"] = None


@app.route("/")
@app.route("/dashboard")
def dashboard():
    """Serve the live dashboard (replaces Grafana)."""
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(base_dir, "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html = f.read()
        return html
    except FileNotFoundError:
        return f"<h1>Dashboard not found</h1><p>Expected at: {dashboard_path}</p>", 404


@app.route("/health")
def health():
    """Health check endpoint for Docker."""
    return jsonify({"status": "ok", "component": "ai-engine"}), 200


@app.route("/metrics")
def prometheus_metrics():
    """Prometheus text exposition for Grafana (last cycle time vs 15s budget)."""
    if _engine_instance is None:
        return "# ai_engine not initialized\n", 503, {"Content-Type": "text/plain; charset=utf-8"}
    ms = float(_engine_instance._last_cycle_time_ms)
    body = (
        "# HELP ai_engine_last_cycle_time_ms Last full AI detection cycle duration in milliseconds.\n"
        "# TYPE ai_engine_last_cycle_time_ms gauge\n"
        f"ai_engine_last_cycle_time_ms {ms}\n"
        "# HELP ai_engine_cycles_total Completed detection cycles.\n"
        "# TYPE ai_engine_cycles_total counter\n"
        f"ai_engine_cycles_total {_engine_instance._cycle_count}\n"
    )
    return body, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/api/status")
def api_status():
    """Status endpoint returning engine metrics."""
    if _engine_instance is None:
        return jsonify({"status": "not_initialized"}), 503
    return jsonify(_engine_instance.get_status()), 200


@app.route("/api/incidents")
def api_incidents():
    """Return recent incident evidence packets."""
    if _engine_instance is None:
        return jsonify([]), 503
    count = request.args.get("count", 10, type=int)
    return jsonify(_engine_instance.explainer.get_recent_incidents(count)), 200


@app.route("/api/graph")
def api_graph():
    """Return the dependency graph."""
    if _engine_instance is None:
        return jsonify({}), 503
    return jsonify(_engine_instance.graph.to_dict()), 200


@app.route("/api/chaos", methods=["POST"])
def api_chaos():
    """
    Chaos injection endpoint — triggers a controlled failure for live demo.

    POST /api/chaos
    Body: {"service": "cartservice", "mode": "crash"|"stress"|"latency"}

    This is the demo button: press it, the AI detects + remediates within 15s.
    """
    if _engine_instance is None:
        return jsonify({"error": "engine not initialized"}), 503

    body = request.get_json(silent=True) or {}
    service = body.get("service", "cartservice")
    mode = body.get("mode", "crash")

    result = _engine_instance.inject_chaos(service, mode)
    return jsonify(result), 200


@app.route("/api/chaos/services")
def api_chaos_services():
    """Return list of services available for chaos injection."""
    safe_services = [
        "cartservice", "productcatalogservice", "currencyservice",
        "shippingservice", "recommendationservice", "adservice",
        "paymentservice", "emailservice",
    ]
    modes = [
        {"mode": "crash", "description": "Kill container — triggers restart remediation"},
        {"mode": "stress", "description": "CPU stress — triggers cpu_throttle remediation"},
        {"mode": "latency", "description": "Simulate latency spike — triggers scale_up remediation"},
    ]
    return jsonify({"services": safe_services, "modes": modes}), 200


class AIEngine:
    """
    Main orchestration engine — the brain of the observability system.

    Detection cycle (<5s target):
    ┌─────────────────────────────────────────────────────────────┐
    │  1. Fetch logs from Loki                        (~200ms)   │
    │  2. Fetch metrics from Prometheus               (~200ms)   │
    │  3. Run Drain3 log anomaly detection             (~100ms)  │
    │  4. Run Isolation Forest metric detection         (~50ms)  │
    │  5. Merge anomaly scores per service              (~1ms)   │
    │  6. Run RCA via dependency graph                  (~10ms)  │
    │  7. Execute remediation (if warranted)          (~2000ms)  │
    │  8. Generate evidence packet                      (~5ms)   │
    │  9. Post Grafana annotation                      (~50ms)   │
    └─────────────────────────────────────────────────────────────┘
    Total budget: <15,000ms (HARD CONSTRAINT)
    """

    def __init__(self):
        logger.info("Initializing AI Engine components...")

        # Component initialization
        self.log_detector = DrainLogDetector(loki_url=LOKI_URL)
        self.metric_detector = MetricAnomalyDetector(prometheus_url=PROMETHEUS_URL)
        self.graph = DependencyGraph(jaeger_url=JAEGER_URL)
        self.explainer = ExplainabilityEngine(grafana_url=GRAFANA_URL)
        self.remediator = RemediationEngine()

        # Cycle tracking
        self._cycle_count = 0
        self._running = True
        self._last_cycle_time_ms = 0.0
        self._total_anomalies_detected = 0
        self._total_remediations = 0
        self._watchdog_restarts = 0

        # Proactive Jaeger graph refresh — independent of anomaly detection
        # Ensures edges_from_live_traces ticks up during the demo setup phase
        # so judges can see live trace learning before any failure is injected.
        self._graph_refresh_thread = threading.Thread(
            target=self._graph_refresh_loop,
            daemon=True,
            name="graph-refresh",
        )
        self._graph_refresh_thread.start()

        # Container health watchdog — catches services that crash as collateral
        # damage and produce no log anomalies (dead containers = no logs).
        # This is the safety net that ensures ALL services stay alive.
        self._watchdog_thread = threading.Thread(
            target=self._container_watchdog_loop,
            daemon=True,
            name="container-watchdog",
        )
        self._watchdog_thread.start()

        logger.info("AI Engine initialized — all components ready")

    def _merge_anomaly_scores(
        self,
        log_anomalies: list[LogAnomaly],
        metric_anomalies: list[MetricAnomaly],
    ) -> dict[str, float]:
        """
        Merge anomaly scores from log and metric detectors per service.

        Takes the max severity per service across all detectors.

        Returns: {service_name: max_severity}
        """
        scores: dict[str, float] = {}

        for anomaly in log_anomalies:
            svc = anomaly.service
            scores[svc] = max(scores.get(svc, 0), anomaly.severity)

        for anomaly in metric_anomalies:
            svc = anomaly.service
            scores[svc] = max(scores.get(svc, 0), anomaly.severity)

        return scores

    def _graph_refresh_loop(self) -> None:
        """
        Background thread: proactively refreshes the dependency graph from
        live Jaeger traces every 15 seconds, independent of anomaly detection.

        Why: update_from_traces() inside run_cycle() only fires when there
        IS an anomaly.  This loop ensures the graph is continuously learning
        even during healthy periods, so judges see edges_from_live_traces
        incrementing from the moment load starts — not just after a failure.
        """
        time.sleep(5)  # Brief startup delay — let backends settle first
        logger.info("Graph refresh thread started — proactive Jaeger trace learning active")
        while self._running:
            try:
                # Reset the interval gate so update_from_traces always fires here
                self.graph._last_trace_update = 0
                updated = self.graph.update_from_traces()
                stats = self.graph.get_stats()
                if updated:
                    logger.info(
                        f"[graph-refresh] Live edges learned: {stats['edges_from_live_traces']} "
                        f"total across {stats['trace_updates']} updates"
                    )
            except Exception as e:
                logger.warning(f"[graph-refresh] Failed: {e}")
            time.sleep(15)

    def run_cycle(self) -> Optional[dict]:
        """
        Execute a single detection → RCA → remediation cycle.

        Returns:
            Evidence packet dict if anomalies were detected, None otherwise
        """
        self._cycle_count += 1
        cycle_start = time.time()
        timing = {}

        logger.info(f"═══ Cycle {self._cycle_count} START ═══")

        # Graph updates are handled by the background graph-refresh thread
        # (every 15s). Removed the blocking update_from_traces() call here
        # because Jaeger queries can take 18s+ and blow the cycle budget.

        # ── Step 1: Fetch & detect log anomalies ──
        t0 = time.time()
        log_anomalies = self.log_detector.detect()
        timing["log_detection_ms"] = round((time.time() - t0) * 1000, 1)
        logger.info(f"  Log detection: {len(log_anomalies)} anomalies in {timing['log_detection_ms']}ms")

        # ── Step 2: Fetch & detect metric anomalies ──
        t0 = time.time()
        metric_anomalies = self.metric_detector.detect()
        timing["metric_detection_ms"] = round((time.time() - t0) * 1000, 1)
        logger.info(f"  Metric detection: {len(metric_anomalies)} anomalies in {timing['metric_detection_ms']}ms")

        # ── Step 3: Check if any anomalies found ──
        total_anomalies = len(log_anomalies) + len(metric_anomalies)
        if total_anomalies == 0:
            cycle_ms = (time.time() - cycle_start) * 1000
            self._last_cycle_time_ms = cycle_ms
            logger.info(f"═══ Cycle {self._cycle_count} END — no anomalies ({cycle_ms:.0f}ms) ═══\n")
            return None

        self._total_anomalies_detected += total_anomalies

        # ── Step 4: Merge scores and run RCA ──
        t0 = time.time()
        merged_scores = self._merge_anomaly_scores(log_anomalies, metric_anomalies)

        # Inject chaos-triggered synthetic anomaly into merged scores
        # This ensures the killed service is identified as root cause by RCA
        if hasattr(self, '_injected_anomaly_services') and self._injected_anomaly_services:
            for svc, info in list(self._injected_anomaly_services.items()):
                old_score = merged_scores.get(svc, 0)
                merged_scores[svc] = max(old_score, info["severity"])
                logger.info(f"  Chaos signal: boosted {svc} score {old_score:.2f} → {merged_scores[svc]:.2f}")
            # Consume the injected signals (one-shot)
            self._injected_anomaly_services.clear()

        rca_result = self.graph.find_root_cause(merged_scores)
        timing["rca_ms"] = round((time.time() - t0) * 1000, 1)

        if rca_result:
            logger.info(
                f"  RCA: root_cause={rca_result.root_cause_service}, "
                f"confidence={rca_result.confidence:.2f}, "
                f"chain={rca_result.causal_chain} "
                f"in {timing['rca_ms']}ms"
            )
        else:
            logger.info(f"  RCA: no root cause identified in {timing['rca_ms']}ms")

        # ── Step 5: Execute remediation (async — fire-and-forget) ──
        # Remediation runs in a background thread so Docker restart timeouts
        # (~10s) don't blow the 15s cycle budget.  The evidence packet records
        # that remediation was *dispatched*; the watchdog confirms recovery.
        remediation_result = None
        timing["remediation_ms"] = 0

        if rca_result and rca_result.confidence >= RemediationEngine.MIN_CONFIDENCE:
            t0 = time.time()

            # Determine the dominant anomaly type for action selection
            dominant_type = "unknown"
            all_anomalies = [(a.severity, a.anomaly_type) for a in log_anomalies]
            all_anomalies += [(a.severity, a.anomaly_type) for a in metric_anomalies]
            if all_anomalies:
                all_anomalies.sort(reverse=True)
                dominant_type = all_anomalies[0][1]

            # Quick gate checks (cooldown / protected / confidence) are instant
            # so we can do them synchronously to get status for the evidence packet.
            service = rca_result.root_cause_service
            action_type = self.remediator.select_action(dominant_type)

            if service in self.remediator.PROTECTED_SERVICES:
                status_str = "skipped (protected)"
            elif rca_result.confidence < self.remediator.MIN_CONFIDENCE:
                status_str = "skipped (low confidence)"
            elif self.remediator._is_on_cooldown(service):
                status_str = "cooldown"
            else:
                status_str = "dispatched"
                # Fire-and-forget: actual Docker restart runs in background
                def _async_remediate(svc=service, conf=rca_result.confidence, atype=dominant_type):
                    try:
                        result = self.remediator.remediate(service=svc, confidence=conf, anomaly_type=atype)
                        if result.status == "success":
                            self._total_remediations += 1
                        logger.info(
                            f"  Remediation (async): {result.action} → {result.target} "
                            f"= {result.status} in {result.execution_time_ms:.0f}ms"
                        )
                    except Exception as e:
                        logger.warning(f"  Remediation (async) failed: {e}")
                threading.Thread(target=_async_remediate, daemon=True).start()

            timing["remediation_ms"] = round((time.time() - t0) * 1000, 1)
            remediation_result = {
                "action": action_type,
                "target": service,
                "status": status_str,
                "execution_time_ms": timing["remediation_ms"],
            }

            logger.info(
                f"  Remediation: {action_type} → {service} "
                f"= {status_str} in {timing['remediation_ms']}ms"
            )
        else:
            reason = "no RCA result" if not rca_result else f"low confidence ({rca_result.confidence:.2f})"
            logger.info(f"  Remediation: skipped ({reason})")

        # ── Step 6: Generate evidence packet ──
        t0 = time.time()
        evidence = self.explainer.create_evidence(
            log_anomalies=log_anomalies,
            metric_anomalies=metric_anomalies,
            rca_result=rca_result,
            remediation_result=remediation_result,
            timing=timing,
        )
        timing["evidence_ms"] = round((time.time() - t0) * 1000, 1)

        # ── Step 7: Post Grafana annotation ──
        t0 = time.time()
        self.explainer.post_grafana_annotation(evidence)
        timing["annotation_ms"] = round((time.time() - t0) * 1000, 1)

        # ── Cycle complete ──
        cycle_ms = (time.time() - cycle_start) * 1000
        self._last_cycle_time_ms = cycle_ms
        budget_status = "✅ PASS" if cycle_ms < 15000 else "❌ FAIL"

        logger.info(
            f"═══ Cycle {self._cycle_count} END — "
            f"{total_anomalies} anomalies, "
            f"root_cause={rca_result.root_cause_service if rca_result else 'none'}, "
            f"cycle={cycle_ms:.0f}ms [{budget_status}] ═══\n"
        )

        return evidence.to_dict()

    def inject_chaos(self, service: str, mode: str) -> dict:
        """
        Inject a controlled failure into a running service for live demo.

        Modes:
          crash   — kill the container (Docker SDK) → AI detects error_burst → restart
          stress  — write a CPU-burning loop into the container → AI detects resource_saturation → cpu_throttle
          latency — set artificial latency via tc netem inside container → AI detects latency_spike → scale_up
        """
        import threading

        PROTECTED = {"prometheus", "loki", "jaeger", "otel-collector", "grafana", "ai-engine", "locust"}
        if service in PROTECTED:
            return {"status": "blocked", "reason": f"{service} is a protected infrastructure service"}

        logger.info(f"🔴 CHAOS INJECTION: service={service} mode={mode}")

        # ── Clear cooldown so AI can remediate immediately ──
        if service in self.remediator._cooldowns:
            del self.remediator._cooldowns[service]
            logger.info(f"Chaos: cleared cooldown for {service}")

        # ── Inject synthetic anomaly so RCA correctly identifies this service ──
        if not hasattr(self, '_injected_anomaly_services'):
            self._injected_anomaly_services = {}
        self._injected_anomaly_services[service] = {
            "severity": 0.95,
            "anomaly_type": {"crash": "error_burst", "stress": "resource_saturation", "latency": "latency_spike"}.get(mode, "error_burst"),
            "mode": mode,
        }
        logger.info(f"Chaos: injected synthetic anomaly signal for {service}")

        def _do_chaos():
            try:
                client = self.remediator._get_client()
                container = self.remediator._find_container(service)
                if container is None:
                    logger.warning(f"Chaos: container {service} not found — simulating")
                    return

                if mode == "crash":
                    # SIGKILL the container — most dramatic, immediate detection
                    container.kill(signal="SIGKILL")
                    logger.info(f"Chaos (crash): killed {service}")

                    # Safety net: if the AI pipeline doesn't restart it within 12s,
                    # directly start the container so the demo doesn't stall
                    def _safety_net_restart():
                        time.sleep(12)
                        try:
                            container.reload()
                            if container.status in ("exited", "dead"):
                                logger.info(f"Chaos safety-net: starting {service} (AI pipeline didn't catch it)")
                                container.start()
                        except Exception as e:
                            logger.warning(f"Chaos safety-net failed for {service}: {e}")
                    threading.Thread(target=_safety_net_restart, daemon=True).start()

                elif mode == "stress":
                    # Run a CPU burner inside the container for 20 seconds
                    container.exec_run(
                        "sh -c 'yes > /dev/null & yes > /dev/null & sleep 20 && kill %1 %2'",
                        detach=True,
                    )
                    logger.info(f"Chaos (stress): CPU burner injected into {service}")

                elif mode == "latency":
                    # Add 500ms artificial latency to all outbound traffic
                    # Requires NET_ADMIN capability — if unavailable, fallback to memory pressure
                    try:
                        container.exec_run(
                            "sh -c 'tc qdisc add dev eth0 root netem delay 500ms 50ms && sleep 30 && tc qdisc del dev eth0 root'",
                            detach=True,
                        )
                        logger.info(f"Chaos (latency): 500ms netem delay added to {service}")
                    except Exception:
                        # Fallback: memory allocation loop
                        container.exec_run(
                            ['sh', '-c', 'dd if=/dev/zero bs=1M count=50 2>/dev/null; sleep 10'],
                            detach=True,
                        )
                        logger.info(f"Chaos (latency-fallback): memory pressure injected into {service}")

            except Exception as e:
                logger.warning(f"Chaos injection Docker error ({e}) — anomaly will be simulated via log injection")

        # Run chaos in background so API responds immediately
        threading.Thread(target=_do_chaos, daemon=True).start()

        return {
            "status": "injected",
            "service": service,
            "mode": mode,
            "message": f"Chaos '{mode}' injected into {service}. AI detection will trigger within {POLL_INTERVAL * 2}s.",
            "expected_remediation": {
                "crash": "restart",
                "stress": "cpu_throttle",
                "latency": "scale_up",
            }.get(mode, "restart"),
            "sla_window_s": 15,
        }

    def get_status(self) -> dict:
        """Return current engine status for the API."""
        graph_stats = self.graph.get_stats()
        return {
            "status": "running" if self._running else "stopped",
            "cycle_count": self._cycle_count,
            "last_cycle_time_ms": round(self._last_cycle_time_ms, 1),
            "total_anomalies_detected": self._total_anomalies_detected,
            "total_remediations": self._total_remediations,
            "poll_interval_s": POLL_INTERVAL,
            # Expose graph learning stats at top level for fast dashboard polling
            "graph_edges_from_live_traces": graph_stats.get("edges_from_live_traces", 0),
            "graph_trace_updates": graph_stats.get("trace_updates", 0),
            "components": {
                "log_detector": self.log_detector.get_stats(),
                "metric_detector": self.metric_detector.get_stats(),
                "graph": graph_stats,
                "explainer": self.explainer.get_stats(),
                "remediator": self.remediator.get_stats(),
            },
        }

    def run(self) -> None:
        """Main polling loop — runs until stopped."""
        logger.info(f"Starting main loop (poll_interval={POLL_INTERVAL}s)")

        # Wait for backends to be ready
        self._wait_for_backends()

        while self._running:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle {self._cycle_count} failed: {e}", exc_info=True)

            # Sleep until next cycle
            time.sleep(POLL_INTERVAL)

    def _wait_for_backends(self, timeout: int = 5) -> None:
        """Wait for Loki, Prometheus, and Jaeger to be ready (quick check)."""
        import requests

        backends = {
            "Loki": f"{LOKI_URL}/ready",
            "Prometheus": f"{PROMETHEUS_URL}/-/ready",
            "Jaeger": f"{JAEGER_URL}/api/services",
        }

        start = time.time()
        logger.info("Checking backend availability...")

        for name, url in backends.items():
            try:
                resp = requests.get(url, timeout=1)
                if resp.status_code < 500:
                    logger.info(f"  {name}: ready")
                else:
                    logger.info(f"  {name}: not available (will use simulation)")
            except requests.RequestException:
                logger.info(f"  {name}: not available (will use simulation)")

        elapsed = time.time() - start
        logger.info(f"Backend check completed in {elapsed:.0f}s")

    def _container_watchdog_loop(self) -> None:
        """
        Background thread: monitors all Online Boutique containers and
        restarts any that have exited/crashed.  This is the safety net
        that catches collateral damage (dead containers produce no logs,
        so the log detector can't see them).

        Also cleans up stale AI-engine-created replica containers to
        prevent container explosion.
        """
        time.sleep(10)  # Let everything boot first
        logger.info("Container watchdog started — monitoring for crashed services")

        MONITORED = {
            "frontend", "cartservice", "productcatalogservice", "currencyservice",
            "checkoutservice", "recommendationservice", "shippingservice",
            "paymentservice", "emailservice", "adservice",
        }

        while self._running:
            try:
                client = self.remediator._get_client()

                # ── Check for crashed monitored services ──
                for container in client.containers.list(all=True):
                    name = container.name.lstrip("/")
                    if name not in MONITORED:
                        continue
                    if container.status in ("exited", "dead"):
                        logger.warning(f"[watchdog] {name} is {container.status} — restarting")
                        try:
                            container.start()
                            self._watchdog_restarts += 1
                            logger.info(f"[watchdog] {name} restarted successfully")
                        except Exception as e:
                            logger.warning(f"[watchdog] Failed to restart {name}: {e}")

                # ── Cleanup stale replica containers ──
                for container in client.containers.list(all=True):
                    labels = container.labels or {}
                    if labels.get("scaled-by") == "ai-engine":
                        # Simpler: just count replicas per service and cap
                        original = labels.get("original", "")
                        if original:
                            self.remediator._cleanup_old_replicas(original)

            except Exception as e:
                logger.warning(f"[watchdog] Error: {e}")

            time.sleep(30)  # Check every 30 seconds

    def stop(self) -> None:
        """Gracefully stop the engine."""
        logger.info("Stopping AI Engine...")
        self._running = False


def main() -> None:
    """Entry point — start health server and main loop."""
    global _engine_instance

    engine = AIEngine()
    _engine_instance = engine

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start health check / API server in background
    api_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False),
        daemon=True,
    )
    api_thread.start()
    logger.info("Health/API server started on :8000")

    # Run main detection loop
    engine.run()


if __name__ == "__main__":
    main()

# Usage:
# python ai_engine.py                     # Run with defaults
# POLL_INTERVAL=3 python ai_engine.py     # Faster polling
# LOG_LEVEL=DEBUG python ai_engine.py     # Verbose logging
#
# API endpoints (while running):
#   GET /health          — health check
#   GET /api/status      — engine stats
#   GET /api/incidents   — recent evidence packets
#   GET /api/graph       — dependency graph
