# Causal RCA engine: trace-based dependency graph bootstrap + upstream traversal
# File: ai_engine/dependency_graph.py
# Performance target: <10ms for graph traversal
# Approach:
#   1. Bootstrap a directed dependency graph from OTel traces (Jaeger API)
#   2. Given an anomalous service, traverse UPSTREAM to find root cause
#   3. Combine anomaly scores from log + metric detectors to rank causal candidates
#   4. Return the most likely root cause service with causal evidence chain

import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# Pre-configured dependency graph for Online Boutique
# This is used as fallback when trace data is insufficient
# Direction: caller → callee (A → B means A depends on B)
DEFAULT_DEPENDENCY_GRAPH: dict[str, list[str]] = {
    "frontend": ["cartservice", "productcatalogservice", "currencyservice",
                  "recommendationservice", "checkoutservice", "shippingservice"],
    "checkoutservice": ["cartservice", "productcatalogservice", "currencyservice",
                        "shippingservice", "emailservice", "paymentservice"],
    "recommendationservice": ["productcatalogservice"],
    "cartservice": ["redis-cart"],
}


@dataclass
class RCAResult:
    """Root cause analysis result with causal evidence chain."""
    root_cause_service: str
    confidence: float              # 0.0 - 1.0
    causal_chain: list[str]        # [root_cause → ... → symptom_service]
    anomaly_scores: dict           # {service: {detector: score}}
    graph_depth: int               # How many hops from symptom to root cause
    contributing_anomalies: list   # Raw anomaly objects that contributed
    analysis_time_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "root_cause_service": self.root_cause_service,
            "confidence": round(self.confidence, 3),
            "causal_chain": self.causal_chain,
            "anomaly_scores": self.anomaly_scores,
            "graph_depth": self.graph_depth,
            "analysis_time_ms": round(self.analysis_time_ms, 2),
            "timestamp": self.timestamp,
        }


class DependencyGraph:
    """
    Directed dependency graph for causal root cause analysis.

    The graph captures caller → callee relationships:
    - An edge A → B means service A calls/depends on service B
    - When B is the root cause, symptoms propagate to A (and A's callers)

    RCA Strategy:
    1. Start from the service(s) showing anomalies
    2. Traverse UPSTREAM (follow dependencies) to find which dependency is unhealthy
    3. The deepest anomalous service in the dependency chain is the root cause
    4. Confidence is based on: anomaly severity, graph position, and cross-signal correlation
    """

    def __init__(self, jaeger_url: str = "http://jaeger:16686"):
        self.jaeger_url = jaeger_url

        # Directed graph: {caller: [callees]}
        self._graph: dict[str, set[str]] = defaultdict(set)
        # Reverse graph: {callee: [callers]}
        self._reverse_graph: dict[str, set[str]] = defaultdict(set)

        # Tracks when graph was last updated from traces
        self._last_trace_update: float = 0
        self._trace_update_interval: float = 30  # Update every 30s (aggressive for demo)
        self._edges_from_traces: int = 0          # How many edges learned from live traces
        self._trace_updates_count: int = 0        # How many successful trace updates

        # Bootstrap with known topology (must be after attribute init)
        self._load_default_graph()

    def _load_default_graph(self) -> None:
        """Load the pre-configured dependency graph."""
        for caller, callees in DEFAULT_DEPENDENCY_GRAPH.items():
            for callee in callees:
                self.add_edge(caller, callee)
        logger.info(f"Loaded default graph: {self.get_stats()}")

    def add_edge(self, caller: str, callee: str) -> None:
        """Add a dependency edge: caller → callee."""
        self._graph[caller].add(callee)
        self._reverse_graph[callee].add(caller)

    def get_dependencies(self, service: str) -> set[str]:
        """Get direct dependencies (callees) of a service."""
        return self._graph.get(service, set())

    def get_dependents(self, service: str) -> set[str]:
        """Get services that depend on (call) this service."""
        return self._reverse_graph.get(service, set())

    def get_all_services(self) -> set[str]:
        """Get all services in the graph."""
        services = set()
        for k, vs in self._graph.items():
            services.add(k)
            services.update(vs)
        for k, vs in self._reverse_graph.items():
            services.add(k)
            services.update(vs)
        return services

    def update_from_traces(self) -> bool:
        """
        Bootstrap/update dependency graph from Jaeger trace data.

        Parses recent traces to discover caller-callee relationships
        based on parent-child span relationships.

        Returns: True if graph was updated
        """
        now = time.time()
        if now - self._last_trace_update < self._trace_update_interval:
            return False

        t_start = time.time()

        try:
            # Get list of services from Jaeger
            resp = requests.get(f"{self.jaeger_url}/api/services", timeout=3)
            resp.raise_for_status()
            services = resp.json().get("data", [])

            edges_added = 0
            for service in services[:8]:  # Cap to avoid slow queries
                try:
                    trace_resp = requests.get(
                        f"{self.jaeger_url}/api/traces",
                        params={
                            "service": service,
                            "limit": 20,
                            "lookback": "5m",
                        },
                        timeout=3,
                    )
                    trace_resp.raise_for_status()
                    traces = trace_resp.json().get("data", [])

                    for trace in traces:
                        spans = trace.get("spans", [])
                        # Build span_id → service mapping
                        span_services = {}
                        for span in spans:
                            span_id = span.get("spanID", "")
                            proc_id = span.get("processID", "")
                            processes = trace.get("processes", {})
                            svc = processes.get(proc_id, {}).get("serviceName", "")
                            if span_id and svc:
                                span_services[span_id] = svc

                        # Derive edges from parent-child relationships
                        for span in spans:
                            child_svc = span_services.get(span.get("spanID", ""), "")
                            for ref in span.get("references", []):
                                if ref.get("refType") == "CHILD_OF":
                                    parent_svc = span_services.get(ref.get("spanID", ""), "")
                                    if parent_svc and child_svc and parent_svc != child_svc:
                                        if child_svc not in self._graph.get(parent_svc, set()):
                                            self.add_edge(parent_svc, child_svc)
                                            edges_added += 1

                except requests.RequestException:
                    continue

            self._last_trace_update = now
            self._edges_from_traces += edges_added
            self._trace_updates_count += 1
            elapsed = (time.time() - t_start) * 1000
            logger.info(
                f"Graph updated from live Jaeger traces: +{edges_added} new edges "
                f"({self._edges_from_traces} total learned) in {elapsed:.1f}ms"
            )
            return True

        except requests.RequestException as e:
            logger.warning(f"Failed to update graph from Jaeger: {e}")
            self._last_trace_update = now  # Don't retry immediately
            return False

    def find_root_cause(self, anomalous_services: dict[str, float],
                        max_depth: int = 5) -> Optional[RCAResult]:
        """
        Find the root cause by traversing the dependency graph.

        Algorithm:
        1. For each anomalous service, traverse its dependencies (BFS)
        2. If a dependency is ALSO anomalous, it's a stronger root cause candidate
        3. The deepest anomalous node in the dependency chain is the root cause
        4. Confidence = f(severity, depth, number of affected dependents)

        Args:
            anomalous_services: {service_name: max_severity_score}
            max_depth: Maximum BFS depth for traversal

        Returns:
            RCAResult with root cause identification, or None if no anomalies

        Performance: <10ms (graph traversal only, no I/O)
        """
        t_start = time.time()

        if not anomalous_services:
            return None

        # Graph is updated by the background refresh thread — skip here
        # to avoid adding Jaeger I/O latency to the critical RCA path.

        # Candidate root causes: {service: (depth, severity, chain)}
        candidates: dict[str, tuple[int, float, list[str]]] = {}

        for symptom_service, severity in anomalous_services.items():
            # BFS traversal of dependencies
            queue: deque[tuple[str, int, list[str]]] = deque()
            queue.append((symptom_service, 0, [symptom_service]))
            visited = {symptom_service}

            while queue:
                current, depth, chain = queue.popleft()
                if depth > max_depth:
                    continue

                # Check if this dependency is anomalous
                current_severity = anomalous_services.get(current, 0.0)
                if current_severity > 0:
                    # This service is anomalous — potential root cause
                    existing = candidates.get(current)
                    if existing is None or depth > existing[0] or current_severity > existing[1]:
                        candidates[current] = (depth, current_severity, list(chain))

                # Traverse to dependencies (callees)
                for dep in self.get_dependencies(current):
                    if dep not in visited:
                        visited.add(dep)
                        queue.append((dep, depth + 1, chain + [dep]))

        if not candidates:
            # All anomalous services are leaf nodes — pick highest severity
            best = max(anomalous_services.items(), key=lambda x: x[1])
            elapsed = (time.time() - t_start) * 1000
            return RCAResult(
                root_cause_service=best[0],
                confidence=best[1] * 0.5,  # Lower confidence for leaf-only
                causal_chain=[best[0]],
                anomaly_scores={best[0]: {"combined": best[1]}},
                graph_depth=0,
                contributing_anomalies=[],
                analysis_time_ms=elapsed,
            )

        # Rank candidates: prefer deepest + highest severity
        ranked = sorted(
            candidates.items(),
            key=lambda x: (x[1][0], x[1][1]),  # (depth, severity)
            reverse=True,
        )

        best_service, (depth, severity, chain) = ranked[0]

        # Calculate confidence
        # Factors: severity of root cause, number of affected dependents, depth
        num_affected = len(self.get_dependents(best_service))
        depth_bonus = min(depth * 0.1, 0.3)
        dependent_bonus = min(num_affected * 0.05, 0.2)
        confidence = min(1.0, severity * 0.5 + depth_bonus + dependent_bonus + 0.2)

        # Collect anomaly scores for all services in the chain
        chain_scores = {}
        for svc in chain:
            if svc in anomalous_services:
                chain_scores[svc] = {"combined": anomalous_services[svc]}

        elapsed = (time.time() - t_start) * 1000

        return RCAResult(
            root_cause_service=best_service,
            confidence=confidence,
            causal_chain=chain,
            anomaly_scores=chain_scores,
            graph_depth=depth,
            contributing_anomalies=[],
            analysis_time_ms=elapsed,
        )

    def get_impact_zone(self, service: str, max_depth: int = 3) -> list[str]:
        """
        Find all services that would be impacted if this service fails.

        Traverses the reverse graph (callers/dependents) to find blast radius.
        """
        impacted = []
        queue: deque[tuple[str, int]] = deque([(service, 0)])
        visited = {service}

        while queue:
            current, depth = queue.popleft()
            if depth > max_depth:
                continue
            if current != service:
                impacted.append(current)
            for caller in self.get_dependents(current):
                if caller not in visited:
                    visited.add(caller)
                    queue.append((caller, depth + 1))

        return impacted

    def get_stats(self) -> dict:
        """Return graph statistics including live trace learning metrics."""
        return {
            "total_services": len(self.get_all_services()),
            "total_edges": sum(len(v) for v in self._graph.values()),
            "edges_from_live_traces": self._edges_from_traces,
            "trace_updates": self._trace_updates_count,
            "last_update_ago_s": round(time.time() - self._last_trace_update, 0),
            "services": sorted(self.get_all_services()),
        }

    def to_dict(self) -> dict:
        """Serialize graph for debugging/visualization."""
        return {
            caller: sorted(callees)
            for caller, callees in sorted(self._graph.items())
        }


# ─── Standalone Testing ──────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import os

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s")

    jaeger_url = os.getenv("JAEGER_URL", "http://localhost:16686")
    graph = DependencyGraph(jaeger_url=jaeger_url)

    print("=== Dependency Graph ===")
    print(json.dumps(graph.to_dict(), indent=2))
    print(f"\nStats: {json.dumps(graph.get_stats(), indent=2)}")

    # Scenario 1: cartservice is unhealthy → affects frontend and checkoutservice
    print("\n=== RCA Scenario 1: cartservice down ===")
    anomalies = {
        "frontend": 0.6,       # Symptom: high latency
        "checkoutservice": 0.7, # Symptom: errors
        "cartservice": 0.9,     # Root cause: crash/OOM
    }
    t0 = time.time()
    result = graph.find_root_cause(anomalies)
    elapsed = (time.time() - t0) * 1000
    print(f"Analysis took {elapsed:.2f}ms")
    if result:
        print(json.dumps(result.to_dict(), indent=2))
    print(f"Impact zone for cartservice: {graph.get_impact_zone('cartservice')}")

    # Scenario 2: redis-cart down → cascades through cartservice
    print("\n=== RCA Scenario 2: redis-cart down ===")
    anomalies2 = {
        "frontend": 0.5,
        "cartservice": 0.8,
        "redis-cart": 0.95,
    }
    t0 = time.time()
    result2 = graph.find_root_cause(anomalies2)
    elapsed = (time.time() - t0) * 1000
    print(f"Analysis took {elapsed:.2f}ms")
    if result2:
        print(json.dumps(result2.to_dict(), indent=2))

    # Scenario 3: productcatalogservice CPU spike
    print("\n=== RCA Scenario 3: productcatalogservice CPU spike ===")
    anomalies3 = {
        "frontend": 0.4,
        "recommendationservice": 0.6,
        "productcatalogservice": 0.85,
    }
    t0 = time.time()
    result3 = graph.find_root_cause(anomalies3)
    elapsed = (time.time() - t0) * 1000
    print(f"Analysis took {elapsed:.2f}ms")
    if result3:
        print(json.dumps(result3.to_dict(), indent=2))

# Usage:
# python dependency_graph.py                              # Run standalone test
# JAEGER_URL=http://localhost:16686 python dependency_graph.py  # Test with local Jaeger
