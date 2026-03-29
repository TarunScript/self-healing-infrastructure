# Structured evidence packet generator for explainable AI remediation
# File: ai_engine/explainability.py
# Purpose: Creates judge-ready JSON evidence packets linking detection → RCA → remediation
# Every automated action gets a structured "why" document with full causal chain

import time
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class EvidencePacket:
    """
    Structured evidence for a single detection → RCA → remediation cycle.

    This is the core explainability artifact — it answers:
    - WHAT was detected (anomaly details)
    - WHY it happened (causal chain from RCA)
    - WHAT action was taken (remediation details)
    - HOW CONFIDENT the system is (risk assessment)
    """

    # Unique identifiers
    incident_id: str
    cycle_timestamp: float

    # Detection evidence
    detection: dict = field(default_factory=dict)

    # RCA evidence
    root_cause: dict = field(default_factory=dict)

    # Remediation evidence
    remediation: dict = field(default_factory=dict)

    # Timing
    timing: dict = field(default_factory=dict)

    # Outcome tracking
    outcome: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "cycle_timestamp": self.cycle_timestamp,
            "detection": self.detection,
            "root_cause": self.root_cause,
            "remediation": self.remediation,
            "timing": self.timing,
            "outcome": self.outcome,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class ExplainabilityEngine:
    """
    Generates structured evidence packets for every AI-driven remediation action.

    Evidence schema (for Grafana annotations and judge review):
    {
      "incident_id": "unique-hash",
      "cycle_timestamp": 1711573400.0,
      "detection": {
        "anomalies": [...],
        "total_anomaly_score": 0.85,
        "signals": ["logs", "metrics", "traces"],
        "services_affected": ["frontend", "cartservice"]
      },
      "root_cause": {
        "service": "cartservice",
        "confidence": 0.92,
        "causal_chain": ["cartservice", "frontend"],
        "reasoning": "Upstream dependency cartservice shows...",
        "impact_zone": ["frontend", "checkoutservice"]
      },
      "remediation": {
        "action": "restart_container",
        "target": "cartservice",
        "confidence_gate": "passed (0.92 > 0.7)",
        "execution_time_ms": 1200,
        "status": "success"
      },
      "timing": {
        "detection_ms": 45,
        "rca_ms": 8,
        "remediation_ms": 1200,
        "total_cycle_ms": 1253,
        "budget_15s": "PASS"
      }
    }
    """

    def __init__(self, grafana_url: str = "http://grafana:3000"):
        self.grafana_url = grafana_url
        # History of evidence packets for this session
        self._history: list[EvidencePacket] = []
        self._max_history = 100

    def _generate_incident_id(self, service: str, anomaly_type: str, timestamp: float) -> str:
        """Generate a deterministic incident ID from key fields."""
        raw = f"{service}:{anomaly_type}:{int(timestamp)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _generate_reasoning(self, root_cause: dict, detection: dict) -> str:
        """Generate human-readable reasoning text for the evidence packet."""
        service = root_cause.get("root_cause_service", "unknown")
        confidence = root_cause.get("confidence", 0)
        chain = root_cause.get("causal_chain", [])
        anomaly_scores = root_cause.get("anomaly_scores", {})

        lines = []
        lines.append(f"Root cause identified: {service} (confidence: {confidence:.0%})")

        if len(chain) > 1:
            chain_str = " → ".join(chain)
            lines.append(f"Causal chain: {chain_str}")
            lines.append(f"Impact propagation: {service} failure cascades to {', '.join(chain[1:])}")

        # Describe contributing anomalies
        anomaly_types = detection.get("anomaly_types", {})
        if anomaly_types:
            type_desc = {
                "new_template": "new error log patterns",
                "frequency_spike": "log frequency spike",
                "error_burst": "error log burst",
                "latency_spike": "latency spike",
                "error_rate": "elevated error rate",
                "resource_saturation": "resource saturation (CPU/memory)",
                "multi_signal": "multiple signal anomalies",
            }
            for svc, types in anomaly_types.items():
                type_strs = [type_desc.get(t, t) for t in types]
                lines.append(f"  {svc}: detected {', '.join(type_strs)}")

        return "; ".join(lines)

    def create_evidence(
        self,
        log_anomalies: list,
        metric_anomalies: list,
        rca_result: Optional[Any],
        remediation_result: Optional[dict],
        timing: dict,
    ) -> EvidencePacket:
        """
        Create a structured evidence packet from a detection cycle's outputs.

        Args:
            log_anomalies: List of LogAnomaly objects (or dicts via .to_dict())
            metric_anomalies: List of MetricAnomaly objects (or dicts)
            rca_result: RCAResult object (or dict) from dependency_graph
            remediation_result: Dict with remediation action details
            timing: Dict with per-stage timing measurements

        Returns:
            Complete EvidencePacket ready for Grafana annotation
        """
        now = time.time()

        # Normalize anomalies to dicts
        log_dicts = [a.to_dict() if hasattr(a, 'to_dict') else a for a in (log_anomalies or [])]
        metric_dicts = [a.to_dict() if hasattr(a, 'to_dict') else a for a in (metric_anomalies or [])]
        rca_dict = rca_result.to_dict() if hasattr(rca_result, 'to_dict') else (rca_result or {})
        rem_dict = remediation_result or {}

        # Determine affected services
        services_affected = set()
        anomaly_types_by_service: dict[str, list[str]] = {}

        for a in log_dicts:
            svc = a.get("service", "unknown")
            services_affected.add(svc)
            anomaly_types_by_service.setdefault(svc, []).append(a.get("anomaly_type", "unknown"))

        for a in metric_dicts:
            svc = a.get("service", "unknown")
            services_affected.add(svc)
            anomaly_types_by_service.setdefault(svc, []).append(a.get("anomaly_type", "unknown"))

        # Compute total anomaly score
        all_scores = [a.get("severity", 0) for a in log_dicts + metric_dicts]
        total_score = max(all_scores) if all_scores else 0

        # Determine which signal types contributed
        signals = []
        if log_dicts:
            signals.append("logs")
        if metric_dicts:
            signals.append("metrics")
        # Traces are implicitly used via the dependency graph
        if rca_dict:
            signals.append("traces")

        # Build detection section
        detection = {
            "log_anomalies": log_dicts[:5],
            "metric_anomalies": metric_dicts[:5],
            "total_anomaly_score": round(total_score, 3),
            "signals": signals,
            "services_affected": sorted(services_affected),
            "anomaly_types": anomaly_types_by_service,
        }

        # Build root cause section
        root_cause_service = rca_dict.get("root_cause_service", "unknown")
        root_cause = {
            "service": root_cause_service,
            "confidence": rca_dict.get("confidence", 0),
            "causal_chain": rca_dict.get("causal_chain", []),
            "anomaly_scores": rca_dict.get("anomaly_scores", {}),
            "graph_depth": rca_dict.get("graph_depth", 0),
            "reasoning": self._generate_reasoning(rca_dict, detection),
        }

        # Build remediation section
        remediation = {
            "action": rem_dict.get("action", "none"),
            "target": rem_dict.get("target", root_cause_service),
            "confidence_gate": rem_dict.get("confidence_gate", "not_evaluated"),
            "execution_time_ms": rem_dict.get("execution_time_ms", 0),
            "status": rem_dict.get("status", "pending"),
            "details": rem_dict.get("details", ""),
        }

        # Build timing section
        total_ms = sum(v for v in timing.values() if isinstance(v, (int, float)))
        timing_section = {
            **timing,
            "total_cycle_ms": round(total_ms, 1),
            "budget_15s": "PASS" if total_ms < 15000 else "FAIL",
            "budget_remaining_ms": round(15000 - total_ms, 1),
        }

        # Generate incident ID
        incident_id = self._generate_incident_id(
            root_cause_service,
            log_dicts[0].get("anomaly_type", "metric") if log_dicts else "metric",
            now,
        )

        packet = EvidencePacket(
            incident_id=incident_id,
            cycle_timestamp=now,
            detection=detection,
            root_cause=root_cause,
            remediation=remediation,
            timing=timing_section,
        )

        # Store in history
        self._history.append(packet)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.info(
            f"Evidence packet created: incident={incident_id}, "
            f"root_cause={root_cause_service}, "
            f"confidence={root_cause['confidence']:.2f}, "
            f"total_cycle={total_ms:.0f}ms"
        )

        return packet

    def post_grafana_annotation(self, packet: EvidencePacket) -> bool:
        """
        Post an evidence packet as a Grafana annotation for dashboard display.

        Creates a clickable annotation that judges can inspect to see full evidence.
        """
        try:
            root_svc = packet.root_cause.get("service", "unknown")
            confidence = packet.root_cause.get("confidence", 0)
            action = packet.remediation.get("action", "none")
            total_ms = packet.timing.get("total_cycle_ms", 0)
            budget = packet.timing.get("budget_15s", "?")

            # Annotation text (shown on hover in Grafana)
            text = (
                f"🔍 **Root Cause**: {root_svc}\n"
                f"📊 **Confidence**: {confidence:.0%}\n"
                f"🔧 **Action**: {action}\n"
                f"⏱️ **Cycle**: {total_ms:.0f}ms [{budget}]\n"
                f"📋 **Chain**: {' → '.join(packet.root_cause.get('causal_chain', []))}\n\n"
                f"<details><summary>Full Evidence</summary>\n\n"
                f"```json\n{packet.to_json()}\n```\n</details>"
            )

            annotation = {
                "dashboardUID": "ai-obs-rca",
                "time": int(packet.cycle_timestamp * 1000),
                "tags": [
                    "ai-rca",
                    f"service:{root_svc}",
                    f"action:{action}",
                    budget,
                ],
                "text": text,
            }

            resp = requests.post(
                f"{self.grafana_url}/api/annotations",
                json=annotation,
                headers={
                    "Content-Type": "application/json",
                },
                timeout=3,
            )
            resp.raise_for_status()
            logger.info(f"Grafana annotation posted: incident={packet.incident_id}")
            return True

        except requests.RequestException as e:
            logger.warning(f"Failed to post Grafana annotation: {e}")
            return False

    def get_recent_incidents(self, count: int = 10) -> list[dict]:
        """Return recent evidence packets for API/dashboard consumption."""
        return [p.to_dict() for p in self._history[-count:]]

    def get_stats(self) -> dict:
        """Return engine statistics."""
        return {
            "total_incidents": len(self._history),
            "pass_rate": sum(
                1 for p in self._history if p.timing.get("budget_15s") == "PASS"
            ) / max(len(self._history), 1),
        }


# ─── Standalone Testing ──────────────────────────────────────────────
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s")

    engine = ExplainabilityEngine(
        grafana_url=os.getenv("GRAFANA_URL", "http://localhost:3000")
    )

    # Simulate a complete detection cycle
    mock_log_anomalies = [
        {
            "detector": "drain_log",
            "service": "cartservice",
            "template": "ERROR: connection refused to redis-cart:6379",
            "anomaly_type": "error_burst",
            "severity": 0.9,
            "sample_logs": ["ERROR: connection refused to redis-cart:6379"],
            "template_id": 42,
            "current_count": 25,
            "baseline_count": 1.0,
            "trace_ids": ["abc123", "def456"],
        }
    ]

    mock_metric_anomalies = [
        {
            "detector": "isolation_forest_metric",
            "service": "cartservice",
            "anomaly_type": "error_rate",
            "severity": 0.85,
            "anomaly_score": -0.42,
            "contributing_metrics": {
                "error_rate": {"current": 0.45, "baseline_mean": 0.01, "z_score": 8.2},
                "p99_latency_ms": {"current": 2500, "baseline_mean": 120, "z_score": 12.1},
            },
        }
    ]

    mock_rca = {
        "root_cause_service": "cartservice",
        "confidence": 0.92,
        "causal_chain": ["cartservice", "frontend", "checkoutservice"],
        "anomaly_scores": {"cartservice": {"combined": 0.9}, "frontend": {"combined": 0.6}},
        "graph_depth": 1,
    }

    mock_remediation = {
        "action": "restart_container",
        "target": "cartservice",
        "confidence_gate": "passed (0.92 > 0.7)",
        "execution_time_ms": 1200,
        "status": "success",
        "details": "Container cartservice restarted successfully",
    }

    mock_timing = {
        "detection_ms": 45,
        "rca_ms": 8,
        "remediation_ms": 1200,
    }

    packet = engine.create_evidence(
        log_anomalies=mock_log_anomalies,
        metric_anomalies=mock_metric_anomalies,
        rca_result=mock_rca,
        remediation_result=mock_remediation,
        timing=mock_timing,
    )

    print("\n=== Evidence Packet ===")
    print(packet.to_json())

    print(f"\n=== Stats ===")
    print(json.dumps(engine.get_stats(), indent=2))

    # Try posting to Grafana
    print("\n=== Posting to Grafana ===")
    success = engine.post_grafana_annotation(packet)
    print(f"Annotation posted: {success}")

# Usage:
# python explainability.py                                  # Run standalone test
# GRAFANA_URL=http://localhost:3000 python explainability.py  # Test with local Grafana
