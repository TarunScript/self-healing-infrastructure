# AI-Powered Autonomous Observability and Self-Healing System

## Overview
This project presents an AI-driven observability system designed to monitor microservices, detect anomalies, identify root causes, and automatically take corrective actions in real time.

Traditional monitoring tools focus primarily on alerting. In contrast, this system aims to move beyond observation by enabling automated diagnosis and recovery, reducing the need for manual intervention.

---

## Key Features
* Log anomaly detection using Drain3 for pattern extraction and DistilBERT for semantic understanding
* Metric anomaly detection using Isolation Forest for outlier detection and Long Short-Term Memory for time-series analysis
* Graph-based root cause analysis using dependency graph traversal
* Automated self-healing actions such as restarting or scaling services
* Integration with a standard observability stack including Prometheus, Loki, Jaeger, and Grafana

---

## Architecture
The system follows a pipeline-oriented architecture:

Microservices → Observability Stack → AI Engine → Root Cause Analysis → Decision Engine → Action → Visualization

### Flow Description
1. Logs, metrics, and traces are collected from running services
2. Machine learning models analyze incoming data streams
3. Anomalies are detected across logs and metrics
4. Signals are correlated to identify affected services
5. Root cause analysis is performed using a dependency graph
6. The system evaluates confidence and determines corrective actions
7. Actions are executed and reflected in the monitoring dashboard

---

## Machine Learning Approach
The system combines multiple learning strategies:

* Drain3 performs online log template extraction without requiring labeled data
* DistilBERT is used as a pretrained language model for log classification using a zero-shot approach
* Isolation Forest is applied for unsupervised anomaly detection in metrics
* LSTM is used to capture temporal patterns and detect gradual system degradation

A key design choice is avoiding reliance on labeled datasets. Instead, the system learns from live system behavior using unsupervised and pretrained models.

---

## Root Cause Analysis
The system models services as a dependency graph, where nodes represent services and edges represent interactions between them.

A graph traversal approach based on Breadth-First Search (BFS) is used to trace upstream dependencies and identify the true origin of failures rather than surface-level symptoms.

---

## Technology Stack
* Python (FastAPI for backend services)
* Docker and Docker Compose for containerization
* Prometheus for metrics collection
* Loki for log aggregation
* Jaeger for distributed tracing
* Grafana for visualization
* Locust for load testing and failure simulation

---

## System Workflow
1. The system continuously collects logs and metrics
2. AI models process incoming data in real time
3. Anomalies are identified across multiple signals
4. Root cause analysis determines the source of failure
5. The system executes automated corrective actions
6. System stability is restored and reflected in dashboards

---

## Scalability Considerations
While the current implementation is optimized for demonstration and low-latency response, it can be extended for production environments through:

* Deployment on Kubernetes for orchestration and scaling
* Integration with streaming platforms such as Kafka
* Distributed inference for machine learning components
* Optimized model serving using GPU acceleration

---

##  Novelty
The primary contribution of this project lies in combining detection, diagnosis, and remediation into a single closed-loop system.

It integrates multiple data sources (logs, metrics, and traces), applies machine learning for anomaly detection, performs structured root cause analysis, and executes automated recovery actions. This moves the system from passive monitoring toward autonomous operation.

---

## Contributors
(Add team member names here)

---

## License
