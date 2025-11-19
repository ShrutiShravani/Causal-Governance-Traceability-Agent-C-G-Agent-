# 🧭 Causal Governance Agent (C-G Agent)

**C-G Agent** is an auditable AI decision system that combines **causal inference**, **explainability**, and **policy-as-code governance** to ensure every model decision is transparent, reproducible, and compliant with global AI regulations such as the **EU AI Act**, **OECD AI Principles**, and **China’s AI Governance Action Plan**.

The agent detects bias, enforces fairness, and maintains immutable traceability across every decision — from prediction to human oversight.

---

## 🧩 High-Level Architecture

**Runtime Flow**

[Client] → [API Gateway] → [Preprocessor]
↓
[Causal Prediction Engine]
↓
[XAI Explainer (SHAP)]
↓
[Governance Orchestrator (Policy-as-Code)]
↓
┌─────────────┐
│ Pass │→ [Action / Response]
└─────────────┘
┌─────────────┐
│ Fail/Ambig. │→ [Human Review UI]
└─────────────┘
↓
[Traceability Recorder (Immutable Logs)]


---

## ✨ Key Features

| Layer | Function | Tools |
|-------|-----------|-------|
| **Data Pipeline** | Version-controlled ingestion | DVC, pandas |
| **Causal Engine** | Causal ML (ATE, ITE estimation) | DoWhy, EconML |
| **Explainability** | Model explanations | SHAP, Alibi/DiCE |
| **Governance** | Policy-as-Code enforcement | Open Policy Agent (OPA), Rego |
| **Traceability** | Append-only audit chain | PostgreSQL + SHA-256 |
| **Human Oversight** | Escalation & reviewer UI | FastAPI, frontend UI |
| **Compliance Assistant (optional)** | RAG for legal article retrieval | LangChain, FAISS |


---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourname/causal-governance-agent.git
cd causal-governance-agent

2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Linux/Mac
.venv\Scripts\activate      # on Windows

3. Install dependencies
pip install -r requirements.txt

4. Initialize Postgres database
psql -f db/schema.sql

5. Start OPA policy engine
opa run --server policy/regos/

6. Run the FastAPI server
uvicorn src.api.server:app --reload

-Example Usage
Prediction Request
POST /predict


Request

{
  "user_id": "user123",
  "features": {
    "age": 42,
    "income": 48000,
    "loan_amount": 12000
  },
  "context": {"region": "DE"}



Response (Pass)

{
  "task_id": "uuid-1234",
  "prediction": {"outcome": "loan_approve", "score": 0.78},
  "causal_explanation": {
    "top_causes": [{"feature": "debt_to_income", "ate": 0.34}]
  },
  "policy_check": {"status": "pass"},
  "audit_log_ref": "/audit/task/uuid-1234"
}


Response (Escalated)

{
  "task_id": "uuid-1234",
  "status": "escalated",
  "policy_check": {"status": "fail", "reason": "proxy_feature_detected"},
  "human_review_url": "https://review-ui/task/uuid-1234"
}

🧪 Testing & Monitoring

Testing

Unit tests: pytest

Policy regression tests: opa test

Integration: end-to-end predict calls

Evaluation Metrics

Predictive: AUC, precision/recall

Causal validity: robustness of ATE, sensitivity analysis

Fairness: disparate impact, equalized odds

Explainability: fidelity & stability

Operational: latency, escalation %, audit completeness

Monitoring

Prometheus + Grafana dashboards

OpenTelemetry traces

Alerts for: policy failures, data drift, latency spikes

🚀 Deployment
Docker
docker build -t cg-agent .
docker run -p 8000:8000 cg-agent

Kubernetes (Helm)
helm install cg-agent ./helm/

CI/CD

GitHub Actions → lint + test + container build

Image scanning & secret detection

Continuous compliance re-evaluation jobs

🔐 Security & Governance

All audit events hashed (SHA-256) & chain-linked

Immutable, append-only design for tamper-evidence

Secrets stored via Vault / k8s secrets

Encrypted backups with retention schedule

Periodic SOC review & penetration testing