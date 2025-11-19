import pytest
from unittest.mock import MagicMock, patch
from src.agent.conversational_query_agent import QueryAgent


@pytest.fixture
def mock_service():
    """Mocked TraceabilityAgentService."""
    service = MagicMock()
    return service


@pytest.fixture
def agent(mock_service):
    """QueryAgent with mocked dependencies."""
    return QueryAgent(service=mock_service)


# ---------------------------------------------------------
# 1. ROOT-CAUSE QUERY (Why was transaction blocked)
# ---------------------------------------------------------
def test_query_agent_root_cause_integration(agent, mock_service):
    txn_id = "TXN-123"

    # Mock structured graph response
    mock_service.query_audit_trail.return_value = {
        "final_result": "BLOCKED",
        "confidence_status": "High",
        "causal_violations": ["credit_limit", "age_anomaly"],
        "policies_enforced": ["RiskPolicy", "FairUsePolicy"]
    }

    query = "Why was this transaction blocked?"
    response = agent.process_query(query, txn_id)

    # ---- Validate service call ----
    assert mock_service.query_audit_trail.call_count == 1
    cypher = mock_service.query_audit_trail.call_args[0][0]
    assert "MATCH (t:Transaction {id: 'TXN-123'})" in cypher

    # ---- Validate response text ----
    assert "BLOCKED" in response
    assert "credit_limit" in response
    assert "RiskPolicy" in response
    assert "assist" in response.lower()

    # ---- Validate conversation history ----
    assert len(agent.conversation_history) == 1
    assert agent.conversation_history[0]["user"] == query


# ---------------------------------------------------------
# 2. DATA PROVENANCE / HASH QUERY
# ---------------------------------------------------------
def test_query_agent_data_hash_integration(agent, mock_service):
    txn_id = "TXN-777"

    mock_service.query_audit_trail.return_value = {
        "data_provenance_hash": "SHA256-ABC-123"
    }

    response = agent.process_query("Show me data hash", txn_id)

    assert "SHA256-ABC-123" in response
    assert mock_service.query_audit_trail.call_count == 1


# ---------------------------------------------------------
# 3. EVIDENCE / REPORT QUERY
# ---------------------------------------------------------
def test_query_agent_evidence_integration(agent, mock_service):
    txn_id = "TXN-009"

    mock_service.query_audit_trail.return_value = {
        "evidence_artifact_id": "ART-999"
    }

    response = agent.process_query("Show me the report", txn_id)

    assert "ART-999" in response
    assert "oversight" in response.lower()
    assert mock_service.query_audit_trail.call_count == 1


# ---------------------------------------------------------
# 4. PREDICTION SCORE QUERY
# ---------------------------------------------------------
def test_query_agent_prediction_score_integration(agent, mock_service):
    txn_id = "TXN-456"

    mock_service.query_audit_trail.return_value = {
        "label": "Approve",
        "probability": 0.83,
        "model_version": "v2.4"
    }

    response = agent.process_query("Show prediction score", txn_id)

    assert "Approve" in response
    assert "0.83" in response
    assert "v2.4" in response


# ---------------------------------------------------------
# 5. UNKNOWN QUERY → FALLBACK
# ---------------------------------------------------------
def test_query_agent_fallback_integration(agent, mock_service):
    txn_id = "TXN-500"

    response = agent.process_query("something unrecognized?", txn_id)

    # No service call expected
    mock_service.query_audit_trail.assert_not_called()

    assert "could not generate" in response.lower()


# ---------------------------------------------------------
# 6. NO RESULTS FOUND → Graceful message
# ---------------------------------------------------------
def test_query_agent_no_results(agent, mock_service):
    txn_id = "TXN-111"

    agent.service.query_audit_trail.return_value = None

    response = agent.process_query("Why was this rejected?", txn_id)

    assert "No audit records" in response
