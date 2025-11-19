import pytest
from unittest.mock import MagicMock,patch
from datetime import datetime
import uuid
from exception import CGAgentException
from logger import logging
from src.traceability.traceability_agent_service import TraceabilityAgentService
from src.traceability.graph_config import GraphDBClient

def initalize_client(monkeypatch):
    mock_graph= MagicMock(spec=GraphDBClient)
    service = TraceabilityAgentService(graph_client=mock_graph)

    raw_event = {
        "event_id": "EVT-1",
        "client_transaction_id": "TXN-100",
        "agent_name": "risk_agent",
        "event_type": "PREDICTION_EXECUTED",
        "created_at": "2025-11-14T10:00:00",
        "extra": {
            "prediction_label": "approve",
            "prediction_probability": 0.81,
            "model_version": "v1.0.0"
        }
    }

    service.synthesize_and_ingest_event(raw_event)

    # verify execute_cypher was called exactly once
    mock_graph.execute_cypher.assert_called_once()

    # get the actual cypher the service generated
    generated_cypher, cypher_params = mock_graph.execute_cypher.call_args[0]

    assert "MERGE (a:Agent {name: 'risk_agent'})" in generated_cypher
    assert "MERGE (t:Transaction {id: 'TXN-100'})" in generated_cypher
    assert "CREATE (e:Event {id: 'EVT-1'" in generated_cypher
    assert "CREATE (m:PredictionMetrics" in generated_cypher
    assert "prediction_label" in generated_cypher


#query audit trail

def test_query_audit_trail_final_decision():
    mock_graph = MagicMock(spec=GraphDBClient)
    service = TraceabilityAgentService(graph_client=mock_graph)

    mock_graph.execute_cypher.return_value = {
        'd': {'result': 'APPROVED', 'confidence_status': 'human_review_required'},
        'violations': ['prohibited_attribute'],
        'policies': ['fairness_check']
    }

    query = "MATCH (d:FinalDecision)..."
    result = service.query_audit_trail(query)

    assert result['final_result'] == "APPROVED"
    assert result['confidence_status'] == "human_review_required"
    assert result['causal_violations'] == ['prohibited_attribute']
    assert result['policies_enforced'] == ['fairness_check']

def test_query_audit_trail_dataset_hash():
    mock_graph = MagicMock(spec=GraphDBClient)
    service = TraceabilityAgentService(mock_graph)

    mock_graph.execute_cypher.return_value = {
        'ds.hash': "SHA256-XYZ123"
    }

    result = service.query_audit_trail("MATCH (ds)...")

    assert result["data_provenance_hash"] == "SHA256-XYZ123"

def test_query_audit_trail_artifact():
    mock_graph = MagicMock(spec=GraphDBClient)
    service = TraceabilityAgentService(mock_graph)

    mock_graph.execute_cypher.return_value = {
        'art': {'id': 'ART-1', 'kind': 'XAI_REPORT'}
    }
    result = service.query_audit_trail("MATCH (a:Artifact)...")

    assert result["evidence_artifact_id"] == "ART-1"
    assert result["evidence_kind"] == "XAI_REPORT"
