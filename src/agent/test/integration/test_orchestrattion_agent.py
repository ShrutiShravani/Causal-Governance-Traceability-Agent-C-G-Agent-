import pytest
from unittest.mock import MagicMock
from src.agent.orchestration_agent import OrchestratorAgent
from src.agent.prediction_agent import PredictionAgent
from src.agent.critic_agent import CriticAgent
from src.traceability.audit_logger import log_event


def test_orchestrator_happy_path(monkeypatch):
    trace_id = "trace-123"
    txn = "txn-abc"

    # ----- Mock PredictionAgent -----
    fake_prediction = {"status": "success", "prediction_output": {"foo": "bar"}}
    mock_pred = MagicMock(return_value=fake_prediction)
    monkeypatch.setattr(PredictionAgent, "run", mock_pred)

    # ----- Mock CriticAgent -----
    fake_critic = {"status": "success", "final_decision": "APPROVED"}
    mock_critic = MagicMock(return_value=fake_critic)
    monkeypatch.setattr(CriticAgent, "run", mock_critic)

    # ----- Mock telemetry tracer -----
    monkeypatch.setattr(
        "src.agent.orchestrator_agent.get_tracer",
        lambda: MagicMock()
    )

    orch = OrchestratorAgent(
        agent_name="Orchestrator_Agent",
        trace_id=trace_id,
        client_transaction_id=txn
    )

    payload = {"input": "test"}

    result = orch.run(payload)

    # ----- Assertions -----
    mock_pred.assert_called_once()
    mock_critic.assert_called_once()

    assert result["status"] == "success"
    assert result["trace_id"] == trace_id
    assert result["client_transaction_id"] == txn
    assert result["prediction"] == fake_prediction
    assert result["critic"] == fake_critic


def test_orchestrator_failure_path(monkeypatch):
    trace_id = "trace-err"
    txn = "txn-err"

    # PredictionAgent.run will fail
    monkeypatch.setattr(
        PredictionAgent, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("FAILED"))
    )

    # Mock log_event to verify error logging
    mock_log = MagicMock()
    monkeypatch.setattr("src.agent.orchestrator_agent.log_event", mock_log)

    # Mock tracer
    monkeypatch.setattr(
        "src.agent.orchestrator_agent.get_tracer",
        lambda: MagicMock()
    )

    orch = OrchestratorAgent("Orchestrator_Agent", trace_id, txn)

    result = orch.run({"x": 1})

    assert result["status"] == "failed"
    assert "error" in result

    mock_log.assert_called_once()
