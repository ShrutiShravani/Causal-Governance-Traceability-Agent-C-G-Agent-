import pytest
from unittest.mock import MagicMock
from datetime import datetime

from src.agent.critic_agent import CriticAgent
from src.governance_layer.policy_engine import PolicyEngine
from src.governance_layer.escalation import escalation
from src.traceability.telemetry import get_tracer


@pytest.fixture
def fake_prediction_output():
    """
    Fake output from PredictionAgent: prediction + xai
    """
    return {
        "probabilities": [0.85],  # high confidence
        "xai_artifact": {
            "local_top_drivers": ["income", "age"],
            "expected_value": 0.2,
            "risk_factors": ["age"]
        }
    }


@pytest.fixture
def critic_agent():
    return CriticAgent(
        trace_id="test-trace",
        client_transaction_id="txn-123"
    )


def test_critic_agent_approved(monkeypatch, critic_agent, fake_prediction_output):
    """
    Case 1: No violation + confidence is allowed → APPROVED
    """

    # ---- Mock PolicyEngine.evaluate ----
    fake_policy_results = {
        "confidence_threshold": {"allow": True},
        "prohibited_attributes": {"violations": []}
    }

    monkeypatch.setattr(
        PolicyEngine,
        "evaluate",
        lambda self, inp: fake_policy_results
    )

    # ---- Mock escalation (should NOT be called for APPROVED) ----
    fake_escalate = MagicMock()
    monkeypatch.setattr(escalation, "escalate_human_review", fake_escalate)

    # ---- Mock tracer ----
    monkeypatch.setattr("src.agent.critic_agent.get_tracer", lambda: MagicMock())

    # ---- Run ----
    result = critic_agent.evaluate_prediction(fake_prediction_output)

    # ---- Assertions ----
    assert result["final_decision"] == "APPROVED"
    assert result["violations"] == []
    assert result["confidence_status"] == "pass"
    assert "prediction_id" in result
    fake_escalate.assert_not_called()


def test_critic_agent_blocked_confidence_fail(monkeypatch, critic_agent, fake_prediction_output):
    """
    Case 2: Confidence threshold fails → BLOCKED + human review triggered
    """

    # Lower confidence to trigger failure
    fake_prediction_output["probabilities"] = [0.40]

    fake_policy_results = {
        "confidence_threshold": {"allow": False},  # FAIL
        "prohibited_attributes": {"violations": []}
    }

    monkeypatch.setattr(
        PolicyEngine,
        "evaluate",
        lambda self, inp: fake_policy_results
    )

    fake_escalate = MagicMock()
    monkeypatch.setattr(escalation, "escalate_human_review", fake_escalate)

    # Run test
    result = critic_agent.evaluate_prediction(fake_prediction_output)

    assert result["final_decision"] == "BLOCKED"
    assert result["confidence_status"] == "human_review_required"
    fake_escalate.assert_called_once()


def test_critic_agent_blocked_prohibited_attributes(monkeypatch, critic_agent, fake_prediction_output):
    """
    Case 3: Violations in prohibited attributes → BLOCKED but NO escalation
    Because escalation only happens if BOTH:
        - BLOCKED
        - confidence_status == human_review_required
    """

    fake_policy_results = {
        "confidence_threshold": {"allow": True},
        "prohibited_attributes": {"violations": ["gender", "race"]}
    }

    monkeypatch.setattr(
        PolicyEngine,
        "evaluate",
        lambda self, inp: fake_policy_results
    )

    fake_escalate = MagicMock()
    monkeypatch.setattr(escalation, "escalate_human_review", fake_escalate)

    result = critic_agent.evaluate_prediction(fake_prediction_output)

    assert result["final_decision"] == "BLOCKED"
    assert result["violations"] == ["gender", "race"]

    # Should NOT escalate because confidence is PASS
    fake_escalate.assert_not_called()
