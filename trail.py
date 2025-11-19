import pytest
from unittest.mock import patch, MagicMock
from governance_layer.critic_agent import CriticAgent
from exception import CGAgentException


# ------------------------------------------------------
# Helper to build a CriticAgent with mock prediction_output
# ------------------------------------------------------
def build_agent():
    agent = CriticAgent(trace_id="TRACE123", client_transaction_id="TXN001")
    # Inject mock prediction_output
    agent.prediction_output = {
        "xai_artifact": {"local_top_drivers": ["age"]},
        "probabilities": [0.92]
    }
    return agent


# ==========================================================
# TEST 1 — HAPPY PATH (APPROVED)
# ==========================================================
def test_critic_agent_approved():
    agent = build_agent()

    mock_policy_output = {
        "confidence_threshold": {"allow": True},
        "prohibited_attributes": {"violations": []},
    }

    with patch.object(agent.policy_engine, "evaluate", return_value=mock_policy_output), \
         patch("governance_layer.critic_agent.log_event") as mock_log, \
         patch("governance_layer.critic_agent.escalate_human_review") as mock_escalate, \
         patch.object(agent.tracer, "start_as_current_span", return_value=MagicMock()):

        result = agent.evaluate_prediction(agent.prediction_output)

        assert result["final_decision"] == "APPROVED"
        mock_escalate.assert_not_called()
        mock_log.assert_called_once()


# ==========================================================
# TEST 2 — BLOCKED due to prohibited attributes (NO escalation)
# ==========================================================
def test_critic_agent_blocked_prohibited_attributes():
    agent = build_agent()

    mock_policy_output = {
        "confidence_threshold": {"allow": True},
        "prohibited_attributes": {"violations": ["gender"]},
    }

    with patch.object(agent.policy_engine, "evaluate", return_value=mock_policy_output), \
         patch("governance_layer.critic_agent.log_event") as mock_log, \
         patch("governance_layer.critic_agent.escalate_human_review") as mock_escalate:

        result = agent.evaluate_prediction(agent.prediction_output)

        assert result["final_decision"] == "BLOCKED"
        mock_escalate.assert_not_called()


# ==========================================================
# TEST 3 — BLOCKED due to low confidence (ESCALATION expected)
# ==========================================================
def test_critic_agent_blocked_low_confidence():
    agent = build_agent()

    mock_policy_output = {
        "confidence_threshold": {"allow": False},
        "prohibited_attributes": {"violations": []},
    }

    with patch.object(agent.policy_engine, "evaluate", return_value=mock_policy_output), \
         patch("governance_layer.critic_agent.log_event"), \
         patch("governance_layer.critic_agent.escalate_human_review") as mock_escalate:

        result = agent.evaluate_prediction(agent.prediction_output)

        assert result["final_decision"] == "BLOCKED"
        mock_escalate.assert_called_once()


# ==========================================================
# TEST 4 — ERROR inside evaluate_prediction → raises CGAgentException
# ==========================================================
def test_critic_agent_evaluate_prediction_exception():
    agent = build_agent()

    # Force policy engine to crash
    with patch.object(agent.policy_engine, "evaluate", side_effect=Exception("policy fail")):
        with pytest.raises(CGAgentException):
            agent.evaluate_prediction(agent.prediction_output)


# ==========================================================
# TEST 5 — Partial policy result (MISSING confidence_threshold)
# ==========================================================
def test_critic_agent_partial_policy_result():
    agent = build_agent()

    mock_policy_output = {
        "prohibited_attributes": {"violations": []}
    }

    with patch.object(agent.policy_engine, "evaluate", return_value=mock_policy_output), \
         patch("governance_layer.critic_agent.log_event"):

        result = agent.evaluate_prediction(agent.prediction_output)

        # Should still APPROVE
        assert result["final_decision"] == "APPROVED"
        assert result["confidence_status"] == "pass"


# ==========================================================
# TEST 6 — Verify logging + tracing span is created
# ==========================================================
def test_critic_agent_logging_and_tracing():
    agent = build_agent()

    mock_eval_result = {
        "confidence_threshold": {"allow": True},
        "prohibited_attributes": {"violations": []},
    }

    mock_span = MagicMock()

    with patch.object(agent.policy_engine, "evaluate", return_value=mock_eval_result), \
         patch("governance_layer.critic_agent.log_event") as mock_log, \
         patch.object(agent.tracer, "start_as_current_span", return_value=mock_span), \
         patch("governance_layer.critic_agent.extract_trace_context", return_value=None):

        agent.run({"dummy": "payload"})

        mock_log.assert_called()
        mock_span.__enter__.assert_called()     # span started
        mock_span.__exit__.assert_called()      # span ended


# ==========================================================
# TEST 7 — Response contract validation
# ==========================================================
def test_critic_agent_response_contract():
    agent = build_agent()

    mock_eval_result = {
        "confidence_threshold": {"allow": True},
        "prohibited_attributes": {"violations": []},
    }

    with patch.object(agent.policy_engine, "evaluate", return_value=mock_eval_result), \
         patch("governance_layer.critic_agent.log_event"), \
         patch.object(agent.tracer, "start_as_current_span", return_value=MagicMock()), \
         patch("governance_layer.critic_agent.extract_trace_context", return_value=None):

        response = agent.run({"dummy": "payload"})

        # Validate schema:
        assert response["agent"] == "Critic_Agent"
        assert response["status"] == "success"
        assert "prediction_output" in response
        pred = response["prediction_output"]

        assert "policy_results" in pred
        assert "prediction_id" in pred
        assert "final_decision" in pred
        assert "timestamp" in pred

        # prediction_id must be UUID
        import uuid
        uuid.UUID(pred["prediction_id"])
