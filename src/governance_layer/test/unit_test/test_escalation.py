import pytest
from unittest.mock import patch, MagicMock
from exception import CGAgentException
from src.governance_layer.escalation import escalate_human_review

def test_escalation_success():
    mock_response=MagicMock()
    mock_response.raise_for_status.return_value=None
    mock_response.json.return_value={"result":{"allow":True}}

    mock_violations={"policy":"bias_check"}

    with patch ("src.traceability.audit_logger.register_artifact",return_value="ART123") as mock_artifact, patch("src.traceability.log_event") as mock_log:

        result= escalate_human_review(agent_name="critic_agent",
            client_transaction_id="TXN001",
            trace_id="TRACE123",
            violations=mock_violations,
            confidence=0.45)
        
        #mock register and log event called
        mock_artifact.assert_called_once()
        mock_log.assert_called_once()

        assert result["artifact_id"] == "ART123"
        
        #chcek escalation_id is correct
        assert result["escalation_id"].startswith(f"ESC-TXN001")
        # Result contains expected keys
        assert result["violations"] == mock_violations
        assert result["confidence"] == 0.45
        assert result["action"] == "Human review required"
        
        

#log regoster artifact failure
def test_escalation_register_artifact_failure():
    with patch("src.traceability.audit_logger.register_artifact", side_effect=Exception("fail")):
        with pytest.raises(CGAgentException):
            escalate_human_review(
                "critic_agent", "TXN001", "TRACE123", [], 0.5
            )

#log event failure
def test_escalation_log_event_failure():
    with patch("src.traceability.audit_logger.register_artifact", return_value="ART123"), \
         patch("src.traceability.audit_logger.log_event", side_effect=Exception("log error")):

        with pytest.raises(CGAgentException):
            escalate_human_review(
                "critic_agent", "TXN001", "TRACE123", [], 0.5
            )



