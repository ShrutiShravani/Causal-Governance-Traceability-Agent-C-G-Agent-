from exception import CGAgentException
from logger import logging
from typing import List,Dict,Any
from src.traceability.audit_logger import log_event, register_artifact
from datetime import datetime
import os,sys

def escalate_human_review(agent_name:str,client_transaction_id:str,trace_id:str,violations:List[Dict[str,Any]],confidence:float):
    """
    Handles human-in-the-loop escalation for predictions that violate governance policies.

    Args:
        prediction_id: Unique ID for the prediction being evaluated.
        violations: List of detected policy violations from OPA.
        confidence: Model confidence for this prediction.
    """
    try:
        escalation_id= f"ESC-{client_transaction_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        logging.info(f"Escalation triggered for client_transaction_id={client_transaction_id},escalation_id={escalation_id}")

        #Build extra payload for audit logging
        escalation_payload={
            "escalation_id": escalation_id,
            "violations": violations,
            "confidence": confidence,
            "action": "Human review required",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        #log event for governnace traceability
        log_event(
            agent_name=agent_name,
            trace_id= trace_id,
            client_transaction_id=client_transaction_id,
            phase="Phase3_Governance",
            event_type="HUMAN_ESCALATION",
            summary="Prediction requires human review due to policy violations or low confidence",
            extra_payload={
                "escalation_payload":escalation_payload
            }

        )
        return escalation_payload
    except Exception as e:
        logging.info(f"Error durign escalation handling :{e}")
        raise CGAgentException(e,sys)
        