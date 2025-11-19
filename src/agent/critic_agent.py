from logger import logging
import uuid
from typing import Dict, Any, List
from src.governance_layer.policy_engine import PolicyEngine
from src.governance_layer.escalation import escalate_human_review
from exception import CGAgentException
from datetime import datetime
from src.traceability.telemetry import get_tracer, inject_trace_context,extract_trace_context
from src.traceability.audit_logger import log_event
import os,sys

class CriticAgent:
    """
    Governance Layer Critic Agent.
    Evaluates Prediction + XAI artifacts against Rego policies via OPA.
    """
    agent_name="Critic_Agent"
    def __init__(self, trace_id:str,client_transaction_id:str):
        self.trace_id = trace_id
        self.client_transaction_id= client_transaction_id
        self.tracer=get_tracer()
        self.policy_engine= PolicyEngine()

        logging.info(f"PredictionAgent initialized with trace_id={self.trace_id}")

    def run(self,payload:dict=None)->dict:
        """
        Runs all relevant policies on the prediction output and XAI artifact.
        """
        try:
            logging.info("PredictionAgent: Starting workflow...")
            ctx= extract_trace_context(payload)
            # do prediction
            print(f"[PredictionAgent] Running under trace_id={self.trace_id}")

            with self.tracer.start_as_current_span("orchestration_logic",context=ctx):
                logging.info(f"[PredictionAgent] Running under trace_id={self.trace_id}")
                evaluation_output = self.evaluate_prediction(self.prediction_output)

                return {
                    "agent":self.agent_name,
                    "status":"success",
                    "trace_id": self.trace_id,
                    "client_transaction_id": self.client_transaction_id,
                    "prediction_output":evaluation_output

                }
        except Exception as e:
            log_event(
                trace_id=self.trace_id,
                client_transaction_id=self.client_transaction_id,
                agent_name=self.agent_name,
                phase="Phase4_Evaluation",
                event_type="ERROR",
                summary=str(e)
            )
            return {
                "agent":self.agent_name,
                "status":"failed",
                "error":str(e)
            }
    
    def evaluate_prediction(self,prediction_output:Dict[str,Any]):
        """
        Runs all relevant policies on the prediction output and XAI artifact.
        """
        try:
            prediction_id= str(uuid.uuid4())
            xai_artifact= prediction_output.get("xai_artifact",{})
            probas= prediction_output.get("probabilities",[])

            #Policy 1 : prohibited attribute check
            policy_input = {
                "top_drivers": xai_artifact.get("local_top_drivers", []),
                "confidence": max(probas) if probas else 0.0,
            }
            
            #evaluate all active polciies via PolicyEngine
            policy_results= self.policy_engine.evaluate(policy_input)

            #aggregate results
            violations=[]
            confidence_status= "pass"

            for policy_name,result in policy_results.items():
                if policy_name.lower()=="confidence_threshold":
                    confidence_status= "pass" if result.get("allow",False) else "human_review_required"
                if policy_name.lower() == "prohibited_attributes":
                    violations = result.get("violations", [])

          

            final_decision="APPROVED" if not violations and confidence_status=="pass" else "BLOCKED"

            output ={
                "trace_id": self.trace_id,
                "prediction_id": prediction_id,
                "policy_results":policy_results,
                "violations": violations,
                "confidence_status":confidence_status,
                "final_decision": final_decision,
                "timestamp": datetime.utcnow().isoformat() + "Z"

            }

            log_event(
               agent_name=self.agent_name,
               trace_id=self.trace_id,
               client_transaction_id=self.client_transaction_id,
               phase="Phase3_Governance",
               event_type="POLICY_ENFORCED",
               summary=f"Critic Agent computed decision : {final_decision}",
               extra_payload={
                "prediciton_id":output["prediction_id"],
                "final_decision":final_decision,
                "policy_results_summary": output["policy_results"],
                "violations": violations,
                "confidence_status":confidence_status,
               }
            )

             #trigger human escalation if needed
            if final_decision=="BLOCKED" and confidence_status=="human_review_required":
                escalate_human_review(agent_name=self.agent_name,client_transaction_id=self.client_transaction_id,trace_id=self.trace_id,violations=violations,confidence=policy_input["confidence"])
            return output
        
        except Exception as e:
            logging.info(f"CriticAgent evaluation failed: {e}")
            raise CGAgentException(e, sys)
        