from logger import logging
import uuid
from typing import Dict, Any, List
from src.governance_layer.policy_engine import PolicyEngine
from src.governance_layer.escalation import escalate_human_review
from exception import CGAgentException
from datetime import datetime
#from src.traceability.telemetry import get_tracer, inject_trace_context,extract_trace_context
from src.traceability.audit_logger import log_event,register_artifact
import os,sys
import json
from src.data_pipeline.data_hashing import generate_sha256_hash,dataframe_to_stable_bytes
import subprocess


class CriticAgent:
    """
    Governance Layer Critic Agent.
    Evaluates Prediction + XAI artifacts against Rego policies via OPA.
    """
    trace_id="trace_123"
    client_transaction_id="pred_456"
    agent_name="Critic_Agent"
    def __init__(self,trace_id:str,client_transaction_id:str):
        self.trace_id = trace_id
        self.client_transaction_id= client_transaction_id
       # self.tracer=get_tracer()
        self.policy_engine= PolicyEngine()

        logging.info(f"PredictionAgent initialized with trace_id={self.trace_id}")
    
    def get_git_revision(self):
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    def run(self,prediction_output:dict=None)->dict:
        """
        Runs all relevant policies on the prediction output and XAI artifact.
        """
        try:
            logging.info("PredictionAgent: Starting workflow...")
            #ctx= extract_trace_context(payload)
            # do prediction
            print(f"[PredictionAgent] Running under trace_id={self.trace_id}")

            #with self.tracer.start_as_current_span("orchestration_logic",context=ctx):
            logging.info(f"[PredictionAgent] Running under trace_id={self.trace_id}")
            evaluation_output = self.evaluate_prediction(prediction_output)

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
                phase="Phase4_Governance",
                event_type="GOVERNANCE_EVALUATION_FAILED",
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
            logging.info("evaluation started")
            prediction_id= str(uuid.uuid4())
            prediction_evidence_id = prediction_output.get('prediction_evidence_id')
            model_evidence_id = prediction_output.get('model_evidence_id')
            model_info = prediction_output.get('model_info', {})
            xai_artifact= prediction_output.get("xai_artifact",{})
            probabilities= prediction_output.get("probabilities",[])
            confidence_score = prediction_output.get("confidence_scores", 0.0)
            logging.info(f"prediction_id: {prediction_id},xai_artifact: {xai_artifact},probabilties: {probabilities},confidence_score: {confidence_score}")

            #Policy 1 : prohibited attribute check
            policy_input = {
               "confidence_scores": confidence_score,
                
                # For prohibited_attribute_check policy  
                "local_top_drivers": dict(xai_artifact.get("local_top_drivers", [])),
                "risk_factors": xai_artifact.get("risk_factors", []),
                
                # For provenance_compliance_check policy
                "model_version": model_info.get('mlflow_model_version', 'unknown'),
                "model_name": model_info.get('mlflow_model_name', 'unknown'),
                "features_used": list(dict(xai_artifact.get("local_top_drivers", [])).keys()),
                
                # Additional context
                "prediction_label": 1 if prediction_output.get("is_default_risk", False) else 0,
                "probability_class_1": probabilities[1] if len(probabilities) > 1 else probabilities[0]
            }

            
            #evaluate all active polciies via PolicyEngine
            policy_results= self.policy_engine.evaluate(policy_input)

            #aggregate results
            violations=[]
            confidence_status= "pass"

            for policy_name,opa_result in policy_results.items():
                policy_decision=opa_result.get("result",{})
                if policy_name.lower()=="oversight_threshold_check":
                    confidence_status= "pass" if policy_decision.get("allow",True) else "human_review_required"
                if policy_name.lower() == "prohibited_attribute_check":
                     if "prohibited_attribute_check" in policy_name:
                    # Check if prohibited attributes policy failed
                        if not policy_decision.get("allow", False):
                            violations.append({
                                "policy_name": policy_name,
                                "violation_type": "PROHIBITED_ATTRIBUTES",
                                "description": "Protected attributes have excessive influence",
                                "opa_result": policy_decision
                            })

            
            has_violations=len(violations)>0
            needs_human_review=confidence_status=="human_review_required"

            final_decision="APPROVED" if not has_violations and not needs_human_review else "BLOCKED"
            

            governance_data = self._save_governance_report({
                "trace_id": self.trace_id,
                "prediction_id": prediction_id,
                "prediction_evidence_id": prediction_evidence_id,
                "model_evidence_id": model_evidence_id,
                "policy_results": policy_results,
                "violations": violations,
                "confidence_status": confidence_status,
                "final_decision": final_decision,
                "evaluation_timestamp": datetime.now().isoformat()
            })
            os.makeidrs("src/governance/reports",exist_ok=True)
            report_path = f"src/governance/reports/governance_{self.client_transaction_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path,"w") as f:
              json.dump(governance_data,f,indent=2,default=str)
            logging.info(f"Governance report saved to: {report_path}")
             
             # REGISTER GOVERNANCE DECISION AS EVIDENCE
            governance_evidence_id = register_artifact("evidence_pointer", {
                "kind": "GOVERNANCE_DECISION",
                "dvc_rev": self.get_git_revision(),
                "dvc_path":report_path,
                "sha256": generate_sha256_hash(open(governance_data,"rb").read()),
                "prediction_evidence_id": prediction_evidence_id,
                "model_evidence_id": model_evidence_id,
                "final_decision": final_decision,
                "violation_count": len(violations)
            }, self.client_transaction_id)

            governance_event_id=log_event(
               agent_name=self.agent_name,
               trace_id=self.trace_id,
               client_transaction_id=self.client_transaction_id,
               phase="Phase3_Governance",
               event_type="POLICY_EVALUATION_COMPLETED",
               summary=f"Critic Agent computed decision : {final_decision}",
               extra_payload={
                "prediciton_id":prediction_id,
                "final_decision":final_decision,
                "policy_results_summary": {k: v.get('result', {}) for k, v in policy_results.items()},
                "violations": violations,
                "confidence_status": confidence_status,
                "governance_evidence_id": governance_evidence_id,
                "prediction_evidence_id": prediction_evidence_id,
                "model_evidence_id": model_evidence_id
               }
            )
            register_artifact("event_link", {
                "event_id": governance_event_id,
                "evidence_id": governance_evidence_id
            }, self.client_transaction_id)

            if prediction_evidence_id:
                register_artifact("event_link", {
                    "event_id": governance_event_id,
                    "evidence_id": prediction_evidence_id
                }, self.client_transaction_id)

            if model_evidence_id:
                register_artifact("event_link", {
                    "event_id": governance_event_id,
                    "evidence_id": model_evidence_id
                }, self.client_transaction_id)

             #trigger human escalation if needed
            if final_decision in ["BLOCKED", "REQUIRES_HUMAN_REVIEW"]:
                escalate_human_review(agent_name=self.agent_name,client_transaction_id=self.client_transaction_id,trace_id=self.trace_id,violations=violations,confidence=policy_input["confidence"])
            return {
                "trace_id": self.trace_id,
                "prediction_id": prediction_id,
                "policy_results": policy_results,
                "violations": violations,
                "confidence_status": confidence_status,
                "final_decision": final_decision,
                "governance_evidence_id": governance_evidence_id,
                "governance_event_id": governance_event_id
            }
        
        except Exception as e:
            logging.info(f"CriticAgent evaluation failed: {e}")
            raise CGAgentException(e, sys)

if __name__=="__main__":
    prediction_output={
  "predictions": [
    0
  ],
  "probabilities": [
    0.2077976492653883,
    0.2077976492653883
  ],
  "confidence_scores": 0.4077976492653883,
  "is_default_risk": False,
  "prediction_label": 0,
  "probability_class_0": 0.2077976492653883,
  "probability_class_1": 0.2077976492653883,
  "model_version": "gb_model_v1",
  "xai_artifact": {
    "global_feature_importance": {
      "LIMIT_BAL": 0.04121871755189181,
      "SEX": 0.14031442993192866,
      "EDUCATION": 0.3150085780790633,
      "MARRIAGE": 0.18743889517404794,
      "AGE": 0.003614263620074616,
      "PAY_0": 0.5748784556292217,
      "PAY_2": 0.015505985053678487,
      "PAY_3": 0.09054606099165365,
      "PAY_4": 0.11232378899299861,
      "PAY_5": 0.10041865415727587,
      "PAY_6": 0.23185615063444714,
      "BILL_AMT1": 0.021869664103199002,
      "BILL_AMT2": 0.0012531385176473384,
      "BILL_AMT3": 0.0003294297405228622,
      "BILL_AMT4": 0.00021051940365329416,
      "BILL_AMT5": 0.010390254456450056,
      "BILL_AMT6": 0.002457520615905833,
      "PAY_AMT1": 0.04294931828259888,
      "PAY_AMT2": 0.024310463268718597,
      "PAY_AMT3": 0.0014000659509248211,
      "PAY_AMT4": 0.006426919768574474,
      "PAY_AMT5": 0.004047400777552523,
      "PAY_AMT6": 0.022675288137045523,
      "total_months_late": 0.8348062651841427,
      "utilization_rate": 0.10635626633123582
    },
    "local_top_drivers": [
      [
        "total_months_late",
        -0.8348062651841427
      ],
      [
        "PAY_0",
        -0.5748784556292217
      ],
      [
        "EDUCATION",
        -0.3150085780790633
      ],
      [
        "PAY_6",
        -0.23185615063444714
      ],
      [
        "MARRIAGE",
        0.18743889517404794
      ]
    ],
    "expected_value": 0.475287785450749,
    "causal_graph": {
      "LIMIT_BAL": "weak",
      "SEX": "moderate_positive",
      "EDUCATION": "strong_negative",
      "MARRIAGE": "strong_positive",
      "AGE": "weak",
      "PAY_0": "strong_negative",
      "PAY_2": "weak",
      "PAY_3": "moderate_negative",
      "PAY_4": "strong_negative",
      "PAY_5": "strong_negative",
      "PAY_6": "strong_negative",
      "BILL_AMT1": "weak",
      "BILL_AMT2": "weak",
      "BILL_AMT3": "weak",
      "BILL_AMT4": "weak",
      "BILL_AMT5": "weak",
      "BILL_AMT6": "weak",
      "PAY_AMT1": "weak",
      "PAY_AMT2": "weak",
      "PAY_AMT3": "weak",
      "PAY_AMT4": "weak",
      "PAY_AMT5": "weak",
      "PAY_AMT6": "weak",
      "total_months_late": "strong_negative",
      "utilization_rate": "moderate_positive"
    },
    "risk_factors": [
      "total_months_late",
      "PAY_0",
      "EDUCATION",
      "PAY_6",
      "MARRIAGE"
    ]
  },
  "artifact_path": "src/model/predictions/PREDICTION_AGENT_V1.0_20251123_151823",
  "input_feature_hash": "4fc1b9b9bb2ed4556868ceddceb4871c222ebdeb900a934a5ed2c00524984199",
}
  
    critic = CriticAgent(trace_id="trace_123", client_transaction_id="pred_456")
    result = critic.run(prediction_output)  # ✅ Now accepts the prediction data