from exception import CGAgentException
from src.agent.prediction_agent import PredictionAgent
from src.agent.critic_agent import CriticAgent
from src.traceability.telemetry import get_tracer, extract_trace_context,inject_trace_context
import os,sys
import logging
from src.traceability.audit_logger import log_event
import shortuuid

class OrchestratorAgent:
    def __init__(self,agent_name,trace_id,client_transaction_id= str(shortuuid.uuid())):
        self.trace_id=trace_id
        self.client_transaction_id=client_transaction_id
        self.tracer=get_tracer()
        self.agent_name="Orchestrator_Agent"
    
    def run(self,payload):
        try:
            logging.info(f"[OrchestratorAgent] Starting with trace_id={self.trace_id}")
            ctx= extract_trace_context(payload)

            with self.tracer.start_as_current_span("orchestration_logic",context=ctx) as span:
                
                payload = inject_trace_context(payload)

                #prediction agent
                logging.info(f"[PredictionAgent] trace_id={self.trace_id}, txn_id={self.client_transaction_id}")
                prediction=PredictionAgent(self.trace_id,self.client_transaction_id).run(payload)
               
                logging.info("prediction completed successfully")
   
                #critic agent
                critic=CriticAgent(self.trace_id,self.client_transaction_id)
                logging.info("critic chceked done successfully")
                return {"trace_id": self.trace_id,
                "client_transaction_id": self.client_transaction_id,"agent":self.agent_name,"status":"success","prediction":prediction,"critic":critic}
                
        except Exception as e:
                log_event(
                    trace_id=self.pipeline_trace_id,
                    client_transaction_id=self.client_transaction_id,
                    agent_name=self.agent_name,
                    phase="Orchestration",
                    event_type="ERROR",
                    summary=str(e)
                )
                return {
                    "agent":self.agent_name,
                    "status":"failed",
                    "error":str(e)
                }
            