from logger import logging
from exception import CGAgentException
from src.agent.data_assurance_agent import DataAssuranceAgent
from src.agent.training_agent import TrainingAgent
import shortuuid
from src.traceability.audit_logger import log_event,register_artifact

class Training_pipeline:
    agent_name="TRAINING_PIPELINE"
    def __init__(self,trace_id):
        self.trace_id=trace_id
       
        
    def run_raining_pipeline(self):
        try:
            logging.info(f"[TrainingPipelinet] trace_id={self.trace_id}")
            agent= DataAssuranceAgent(trace_id=self.trace_id,client_transaction_id="DATA-AUDIT-V1.0")
            data_result= agent.run()

            dataset_train_val_id= data_result['result']["dataset_id"]

            agent=TrainingAgent(trace_id=self.trace_id,client_transaction_id="TRAINING_MODEL_V1.0",dataset_id=dataset_train_val_id)
            training_result = agent.run()

            res={"status":"training_pipeline_succesful"}
            return training_result
        except Exception as e:
            res={"status":"training_pipeline_failed"}
            return training_result

if __name__ == "__main__":
    agent =Training_pipeline(trace_id="DATA-PIPELINE-V1.0")
    res=agent.run()
    print(res)
    


