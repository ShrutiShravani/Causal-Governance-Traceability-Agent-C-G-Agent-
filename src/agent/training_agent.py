from pathlib import Path
import logging
import os,sys
import pickle
from src.traceability.audit_logger import log_event,register_artifact
import pandas as pd
from src.causal_engine.trainer import TrainingEngine
from src.causal_engine.explainer import XAIEngine
from typing import Dict,Any
from exception import CGAgentException
import uuid

class TrainingAgent:
    agent_name="Training_Agent"
    model_path="src/model/trained_gb_model.pkl"
    X_train_path="data/datasets/data_ingested/X_train_credit_clean.csv"
    y_train_path="data/datasets/data_ingested/y_train_credit_clean.csv"
    X_val_path="data/datasets/data_ingested/X_val_credit_clean.csv"
    y_val_path="data/datasets/data_ingested/y_val_credit_clean.csv"
    confidence_learning_report="data\quality\confident_learning_report.csv"
    
    logging.info("TrainingAgent initialized.")

    #we have to get trace_id from orchetsrator agent same trcae_id for thta aprticular excution
    def __init__(self, trace_id:str,client_transaction_id:str):
        self.pipeline_trace_id = str(uuid.uuid4()).replace('-', '')[:32]
        self.client_transaction_id= client_transaction_id

    def run(self,payload:dict=None)->dict:
        """
        Executes the FULL prediction lifecycle:
        1. Train → 2. Load → 3. Predict → 4. SHAP Explanations
        """
        try:
            logging.info("PredictionAgent: Starting workflow...")
            #train model
            training_output=self._execute_training_pipleine(self.X_train_path,
                self.y_train_path,
                self.X_val_path,
                self.y_val_path,
                self.confidence_learning_report)
           

            return {
                "agent":self.agent_name,
                "status":"success",
                "result":training_output,

            }
        except Exception as e:
            log_event(
                trace_id=self.pipeline_trace_id,
                client_transaction_id=self.client_transaction_id,
                agent_name=self.agent_name,
                phase="Phase2_ModelTraining",
                event_type="ERROR",
                summary=str(e)
            )
            return {
                "agent":self.agent_name,
                "status":"failed",
                "error":str(e)
            }
    
    def _execute_training_pipleine(self,X_train_path:str,
        y_train_path:str,
        X_val_path:str,
        y_val_path:str,
        confidence_learning_report:str)->Dict[str,Any]:
        
        try:

            result= TrainingEngine.train_causal_model(X_train_path,y_train_path,X_val_path,y_val_path,confidence_learning_report)
            
            xai_artifact= XAIEngine.generate_explanations(model_artifact_path=self.model_path, X_data_path=self.X_train_path)
            return {
                    "status": "training_successful",
                    "training_artifacts": result,
                    "xai_artifact":xai_artifact
                }

        except Exception as e:
            raise CGAgentException(e, None) 