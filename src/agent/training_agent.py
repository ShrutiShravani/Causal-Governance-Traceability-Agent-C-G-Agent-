from pathlib import Path
from logger import logging
import os,sys
import pickle
from src.traceability.audit_logger import log_event,register_artifact
import pandas as pd
from src.causal_engine.trainer import TrainingEngine
from src.causal_engine.explainer import XAIEngine
from typing import Dict,Any
from exception import CGAgentException
import uuid
from src.data_pipeline.data_hashing import generate_sha256_hash,dataframe_to_stable_bytes
import json
import subprocess

class TrainingAgent:
    agent_name="Training_Agent"
    model_path="src/model/trained_gb_model.pkl"
    X_train_path="data/datasets/data_ingested/X_train_credit_clean.csv"
    y_train_path="data/datasets/data_ingested/y_train_credit_clean.csv"
    X_val_path="data/datasets/data_ingested/X_val_credit_clean.csv"
    y_val_path="data/datasets/data_ingested/y_val_credit_clean.csv"
    confidence_learning_report="data/quality/confident_learning_report.csv"
    
    
    logging.info("TrainingAgent initialized.")

    #we have to get trace_id from orchetsrator agent same trcae_id for thta aprticular excution
    def __init__(self, trace_id:str,client_transaction_id:str,dataset_id:str = None):
        self.pipeline_trace_id = trace_id 
        self.client_transaction_id=client_transaction_id
        self.dataset_id = dataset_id
    def get_git_revision(self):
        return subprocess.check_output(["git","rev-parse","HEAD"]).decode("utf-8").strip()

    def run(self,payload:dict=None)->dict:
        """
        Executes the FULL prediction lifecycle:
        1. Train → 2. Load → 3. Predict → 4. SHAP Explanations
        """
        try:
            logging.info("PredictionAgent: Starting workflow...")

            training_start_event_id = log_event(
                trace_id=self.pipeline_trace_id,
                client_transaction_id=self.client_transaction_id,
                agent_name=self.agent_name,
                phase="Phase2_ModelTraining",
                event_type="MODEL_TRAINING_STARTED",  # ✅ CHANGED: STARTED not COMPLETED
                summary="Model training initiated",
                extra_payload={
                    "model_type": "GradientBoosting",
                    "dataset_id": self.dataset_id  # ✅ Reference to dataset
                }
            )

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
            logging.info("Starting model training...")
            result= TrainingEngine.train_causal_model(X_train_path,y_train_path,X_val_path,y_val_path,confidence_learning_report)
             
            run_id=result['run_id']
            model_path= result['model_path']
            mlflow_model_name= result["model_name"]
            mlflow_model_version= result["model_version"]

            # 3. REGISTER MODEL ARTIFACT
            # 2. REGISTER MODEL AS EVIDENCE
            model_evidence_id = register_artifact("evidence_pointer", {
                "kind": "TRAINED_MODEL",
                "dvc_rev": self.get_git_revision(),
                "dvc_path": model_path,
                "sha256": generate_sha256_hash(open(model_path, "rb").read()),
                "mlflow_run_id": run_id,
                "mlflow_model_name": mlflow_model_name,  # Real MLflow name
                "mlflow_model_version": mlflow_model_version,  # Real MLflow version
            }, self.client_transaction_id)

            # 4. LOG TRAINING COMPLETION 
            training_complete_event_id = log_event(
            trace_id=self.pipeline_trace_id,
            client_transaction_id=self.client_transaction_id,
            agent_name=self.agent_name,
            phase="Phase2_ModelTraining",
            event_type="MODEL_TRAINING_COMPLETED", 
            summary=f"Model training completed - Best ROC AUC: {result['best_epoch']['roc_auc']:.4f}",
            extra_payload={
                 "model_evidence_id": model_evidence_id,
                    "mlflow_run_id": run_id,
                    "mlflow_model_name": mlflow_model_name,
                    "mlflow_model_version": mlflow_model_version,
                    "best_roc_auc": result['best_epoch']['roc_auc'],
                    "model_type": "GradientBoosting",
                   
            }
            )
            
            # ADD: Link training to dataset (CRITICAL for lineage)
            # You'll need to get dataset_id from DataAssuranceAgent
            register_artifact("event_link", {
                "event_id": training_complete_event_id,
                "dataset_id": self.dataset_id, 
                "evidence_id": model_evidence_id
                # dataset_id will be added in pipeline orchestration
            })
           
        # 6. GENERATE XAI
            logging.info("Generating XAI explanations...")
            
           
            xai_artifact= XAIEngine.generate_explanations(model_artifact_path=self.model_path,X_data_path=self.X_train_path,run_id=run_id,base_path="src/model/audit_artifact")
            
            
            # 6. REGISTER XAI AS EVIDENCE
            xai_evidence_id = register_artifact("evidence_pointer", {
                "kind": "XAI_EXPLANATION_REPORT",
                "dvc_rev": self.get_git_revision(),
                "dvc_path": xai_artifact['artifact_path'],
                "sha256": generate_sha256_hash(json.dumps(xai_artifact).encode())
            },self.client_transaction_id)

            xai_event_id = log_event(
            trace_id=self.pipeline_trace_id,
            client_transaction_id=self.client_transaction_id, 
            agent_name=self.agent_name,
            phase="Phase2_ModelTraining",
            event_type="XAI_EXPLANATIONS_GENERATED",
            summary="Model explanations and visualizations created",
            extra_payload={
                "model_artifact_id": model_evidence_id,
                "xai_evidence_id": xai_evidence_id,
                "top_features_count": len(xai_artifact.get('top_drivers', []))
            })

            # Step 3: Generate visualization
            logging.info("Generating XAI visualization...")
            XAIEngine.generate_visualization(xai_artifact,run_id=run_id,base_path= "src/model/explainer_artifact")
            
              # 8. LINK XAI EVENT TO XAI EVIDENCE
            register_artifact("event_link", {
                "event_id": xai_event_id,
                "evidence_id": xai_evidence_id
            },self.client_transaction_id)

            # 9. LINK XAI TO MODEL (both are evidence types)
            register_artifact("event_link",{
                "event_id": xai_event_id, 
                "evidence_id": model_evidence_id  # Link XAI to the model it explains
            }, self.client_transaction_id)

                        
            return {
                    "status": "training_successful",
                    "training_artifacts": result,
                    "xai_artifact": xai_artifact,
                    "model_evidence_id": model_evidence_id,
                    "xai_evidence_id": xai_evidence_id,
                    "mlflow_run_id": run_id,
                    "event_id": training_complete_event_id,
                    "mlflow_model_name": mlflow_model_name,
                    "mlflow_model_version": mlflow_model_version
                }

        except Exception as e:
            log_event(
            trace_id=self.pipeline_trace_id,
            client_transaction_id=self.client_transaction_id,
            agent_name=self.agent_name, 
            phase="Phase2_ModelTraining",
            event_type="MODEL_TRAINING_FAILED",
            summary=f"Training failed: {str(e)}",
            extra_payload=None
        )
            raise CGAgentException(e, None) 
