from pathlib import Path
from logger import logging
import os,sys
import pickle
from src.traceability.audit_logger import log_event,register_artifact
import pandas as pd
from src.causal_engine.explainer import XAIEngine
from typing import Dict,Any
from exception import CGAgentException
from src.data_pipeline.data_hashing import generate_sha256_hash,dataframe_to_stable_bytes
import json
from src.traceability.telemetry import get_tracer, inject_trace_context,extract_trace_context
import numpy as np
from src.data_pipeline.data_cleaning import categorical_consolidation,clip_outliers,handle_missing_values,impute_missing_values,prepare_data,check_data_imbalance,handle_imbalance_data
from src.data_pipeline.feature_engineering import run_feature_engineering
from src.data_pipeline.feature_engineering import feature_engineering,create_utlization_rate,encode_categorical_features,scale_numeric_features,get_features



class PredictionAgent:
    agent_name="Prediction_Agent"
    model_path="src/model/trained_gb_model.pkl"
    X_input_path="data/datasets/data_ingested/X_test_credit_clean.csv"
    logging.info("PredictionAgent initialized.")

    #we have to get trace_id from orchetsrator agent same trcae_id for thta aprticular excution
    def __init__(self, trace_id:str,client_transaction_id:str):
        self.trace_id = trace_id
        self.client_transaction_id= client_transaction_id
        self.tracer=get_tracer()
        logging.info(f"PredictionAgent initialized with trace_id={self.trace_id}")

    def run(self,payload:dict=None)->dict:
        """
        Executes the FULL prediction lifecycle:
         Predict → SHAP Explanations
        """
        try:
            logging.info("PredictionAgent: Starting workflow...")
            ctx= extract_trace_context(payload)
            # do prediction
            print(f"[PredictionAgent] Running under trace_id={self.trace_id}")

            with self.tracer.start_as_current_span("orchestration_logic",context=ctx):
                logging.info(f"[PredictionAgent] Running under trace_id={self.trace_id}")
                prediction_output = self._execute_prediction_and_explanation()

                return {
                    "agent":self.agent_name,
                    "status":"success",
                    "trace_id": self.trace_id,
                    "client_transaction_id": self.client_transaction_id,
                    "prediction_output":prediction_output

                }
        except Exception as e:
            log_event(
                trace_id=self.trace_id,
                client_transaction_id=self.client_transaction_id,
                agent_name=self.agent_name,
                phase="Phase3_Prediction",
                event_type="ERROR",
                summary=str(e)
            )
            return {
                "agent":self.agent_name,
                "status":"failed",
                "error":str(e)
            }

    def _execute_prediction_and_explanation(self)->Dict[str,Any]:
        try:
            with open(self.model_path,"rb") as f:
                model=pickle.load(f)
            df=pd.read_csv(self.X_input_path)
            features= df.columns[:-1]
            

            df=categorical_consolidation(df,features)
        
            # Step 2: Clip outliers
            df = clip_outliers(df)
            
            # Step 3: check for missing values
            df = handle_missing_values(df)

            # Step 4: Impute missing values
            df= impute_missing_values(df)

            # Step 5: Chcek for class imbalance
            is_imbalanced= check_data_imbalance(df)
            if is_imbalanced:
                df=handle_imbalance_data(df,target_col="default.payment.next.month",method="upsample")
            
            # Step 6 :prepare train data
            df,ids=prepare_data(df,features)
            data=feature_engineering(df)
            new_data=create_utlization_rate(data)
            new_data=encode_categorical_features(new_data,features)
            new_data=scale_numeric_features(new_data,features)
  
            df_clean=df.drop(columns=["ID","default.payment.next.month"],errors="ignore")
           
            #make predicitons
            pred=model.predict(df_clean)
            prob= model.predict_proba(df_clean)
            confidence_scores = np.max(prob, axis=1)
            avg_confidence = float(np.mean(confidence_scores))
            input_hash=generate_sha256_hash(
                dataframe_to_stable_bytes(df_clean))
            
    
            log_event(
                trace_id=self.trace_id,
                client_transaction_id=self.client_transaction_id,
                agent_name=self.agent_name,
                phase="Phase3_PredictionPipeline",
                event_type="PREDICTION_EXECUTED",
                summary="Model prediction generated successfully.",
                extra_payload={
                    "model_version": "gb_model_v1",      
                    "prediction_label": int(pred),
                    "prediction_probability": float(prob),
                    "avg_confidence": avg_confidence,
                    "sample_confidences": confidence_scores.tolist()[:5]
                }
            )

            xai_artifact= XAIEngine.generate_explanations(model_artifact_path=self.model_path, X_data_path=self.X_input_path)
        
            xai_artifact_path="src/model/explainer"
            os.makedirs(xai_artifact_path,exist_ok=True)
            out_dir = os.path.join(xai_artifact_path,"xai_artifact.json")

            with open (out_dir,"wb") as f:
                json.dump(xai_artifact,"f",indent=4)

            #register evidence pointer
            register_artifact("evidence_pointer",{
                    "kind":"Predcition Data",
                    "xai_artifact_path":str(xai_artifact_path), #here add relative dvc path wher eit is versoined
                    "input_feature_hash": input_hash,
                }
            )
            log_event(
                trace_id = self.trace_id,
                client_transaction_id = self.client_transaction_id,
                agent_name = self.agent_name,
                phase = "Phase3_PredictionPipeline",
                event_type = "LOCAL_EXPLANATION_GENERATED",
                summary = "Generated SHAP + Causal Explanation",
                extra_payload = {
                    "top_drivers": xai_artifact["local_top_drivers"],
                    "expected_value": xai_artifact["expected_value"],
                    "causal_graph": xai_artifact["causal_graph"],
                    "risk_factors": xai_artifact["risk_factors"]
                }
            )
            return {
                "predicitons":pred.tolist(),
                "probabilities":prob.tolist(),
                "confidence": avg_confidence,
                "xai_artifact":xai_artifact
            }
        except Exception as e:
            raise CGAgentException(e,sys)

 
