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
#from src.traceability.telemetry import get_tracer, inject_trace_context,extract_trace_context
import numpy as np
from src.data_pipeline.data_cleaning import categorical_consolidation,clip_outliers,handle_missing_values,impute_missing_values,prepare_data
from src.data_pipeline.feature_engineering import feature_engineering,create_utlization_rate,scale_numeric_features
import uuid
import yaml
import datetime
import joblib
import subprocess
import mlflow
from src.traceability.audit_logger import get_model_version_info

class PredictionAgent:
    agent_name="Prediction_Agent"
    model_path="src/model/trained_gb_model.pkl"
    pickle_path="data/transformed/scaling_params.pkl"
    #X_input_path="data/datasets/data_ingested/X_test_credit_clean.csv"
    logging.info("PredictionAgent initialized.")

    #we have to get trace_id from orchetsrator agent same trcae_id for thta aprticular excution
    def __init__(self, trace_id:str,client_transaction_id:str):
        self.trace_id = trace_id
        self.client_transaction_id= client_transaction_id
        
        #self.tracer=get_tracer()
        logging.info(f"PredictionAgent initialized with trace_id={self.trace_id}")
    
    def get_git_revision(self):
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    def run(self,payload:dict=None)->dict:
        """
        Executes the FULL prediction lifecycle:
         Predict → SHAP Explanations
        """
        try:
            logging.info("PredictionAgent: Starting workflow...")
            """
            ctx= extract_trace_context(payload)
            # do prediction
            print(f"[PredictionAgent] Running under trace_id={self.trace_id}")

            with self.tracer.start_as_current_span("orchestration_logic",context=ctx):
            """
            logging.info(f"[PredictionAgent] Running under trace_id={self.trace_id}")
            
            if payload and 'input_data_path' in payload:
                input_path= payload['input_data_path']
            else:
                input_path = "test_data.csv"
            
            prediction_output = self._execute_prediction_and_explanation(input_path)

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
    
    @staticmethod
    def load_yaml(path:str):
        """load yaml file
        """
        with open(path,'r') as f:
            return yaml.safe_load(f)
    
    def _execute_prediction_and_explanation(self,input_path:str)->Dict[str,Any]:
        try:
            #getting modle info forom sql
            model_info= get_model_version_info()

            with open(self.model_path,"rb") as f:
                model=pickle.load(f)

            df=pd.read_csv(input_path)
            
            features=PredictionAgent.load_yaml(r"src\config\features.yaml")
            
        
            df=categorical_consolidation(df,features)
        
            # Step 2: Clip outliers
            df = clip_outliers(df)
            
            # Step 3: check for missing values
            df = handle_missing_values(df)

            # Step 4: Impute missing values
            df= impute_missing_values(df)

            data=feature_engineering(df)
            new_data=create_utlization_rate(data)
            

            scaling_params = joblib.load(self.pickle_path)

            # Apply scaling using SAVED parameters (no NaN!)
            for col in scaling_params['columns']:
                if col in new_data.columns:
                    new_data[col]=(new_data[col]- scaling_params['means'][col]) / scaling_params['stds'][col]
            
            logging.info(f"{new_data.columns}")
  
            df_clean=new_data.drop(columns=["ID"],errors="ignore")
           
            logging.info(f"Input data shape:{df_clean.shape}")
            logging.info(f"Input data columns:{df_clean.columns.tolist()}")
            logging.info(f"Input data sample:\n {df_clean.head(2)}")

            #make predicitons
            pred=model.predict(df_clean)
            prob= model.predict_proba(df_clean)

            #chcek rpediciton outputs
            logging.info(f"Raw predicitons shape:{pred.shape}")
            logging.info(f"Probability shape:{prob.shape}")
            logging.info(f"Probability sample:{prob[:3]}")

            #handle different prediciton output formats
            """
            if len(prob.shape)==1:
                #singl smaple prediciton
                confidence_scores=np.array([prob.max()])
                pred= np.array([pred]) if np.isscalar(pred) else pred
            elif len(prob.shape)==2:
                confidence_scores = np.max(prob,axis=1)
            else:
                raise ValueError(f"Unexpected probability shape: {prob.shape}")
            avg_confidence = float(np.mean(confidence_scores))
            """
            if len(prob.shape)==1:
                confidence_score= float(prob.max())
                prediction_label = int(pred[0]) if hasattr (pred,'__len__') else int(pred)
                probability_class_1 = float(prob[1]) if len(prob) > 1 else float(prob[0])
                probability_class_0 = float(prob[0]) if len(prob) > 1 else float(prob[0])
                probabilities = [probability_class_0, probability_class_1]

            else:
                # Multiple samples (shouldn't happen for single client)
                confidence_score = float(np.max(prob, axis=1)[0])
                prediction_label = int(pred[0])
                probability_class_0 = float(prob[0][0])
                probability_class_1 = float(prob[0][1])
                probabilities = [probability_class_0, probability_class_1]


            logging.info(f"Confidence scores: {confidence_score}")
          
            input_hash=generate_sha256_hash(
                dataframe_to_stable_bytes(df_clean))
            
            temp_pred_path= f"src/model/predictions/{self.client_transaction_id}_{timestamp}"
            os.makedirs(temp_pred_path,exist_ok=True)
            logging.info(f"Prediction artifacts will be saved to: {temp_pred_path}")
            
            # ✅ CREATE SPECIFIC BASE PATHS FOR XAI TO CREATE ORGANIZED FILES
            xai_explanation_base = os.path.join(temp_pred_path, "xai_explanation")
            xai_viz_base = os.path.join(temp_pred_path, "xai_visualization")
            prediction_file_path = os.path.join(temp_pred_path,"prediction_results.json")
            prediction_file_path = os.path.join(temp_pred_path, "prediction_results.json")  # PredictionAgent created
            complete_output_path = os.path.join(temp_pred_path, "complete_prediction_output.json")  # PredictionAgent created
            xai_artifact_path = f"{xai_explanation_base}_detailed.json"  # ✅ XAIExpliner created: "src/model/predictions/CLIENT_A_123456/xai_explanation_detailed.json"
            xai_plot_path = f"{xai_viz_base}_plot.png"  # ✅ XAIExpliner created: "src/model/predictions/CLIENT_A_123456/xai_visualization_plot.png"
            
            
            with open(prediction_file_path, 'w') as f:
                json.dump({
                    "predictions": pred.tolist(),
                    "probabilities": probabilities,
                    "confidence_score": confidence_score,
                    "prediction_label": prediction_label,
                    "input_feature_hash": input_hash,
                    "timestamp": datetime.now().isoformat(),
                    "client_transaction_id": self.client_transaction_id
                }, f, indent=2)
    
            prediction_event_id=log_event(
                trace_id=self.trace_id,
                client_transaction_id=self.client_transaction_id,
                agent_name=self.agent_name,
                phase="Phase3_PredictionPipeline",
                event_type="PREDICTION_EXECUTED",
                summary="Model prediction generated successfully.",
                extra_payload={
                    "prediction_label": prediction_label,
                    "probability_class_1": probability_class_1,
                    "confidence_score": confidence_score,
                    "is_default_risk": prediction_label == 1,
                    "input_feature_hash": input_hash,
                    "model_evidence_id": model_info.get('evidence_id'),  
                    "mlflow_model_name": model_info.get('mlflow_model_name'),
                    "mlflow_model_version": model_info.get('mlflow_model_version'),
                    "mlflow_run_id": model_info.get('mlflow_run_id')
                }
            )

            #regiser evidence 
            prediction_evidence_id = register_artifact("evidence_pointer", {
                "kind": "PREDICTION_RESULT",
                "dvc_rev": self.get_git_revision(),
                "dvc_path": prediction_file_path,
                "sha256": input_hash,
                "model_evidence_id": model_info.get('evidence_id'),  # ✅ REFERENCE TO MODEL
                "mlflow_model_name": model_info.get('mlflow_model_name'),
                "mlflow_model_version": model_info.get('mlflow_model_version'),
            }, self.client_transaction_id)
            

            register_artifact("event_link", {
                "event_id": prediction_event_id,
                "evidence_id": prediction_evidence_id
            }, self.client_transaction_id)

            #LINK PREDICTION TO MODEL 
            register_artifact("event_link",{
                "event_id": prediction_event_id,
                "evidence_id": model_info.get('evidence_id'),
            },self.client_transaction_id)
           
            #register prediction evidence
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
         
            #genertae xai explanation
            xai_artifact= XAIEngine.generate_explanations(model_artifact_path=self.model_path,X_data_path=None,X_data=df_clean,base_path=xai_explanation_base)
        
            
            XAIEngine.generate_visualization(xai_artifact,run_id=None,base_path=xai_viz_base)

            #register evidence pointer
            xai_evidence_id=register_artifact("evidence_pointer",{
                    "kind":"PREDICTION_XAI_EXPLANATION",
                    'dvc_rev': self.get_git_revision(),
                    "dvc_path":str(xai_artifact_path), #here add relative dvc path wher eit is versoined
                    "sha256": generate_sha256_hash(json.dumps(xai_artifact).encode()),
                },self.client_transaction_id
            )
            
            #register xai visualization

            xai_viz_evidence_id = register_artifact("evidence_pointer", {
            "kind": "XAI_VISUALIZATION", 
            "dvc_rev": self.get_git_revision(),
            "dvc_path": xai_plot_path,  # PECIFIC FILE
            "sha256": generate_sha256_hash(open(xai_plot_path, "rb").read()),
            "xai_evidence_id": xai_evidence_id,
            }, self.client_transaction_id)
        
            xai_event_id=log_event(
                trace_id = self.trace_id,
                client_transaction_id = self.client_transaction_id,
                agent_name = self.agent_name,
                phase = "Phase3_PredictionPipeline",
                event_type = "XAI_EXPLANATIONS_GENERATED",
                summary = "Generated SHAP + Causal Explanation",
                extra_payload = {
                    "top_drivers": xai_artifact["local_top_drivers"],
                    "expected_value": xai_artifact["expected_value"],
                    "causal_graph": xai_artifact["causal_graph"],
                    "risk_factors": xai_artifact["risk_factors"],
                    "confidence_scores":confidence_score,
                    "is_default_risk": prediction_label == 1
                }
                
            )

            viz_event_id = log_event(
                    trace_id=self.trace_id,
                    client_transaction_id=self.client_transaction_id,
                    agent_name=self.agent_name,
                    phase="Phase3_PredictionPipeline", 
                    event_type="XAI_VISUALIZATION_GENERATED",
                    summary="XAI visualization plot created",
                    extra_payload={
                        "plot_path": xai_plot_path,
                        "xai_evidence_id": xai_evidence_id,
                        "xai_viz_evidence_id": xai_viz_evidence_id,
                        "prediction_evidence_id": prediction_evidence_id
                    }
                )


            register_artifact("event_link", {
                "event_id": xai_event_id,
                "evidence_id": xai_evidence_id
            }, self.client_transaction_id)

            register_artifact("event_link", {
                "event_id": xai_event_id,
                "evidence_id": prediction_evidence_id
            }, self.client_transaction_id)

            register_artifact("event_link", {
                "event_id": xai_event_id,
                "evidence_id": model_info.get('evidence_id')  #LINK XAI TO MODEL
            }, self.client_transaction_id)

            register_artifact("event_link", {
                "event_id": viz_event_id,
                "evidence_id": xai_viz_evidence_id
            }, self.client_transaction_id)

            #LINK VISUALIZATION TO XAI EXPLANATION 
            register_artifact("event_link", {
                "event_id": viz_event_id,
                "evidence_id": xai_evidence_id
            }, self.client_transaction_id)


            artifact_paths= {
                "prediction_file": prediction_file_path,
                "complete_output": complete_output_path, 
                "xai_explanation": xai_artifact_path,
                "xai_visualization": xai_plot_path
            }
            prediction_output = {
            "predictions": pred.tolist(),
            "probabilities": probabilities,  # ✅ Now includes both classes
            "confidence_scores": confidence_score,
            "is_default_risk": prediction_label == 1,
            "prediction_label": prediction_label,  # ✅ Added explicit label
            "probability_class_0": probability_class_0,  # ✅ Added class 0 probability
            "probability_class_1": probability_class_1,  # ✅ Added class 1 probability
            "xai_artifact": xai_artifact,
            "artifact_path": temp_pred_path,
            "input_feature_hash": input_hash,
            "prediction_evidence_id":prediction_evidence_id,
            "model_evidence_id":model_info

        }

            output_json_path = os.path.join(temp_pred_path,"complete_prediction_output.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(prediction_output, f, indent=2, default=str)
        
            logging.info(f"Complete prediction output saved to: {output_json_path}")
            return prediction_output,artifact_paths
        except Exception as e:
            logging.error(f"Prediction pipeline failed: {e}")
            raise CGAgentException(e,sys)

 
if __name__=="__main__":
    agent=PredictionAgent(trace_id=str(uuid.uuid4().hex[:16]),client_transaction_id="PREDICTION_AGENT_V1.0")
    res= agent.run()
    print(res)