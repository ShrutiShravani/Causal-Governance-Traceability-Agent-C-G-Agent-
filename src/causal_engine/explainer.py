import pandas as pd
import shap
import pickle
import os,sys
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from logger import logging
from exception import CGAgentException
import numpy as np
import mlflow
import json

class XAIEngine:
    @staticmethod
    def load_trained_model(model_path):
        try:
            with open(model_path,'rb')as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Error: 'trained_model.pkl' not found. Please ensure the file exists.")
        except Exception as e:
            raise CGAgentException(e,sys)
    
    @staticmethod
    def generate_explanations(model_artifact_path:str,X_data_path:str=None,X_data:pd.DataFrame=None,run_id:str= None, base_path:str ="src/model/audit_artifact") -> Dict[str, Any]:
        """
        Generates SHAP explanations - works with or without MLflow
        """
        try:
            logging.info("Starting SHAP explanation generation...")
            
            # Validate input
            if X_data_path is None and X_data is None:
                raise ValueError("Either X_data_path or X_data must be provided")
            if X_data_path is not None and X_data is not None:
                logging.warning("Both X_data_path and X_data provided. Using X_data (DataFrame) and ignoring X_data_path.")
            
            # Handle MLflow only if run_id is provided
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    logging.info(f"Using MLflow run: {run_id}")
                    return XAIEngine.generate_explanations_internal(model_artifact_path=model_artifact_path,X_data_path=X_data_path,X_data=X_data,base_path=base_path,should_log=True)
            else:
                logging.info("Generating explanations without MLflow logging")
                return XAIEngine.generate_explanations_internal(model_artifact_path=model_artifact_path,X_data_path=X_data_path,X_data=X_data,base_path=base_path,should_log=False)
                
        except Exception as e:
            logging.error(f"XAI Engine failed: {e}")
            raise CGAgentException(e, sys)
    
    
    @staticmethod
    def generate_explanations_internal(model_artifact_path:str,X_data_path:str,X_data:pd.DataFrame=None,base_path:str="src/model/audit_artifact",should_log:bool =False)->Dict[str,Any]:
        """
        Generates SHAP explanations for auditability and extracts quantitative metrics.
        
        Args:
            model_artifact: The fitted CausalModel object from trainer.py.
            X_data: The validation set (for global metrics) or a single client row (for local decision).
        
        Returns:
            A dictionary containing the quantitative audit artifacts.
        """
        try:
            logging.info("Starting SHAP explanation generation...")

            model= XAIEngine.load_trained_model(model_artifact_path)
            # Load data from either DataFrame or file path
            if X_data is not None:
                df = X_data.copy()
                logging.info(f"Using provided DataFrame with shape: {df.shape}")
            else:
                df = pd.read_csv(X_data_path)
                logging.info(f"Loaded data from {X_data_path} with shape: {df.shape}")
            
            df=df.drop(columns=["ID"], errors="ignore")
            
            features = df.columns.tolist()
            if len(features) == 0:
                raise ValueError("No features found after dropping columns. Check your data.")

            explainer = shap.TreeExplainer(model.model)

            # Calculate SHAP values for the training dataset
            
            shap_values = explainer.shap_values(df)

            #handle shap values format
            if isinstance(shap_values,list):
                shap_values_class1= shap_values[1]
            else:
                shap_values_class1=shap_values
            shap_values_class1 = np.atleast_2d(shap_values_class1)
        
    
            #global feature importance for model audit/credit report
            global_imp=np.mean(np.abs(shap_values),axis=0)
            feature_importance_map=dict(zip(features,global_imp.tolist()))

            #local explanation (for single prediciton audit)
            local_shap_values=shap_values_class1[0]
        
            local_drivers={
                features[i]:float(local_shap_values[i]) for i in range(len(features))
            }

            #sort and take top 5 most influential factors for logging
            top_drivers= sorted(local_drivers.items(),key=lambda item:abs(item[1]),reverse=True)[:5]
            
            risk_factors=[feature for feature,_ in top_drivers]

            #causal graph
            causal_graph_map={}
            for f ,value in local_drivers.items():
                if value>0.15:
                    causal_graph_map[f]="strong_positive"
                elif value>0.07:
                    causal_graph_map[f]="moderate_positive"
                elif value < -0.10:
                    causal_graph_map[f] = "strong_negative"
                elif value < -0.05:
                    causal_graph_map[f] = "moderate_negative"
                else:
                    causal_graph_map[f] = "weak"

            # Handle expected value
            if hasattr(explainer.expected_value, "__len__") and len(explainer.expected_value) > 1:
                expected_value = explainer.expected_value[1]
            else:
                expected_value = float(explainer.expected_value)
            
            os.makedirs(os.path.dirname(base_path),exist_ok=True)
            artifact_path=f"{base_path}_detailed.json"

            # 4. Compile the machine-readable audit artifact
            audit_artifact = {
                "global_feature_importance": feature_importance_map,
                "local_top_drivers": top_drivers,
                "expected_value": expected_value,
                "causal_graph": causal_graph_map,
                "risk_factors":risk_factors,
                "artifact_path": artifact_path
            }

            logging.info("SHAP audit artifact generated successfully.")
            logging.info(f"Top drivers: {top_drivers}")

            
            with open(artifact_path, 'w') as f:
                json.dump(audit_artifact, f, indent=2)
            logging.info(f"Detailed artifact saved to: {artifact_path}")
            if should_log:
                try:
                    mlflow.log_artifact(artifact_path)
                    logging.info("XAI artifacts logged to MLflow experiment: CausalModel_Training")
                except Exception as e:
                    logging.warning(f"Failed to log to MLflow: {e}")

            return audit_artifact

        except Exception as e:
            logging.error(f"XAI Engine failed: {e}")
            raise CGAgentException(e,sys)

    @staticmethod
    def generate_visualization(audit_artifact: Dict[str, Any],run_id:str = None,base_path:str = "src/model/explainer_artifact"):
        """
        Generate visualization - works with or without MLflow
        """
        try:
            # Handle MLflow only if run_id is provided
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    logging.info(f"Using MLflow run: {run_id}")
                    XAIEngine.generate_visualization_internal(audit_artifact, base_path,should_log=True)
            else:
                logging.info("Generating visualization without MLflow logging")
                XAIEngine.generate_visualization_internal(audit_artifact, base_path, should_log=False)
                
        except Exception as e:
            logging.error(f"Visualization generation failed: {e}")
            raise CGAgentException(e, sys)

    @staticmethod  # FIX 3: Added missing staticmethod decorator
    def generate_visualization_internal(audit_artifact: Dict[str, Any],base_path:str,should_log:bool=False):
        """
        Generate visualization from SHAP audit artifact
        """
        try:
            
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
            # FIX 4: Corrected typo "global_fetaure_importance" to "global_feature_importance"
            global_imp = audit_artifact["global_feature_importance"]

            features = list(global_imp.keys())
            importance = list(global_imp.values())

            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(features))
            plt.barh(y_pos, importance)
            plt.yticks(y_pos, features)
            plt.xlabel("Mean Absolute SHAP Value")
            plt.title('Global Feature Importance')
            plt.tight_layout()

            # FIX 5: Save the plot BEFORE showing it
            plot_path = f"{base_path}_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info(f"SHAP visualization saved to {plot_path}")
            
            plt.show()
            plt.close()

            # Log to MLflow if available
            if should_log:
                try:
                    mlflow.log_artifact(plot_path)
                    logging.info("XAI visualization logged to MLflow experiment: CausalModel_Training")
                except Exception:
                    logging.warning("MLflow not available for logging")
 
        except KeyError as e:
            logging.error(f"Missing key in audit artifact: {e}")
            raise CGAgentException(f"Audit artifact missing expected key: {e}", sys)
        except Exception as e:
            logging.error(f"Visualization generation failed: {e}")
            raise CGAgentException(e, sys)
