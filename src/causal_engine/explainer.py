import pandas as pd
import shap
import pickle
import os,sys
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import logging
from exception import CGAgentException
import numpy as np

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
    def generate_explanations(model_artifact_path:str,X_data_path:str)->Dict[str,Any]:
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
            df= pd.read_csv(X_data_path)
            df.drop(columns=["ID","default.payment.next.month"], errors="ignore")

            features = df.columns.tolist()

            explainer = shap.TreeExplainer(model.model)

            # Calculate SHAP values for the training dataset
            shap_values = explainer.shap_values(df)[1]
        
    
            #global feature importance for model audit/credit report
            global_imp=np.mean(np.abs(shap_values),axis=0)
            feature_importance_map=dict(zip(features,global_imp.tolist()))

            #local explanation (for single prediciton audit)
            local_shap_values=shap_values[0]
        
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
                else:
                    causal_graph_map[f] = "weak"


            # 4. Compile the machine-readable audit artifact
            audit_artifact = {
                "global_feature_importance": feature_importance_map,
                "local_top_drivers": top_drivers,
                "expected_value": explainer.expected_value[1], # Expected value for class 1
                "causal_graph": causal_graph_map,
                "risk_factors":risk_factors
            }

            logging.info("SHAP audit artifact generated successfully.")
            
            shap_path="src/model/explainer"
            os.makedirs(shap_path,exist_ok=True)
            out_dir = os.path.join(shap_path,"summary_plot.png")
            shap.summary_plot(shap_values,df, feature_names=features, plot_type="bar")
            plt.title("SHAP Summary Plot - Feature Importance")
            plt.savefig(out_dir, dpi=300)
            plt.show()    

            #calculate dependdence plot
            feature_index = 5
            shap.dependenceplot(feature_index,shap_values,df,feature_names=features)
            plt.title("SHAP dependence Plot - Feature Importance")
            out_dir= shap_path.join("dependence_plot.png")
            plt.savefig(out_dir, dpi=300)
            plt.show()

            #force plot
            instance_index=5
            shap.forceplot(explainer.expected_value,shap_values[instance_index],df.iloc[instance_index],feature_name=features)

            plt.title(f"SHAP Force Plot - Prediction Explanation for Instance {instance_index}")
            out_dir= shap_path.join("force_plot.png")
            plt.savefig(out_dir, dpi=300)
            plt.show()

            return audit_artifact

        except Exception as e:
            logging.error(f"XAI Engine failed: {e}")
            raise CGAgentException(e,sys)