import os, sys
import json
import pickle
from logger import logging
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from exception import CGAgentException
from src.causal_engine.causal_model import CausalModel, CurriculumScheduler
from src.causal_engine.curricullum_preparer import curriculum_prep
import mlflow

class TrainingEngine:
    @staticmethod
    def read_data(path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(path)
            return data
        except Exception as e:
            raise CGAgentException(e, sys)

    @staticmethod
    def train_causal_model(
        X_train_path: str,
        y_train_path: str,
        X_val_path: str,
        y_val_path: str,
        confidence_report_path: str
    ) -> Dict[str, Any]:

        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
            mlflow.set_experiment("CausalModel_Training")
            logging.info("TrainingEngine: Starting curriculum-based causal training...")

            high_df, low_df= (
                curriculum_prep(
                    X_train_path,
                    y_train_path,
                    confidence_report_path,
                    quality_threshold=0.9
                )
            )
 
            scheduler = CurriculumScheduler(high_df,low_df)
            batches = scheduler.get_batches()
            
            non_feature_cols=["ID","default.payment.next.month", "y", "is_issue", "quality_score","ID"]
            features = [col for col in high_df.columns if col not in non_feature_cols]

            loaded_model = CausalModel(features, model=None)
            clf = loaded_model.model

            X_val= TrainingEngine.read_data(X_val_path)
            y_val= TrainingEngine.read_data(y_val_path)

            # Prepare validation features - remove ID and target column if present
            val_non_feature_columns = ["ID", "default.payment.next.month"]
            val_features = [col for col in X_val.columns if col not in val_non_feature_columns]
            X_val_clean = X_val[val_features]
            
            metrics_history = []

            # ------------------------------------------
            # MLflow Run Start - MOVE EVERYTHING INSIDE
            # ------------------------------------------
            with mlflow.start_run(run_name="causal_training") as run:
                run_id = run.info.run_id
                print("MLflow RUN ID =", run_id)
                # Log parameters
                mlflow.log_param("learning_rate", clf.learning_rate)
                mlflow.log_param("n_estimators", clf.n_estimators)
                mlflow.log_param("max_depth", clf.max_depth)
                mlflow.log_param("subsample", clf.subsample)
                mlflow.log_param("num_features", len(features))
                mlflow.log_param("curriculum_batches", len(batches))
                mlflow.log_param("quality_threshold", 0.9)
                mlflow.log_param("high_conf_samples", len(high_df))
                mlflow.log_param("low_conf_samples", len(low_df))

                # Training loop
                for epoch, batch in enumerate(batches, start=1):
                    X_ep = batch[features]
                    print(X_ep)
                    y_ep = batch['y']

                    loaded_model.fit(X_ep, y_ep)

                    pred = loaded_model.predict(X_val_clean)
                    prob = loaded_model.predict_proba(X_val_clean)

                    acc = accuracy_score(y_val, pred)
                    f1 = f1_score(y_val, pred)
                    auc = roc_auc_score(y_val, prob)

                    metrics_history.append({
                        "epoch": epoch,
                        "acc": acc,
                        "f1": f1,
                        "roc_auc": auc
                    })

                    # MLflow metric logging
                    mlflow.log_metric("val_acc", acc, step=epoch)
                    mlflow.log_metric("val_f1", f1, step=epoch)
                    mlflow.log_metric("val_roc_auc", auc, step=epoch)

                    logging.info(f"Epoch {epoch}: ACC={acc:.4f}, F1={f1:.4f}, ROC_AUC={auc:.4f}")

                # Best epoch
                best_epoch = max(metrics_history, key=lambda x: x["roc_auc"])
                mlflow.log_metric("best_roc_auc", best_epoch["roc_auc"])


                # Save model
                model_path = "src/model/trained_gb_model.pkl"
                os.makedirs("src/model", exist_ok=True)
                with open(model_path, "wb") as f:
                    pickle.dump(loaded_model, f)

                model_name = "CreditRiskModel"
                model_uri = f"runs:/{run_id}/model"

                # Register model in MLflow Model Registry
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name
                )

                # Save metrics
                metrics_path = "src/model/metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics_history, f, indent=4)
           
                # Log artifacts
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(metrics_path)

            return {
                "model_path": model_path,
                "features": features,
                "best_epoch": best_epoch,
                "metrics_history": metrics_history,
                "high_conf_samples": len(high_df),
                "low_conf_samples": len(low_df),
                "quality_threshold": 0.9,
                "run_id":run_id,
                "mlflow_model_name": model_name,
                "mlflow_model_version": registered_model.version,  
            }

        except Exception as e:
            raise CGAgentException(e, sys)