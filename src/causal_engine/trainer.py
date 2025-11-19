import os, sys
import json
import pickle
import logging
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from exception import CGAgentException
from src.causal_engine.causal_model import CausalModel, CurriculumScheduler
from src.causal_engine.curricullum_preparer import curriculum_prep
import mlflow

class TrainingEngine:
    @staticmethod
    def train_causal_model(
        X_train_path: str,
        y_train_path: str,
        X_val_path: str,
        y_val_path: str,
        confidence_report_path: str
    ) -> Dict[str, Any]:

        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
            mlflow.set_experiment("CausalModel_Training")
            logging.info("TrainingEngine: Starting curriculum-based causal training...")

            high_df, low_df, X_train, y_train, X_val, y_val, cl_scores = (
                curriculum_prep(
                    X_train_path,
                    y_train_path,
                    X_val_path,
                    y_val_path,
                    confidence_report_path
                )
            )

            scheduler = CurriculumScheduler(high_df, low_df)
            batches = scheduler.get_batches()

            features = X_train.drop(columns=["ID", "default.payment.next.month"],
                                    errors="ignore").columns.tolist()

            loaded_model = CausalModel(features,model=None)
            clf = loaded_model.model
            mlflow.log_param("learning_rate", clf.learning_rate)
            mlflow.log_param("n_estimators", clf.n_estimators)
            mlflow.log_param("max_depth", clf.max_depth)
            mlflow.log_param("subsample", clf.subsample)
            
            metrics_history = []

            # ------------------------------------------
            # MLflow Run Start
            # ------------------------------------------
            with mlflow.start_run(run_name="causal_training"):

                # Log parameters
                mlflow.log_param("num_features", len(features))
                mlflow.log_param("curriculum_batches", len(batches))

            # Training loop
            for epoch, batch in enumerate(batches, start=1):
                X_ep = batch[features]
                y_ep = batch["y"]

                loaded_model.fit(X_ep, y_ep)

                pred = loaded_model.predict(X_val)
                prob = loaded_model.predict_proba(X_val)

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

            # Save model
            model_path = "src/model/trained_gb_model.pkl"
            os.makedirs("src/model", exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(loaded_model, f)

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
                "curriculum_scores": cl_scores
            }

        except Exception as e:
            raise CGAgentException(e, sys)
