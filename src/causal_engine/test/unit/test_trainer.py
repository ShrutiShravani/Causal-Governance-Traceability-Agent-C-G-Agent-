import pytest
import os
import pandas as pd

from src.causal_engine.trainer import TrainingEngine

def test_training_engine_returns_expected_keys(tmp_path, monkeypatch):
    # ---- Create Tiny Synthetic Data ----
    df_X = pd.DataFrame({
        "ID": [1, 2],
        "feat1": [10, 20],
        "feat2": [1.1, 2.2],
        "default.payment.next.month": [0, 1]
    })
    df_y = pd.DataFrame({"y": [0, 1]})

    df_conf = pd.DataFrame({
        "ID": [1, 2],
        "confidence": [0.9, 0.2]
    })

    # Save to temp folder
    X_train = tmp_path / "X_train.csv"
    y_train = tmp_path / "y_train.csv"
    X_val = tmp_path / "X_val.csv"
    y_val = tmp_path / "y_val.csv"
    conf = tmp_path / "conf.csv"

    df_X.to_csv(X_train, index=False)
    df_y.to_csv(y_train, index=False)
    df_X.to_csv(X_val, index=False)
    df_y.to_csv(y_val, index=False)
    df_conf.to_csv(conf, index=False)

    # ---- Run Training ----
    result = TrainingEngine.train_causal_model(
        str(X_train), str(y_train),
        str(X_val), str(y_val),
        str(conf)
    )

    # ---- Validate Result ----
    assert "model_path" in result
    assert "metrics_history" in result
    assert "best_epoch" in result
    assert "features" in result
    assert "curriculum_scores" in result

    # ---- Validate Model Exists ----
    assert os.path.exists(result["model_path"])
