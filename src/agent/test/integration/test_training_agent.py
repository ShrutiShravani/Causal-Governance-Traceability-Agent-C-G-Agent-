import os
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock,patch
from src.agent.training_agent import TrainingAgent

#fixture
@pytest.fixture
def mock_trace_id():
    return "test-trace-123"

@pytest.fixture
def mock_transaction_id():
    return "txn-987"


@pytest.fixture
def sample_training_data(tmp_path):
    """
    Creates minimal synthetic CSV datasets:
    - X_train, y_train
    - X_val, y_val
    - confident learning report
    """

    # Tiny synthetic datasets
    X_train = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "feat1": [10, 20, 30, 40],
        "feat2": [1.1, 2.2, 3.3, 4.4],
        "default.payment.next.month": [0, 1, 0, 1]
    })

    y_train = pd.DataFrame({"y": [0, 1, 0, 1]})

    X_val = pd.DataFrame({
        "ID": [5, 6],
        "feat1": [15, 25],
        "feat2": [1.5, 2.5],
        "default.payment.next.month": [0, 1]
    })

    y_val = pd.DataFrame({"y": [0, 1]})

    conf_report = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "confidence": [0.9, 0.1, 0.8, 0.2]
    })

    # Create folder structure
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    X_train_path = data_dir / "X_train.csv"
    y_train_path = data_dir / "y_train.csv"
    X_val_path = data_dir / "X_val.csv"
    y_val_path = data_dir / "y_val.csv"
    conf_path = data_dir / "conf_report.csv"

    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_val.to_csv(X_val_path, index=False)
    y_val.to_csv(y_val_path, index=False)
    conf_report.to_csv(conf_path, index=False)

    return {
        "X_train": str(X_train_path),
        "y_train": str(y_train_path),
        "X_val": str(X_val_path),
        "y_val": str(y_val_path),
        "conf": str(conf_path),
        "tmp_dir": tmp_path
    }

#full intergration test

def test_training_agent_full_pipeline(mock_trace_id,mock_transaction_id,sample_training_data,monkeypatch):
    #mock artifcat registration +event logging

    mocked_event_logger=MagicMock()

    mocked_artifact_register=MagicMock()

    monkeypatch.setattr(
        "src.agent.training.EventLogger",
        lambda trace_id:mocked_event_logger
    )
    
    monkeypatch.setattr(
        "src.agent.training_agent.ArtifactRegistry",
        mocked_artifact_register
    )

    #moneketpatch training agent paths
    monkeypatch.setattr(TrainingAgent,"X_train_path",sample_training_data["X_train"])
    monkeypatch.setattr(TrainingAgent, "y_train_path", sample_training_data["y_train"])
    monkeypatch.setattr(TrainingAgent, "X_val_path", sample_training_data["X_val"])
    monkeypatch.setattr(TrainingAgent, "y_val_path", sample_training_data["y_val"])
    monkeypatch.setattr(TrainingAgent, "confidence_learning_report", sample_training_data["conf"])
   #redirect modle save path
    temp_model_path= sample_training_data['tmp_dir']/"trained_gb_model.pkl"

    monkeypatch.setattr(TrainingAgent,"model_path",str(temp_model_path))

    #run agent
    agent=TrainingAgent(
       trace_id=mock_trace_id,
        client_transaction_id=mock_transaction_id
    )

    output = agent.run()

    #verify output structure
    assert output["status"]=="success"

    assert "result" in output
    training_artifacts=output["result"]["training_artifacts"]
    xai_artifact= output["result"]["xai_artifact"]

    #verify model file created

    assert os.path.exists(training_artifacts["model_path"])
    assert training_artifacts["model_path"].endswith(".pkl")

    #verify training metrics
    metrics= training_artifacts["metrics_history"]

    assert len(metrics)>0
    assert "epoch" in metrics[0]

    #verify best epoch metrics
    assert "best_epoch" in training_artifacts
    assert "roc_auc" in training_artifacts["best_epoch"]

    # VERIFY XAI ARTIFACT GENERATED
    assert xai_artifact is not None
    assert "shap_summary_plot" in xai_artifact
    assert os.path.exists(xai_artifact["shap_summary_plot"])


    # VERIFY MONKEYPATCH DATA INJECTION WORKED
    
    assert TrainingAgent.X_train_path == sample_training_data["X_train"]
    assert TrainingAgent.X_val_path == sample_training_data["X_val"]
    assert TrainingAgent.model_path == str(temp_model_path)

    #vent logger wa scalld
    assert mocked_event_logger.log_info.call_count > 0
    assert mocked_event_logger.log_success.call_count == 1
   
   # VERIFY ARTIFACT REGISTRATION OCCURRED
   
    assert mocked_artifact_register.register_artifact.call_count >= 1

    print("\n===== TRAINING AGENT INTEGRATION TEST PASSED SUCCESSFULLY =====\n")


