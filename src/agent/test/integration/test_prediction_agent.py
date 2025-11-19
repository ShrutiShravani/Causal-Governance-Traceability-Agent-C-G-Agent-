import os
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.agent.prediction_agent import PredictionAgent


# FIXTURES
@pytest.fixture
def mock_trace_id():
    return "trace-999"


@pytest.fixture
def mock_transaction_id():
    return "txn-123"


@pytest.fixture
def sample_test_data(tmp_path):
    """
    Create synthetic cleaned input CSV so prediction agent can read it.
    """
    df = pd.DataFrame({
        "ID": [10, 11],
        "feat1": [100, 200],
        "feat2": [1.2, 3.4],
        "default.payment.next.month": [0, 1]
    })

    file_path = tmp_path / "X_test_credit_clean.csv"
    df.to_csv(file_path, index=False)

    return str(file_path)


@pytest.fixture
def mock_model(tmp_path):
    """
    Create a dummy model pickle file.
    """
    model_path = tmp_path / "gb_mock_model.pkl"

    mock_model = MagicMock()
    mock_model.predict.return_value = [1, 0]
    mock_model.predict_proba.return_value = [[0.2, 0.8], [0.7, 0.3]]

    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(mock_model, f)

    return str(model_path)


# MAIN INTEGRATION TEST

def test_prediction_agent_full_pipeline(
    mock_trace_id,
    mock_transaction_id,
    sample_test_data,
    mock_model,
    monkeypatch
):
    """
    Full integration test covering:
    - model load
    - data pipeline
    - prediction
    - XAI explanation call
    - event logging
    - artifact registration
    """


    # 1. MOCK: Data pipeline functions

    monkeypatch.setattr(
        "src.agent.prediction_agent.categorical_consolidation",
        lambda df, f: df
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.clip_outliers",
        lambda df: df
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.handle_missing_values",
        lambda df: df
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.impute_missing_values",
        lambda df: df
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.check_data_imbalance",
        lambda df: False
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.handle_imbalance_data",
        lambda df, target_col, method: df
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.prepare_data",
        lambda df, f: (df, df["ID"])
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.feature_engineering",
        lambda df: df
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.create_utlization_rate",
        lambda df: df
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.encode_categorical_features",
        lambda df, f: df
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.scale_numeric_features",
        lambda df, f: df
    )

   
    # 2. MOCK: SHA256 hashing

    monkeypatch.setattr(
        "src.agent.prediction_agent.generate_sha256_hash",
        lambda b: "mocked-sha256"
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.dataframe_to_stable_bytes",
        lambda df: b"mock-bytes"
    )


    # 3. MOCK: XAI Engine
   
    fake_xai_output = {
        "local_top_drivers": {"feat1": 0.55},
        "expected_value": 0.12,
        "causal_graph": {"node": ["feat1", "feat2"]},
        "risk_factors": ["high_utilization"]
    }

    monkeypatch.setattr(
        "src.agent.prediction_agent.XAIEngine.generate_explanations",
        lambda model_artifact_path, X_data_path: fake_xai_output
    )

    # 4. MOCK: log_event + register_artifact
  
    mock_log_event = MagicMock()
    mock_register = MagicMock()

    monkeypatch.setattr(
        "src.agent.prediction_agent.log_event",
        mock_log_event
    )
    monkeypatch.setattr(
        "src.agent.prediction_agent.register_artifact",
        mock_register
    )


    # 5. MOCK trace context extraction + tracer
   
    monkeypatch.setattr(
        "src.agent.prediction_agent.extract_trace_context",
        lambda payload: None
    )

    fake_tracer = MagicMock()
    fake_span = MagicMock()

    fake_tracer.start_as_current_span.return_value = fake_span

    monkeypatch.setattr(
        "src.agent.prediction_agent.get_tracer",
        lambda: fake_tracer
    )


    # 6. Monkeypatch model + file paths

    monkeypatch.setattr(PredictionAgent, "model_path", mock_model)
    monkeypatch.setattr(PredictionAgent, "X_input_path", sample_test_data)

    # 7. Prepare agent + run
   
    agent = PredictionAgent(
        trace_id=mock_trace_id,
        client_transaction_id=mock_transaction_id
    )

    output = agent.run(payload={})

    # --------------------------------------------------------
    # ASSERTIONS
    # --------------------------------------------------------

    assert output["status"] == "success"
    assert output["agent"] == "Prediction_Agent"

    pred_output = output["prediction_output"]

    # Prediction structure
    assert "predicitons" in pred_output
    assert "probabilities" in pred_output
    assert "confidence" in pred_output
    assert "xai_artifact" in pred_output

    # Values from mock model
    assert pred_output["predicitons"] == [1, 0]
    assert pred_output["probabilities"] == [[0.2, 0.8], [0.7, 0.3]]
    assert isinstance(pred_output["confidence"], float)

    # XAI output
    assert pred_output["xai_artifact"] == fake_xai_output

    # -------------------------
    # Verify event logging
    # -------------------------
    assert mock_log_event.call_count >= 2  # prediction + xai logging

    mock_log_event.assert_any_call(
        trace_id=mock_trace_id,
        client_transaction_id=mock_transaction_id,
        agent_name="Prediction_Agent",
        phase="Phase3_PredictionPipeline",
        event_type="PREDICTION_EXECUTED",
        summary="Model prediction generated successfully.",
        extra_payload=pytest.ANY
    )

    # -------------------------
    # Verify artifact registration
    # -------------------------
    mock_register.assert_called_once()
    args, kwargs = mock_register.call_args
    assert args[0] == "evidence_pointer"
    assert "xai_artifact_path" in args[1]

    print("\n===== PREDICTION AGENT INTEGRATION TEST PASSED =====\n")
