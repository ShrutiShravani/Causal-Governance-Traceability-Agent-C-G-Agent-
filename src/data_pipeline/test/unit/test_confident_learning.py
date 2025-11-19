import pytest
import pandas as pd
import numpy as np
from src.data_pipeline.confident_learning import run_confident_learning_audit
from exception import CGAgentException


@pytest.fixture
def dummy_df():
    """Fake data for unit testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "ID": range(10),
        "LIMIT_BAL": np.random.randint(10000, 50000, 10),
        "AGE": np.random.randint(20, 60, 10),
        "PAY_1": np.random.randint(-2, 2, 10),
        "PAY_2": np.random.randint(-2, 2, 10),
        "default.payment.next.month": np.random.choice([0, 1], 10)
    })
    features = {
        "train_features": {"include": ["LIMIT_BAL", "AGE", "PAY_1", "PAY_2"]},
        "target": "default.payment.next.month"
    }
    return df, features


def test_confident_learning_output_shape(dummy_df):
    """Ensure output dataframe and threshold are returned correctly."""
    df, features = dummy_df
    mislabeled_df, threshold = run_confident_learning_audit(df, features, df["ID"])
    
    # Validate output types
    assert isinstance(mislabeled_df, pd.DataFrame)
    assert isinstance(threshold, float)
    
    # Validate output columns
    expected_cols = {"ID", "original_label", "quality_score"}
    assert expected_cols.issubset(mislabeled_df.columns)
    
    # Validate confidence threshold is as defined
    assert 0 <= threshold <= 1


def test_confident_learning_handles_single_class(dummy_df):
    """Check proper exception is raised when only one class is present."""
    df, features = dummy_df
    df["default.payment.next.month"] = 1  # Make target column single-class
    
    with pytest.raises(CGAgentException):
        run_confident_learning_audit(df, features, df["ID"])


def test_confident_learning_reproducibility(dummy_df):
    """Ensure results are deterministic with same seed and small dataset."""
    df, features = dummy_df
    mislabeled_df_1, _ = run_confident_learning_audit(df, features, df["ID"])
    mislabeled_df_2, _ = run_confident_learning_audit(df, features, df["ID"])
    
    # Ensure both runs produce identical output IDs (order may vary)
    assert set(mislabeled_df_1["ID"]) == set(mislabeled_df_2["ID"])
