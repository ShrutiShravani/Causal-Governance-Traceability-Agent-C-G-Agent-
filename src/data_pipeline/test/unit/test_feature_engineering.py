import pytest
import pandas as pd
import numpy as np
from exception import CGAgentException
from src.data_pipeline.feature_engineering import (
    feature_engineering,
    create_utlization_rate,
    encode_categorical_features,
    scale_numeric_features,
    get_features,
    train_val_split,
    run_feature_engineering
)


@pytest.fixture
def dummy_df():
    """Mock dataset for testing feature engineering."""
    np.random.seed(42)
    df = pd.DataFrame({
        "LIMIT_BAL": np.random.randint(10000, 50000, 10),
        "AGE": np.random.randint(20, 60, 10),
        "BILL_AMT1": np.random.randint(1000, 50000, 10),
        "SEX": np.random.choice([1, 2], 10),
        "EDUCATION": np.random.choice([1, 2, 3, 4], 10),
        "MARRIAGE": np.random.choice([1, 2, 3], 10),
        "default.payment.next.month": np.random.choice([0, 1], 10)
    })

    # Add PAY columns (note: your code includes a trailing space!)
    for i in [0, 1, 2, 3, 4, 5, 6]:
        df[f"PAY_{i} "] = np.random.randint(-1, 3, 10)

    return df


@pytest.fixture
def dummy_features():
    """Mock features configuration."""
    return {
        "categorical_cols": ["SEX", "EDUCATION", "MARRIAGE"],
        "numerical_cols": ["LIMIT_BAL", "AGE", "BILL_AMT1"],
        "train_features": {"include_cols": ["LIMIT_BAL", "AGE", "BILL_AMT1"]},
        "target": "default.payment.next.month"
    }


# ------------------ UNIT TESTS ------------------ #

def test_feature_engineering_creates_total_months_late(dummy_df):
    result = feature_engineering(dummy_df.copy())
    assert "total_months_late" in result.columns
    assert result["total_months_late"].dtype in [np.int64, np.int32]


def test_create_utilization_rate(dummy_df):
    result = create_utlization_rate(dummy_df.copy())
    assert "utilization_rate" in result.columns
    assert all(result["utilization_rate"].between(0, 5))  # sanity range


def test_encode_categorical_features(dummy_df, dummy_features):
    result = encode_categorical_features(dummy_df.copy(), dummy_features)
    # Expect one-hot encoded columns to appear
    assert any("SEX_" in c for c in result.columns)
    assert any("EDUCATION_" in c for c in result.columns)
    assert any("MARRIAGE_" in c for c in result.columns)


def test_scale_numeric_features(dummy_df, dummy_features):
    # Add engineered columns first
    df = feature_engineering(dummy_df.copy())
    df = create_utlization_rate(df)
    result = scale_numeric_features(df.copy(), dummy_features)
    # Ensure mean ≈ 0, std ≈ 1 for existing numeric cols
    cols_to_check = ["LIMIT_BAL", "AGE", "BILL_AMT1", "utilization_rate", "total_months_late"]
    for col in cols_to_check:
        if col in result.columns:
            mean, std = result[col].mean(), result[col].std()
            assert abs(mean) < 1e-6 or np.isnan(mean)
            assert 0.5 <= std <= 1.5 or np.isnan(std)


def test_get_features(dummy_df, dummy_features):
    X, y = get_features(dummy_df.copy(), dummy_features)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == len(y)


def test_train_val_split(dummy_df, dummy_features):
    X, y = get_features(dummy_df.copy(), dummy_features)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_split(X, y)
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(X)
    assert len(X_val) > 0 and len(X_test) > 0


# ------------------ INTEGRATION TEST ------------------ #

def test_run_feature_engineering_full(dummy_df, dummy_features):
    """Full run of feature engineering with fake data."""
    X_train, y_train, X_val, y_val, X_test, y_test, processed_df = run_feature_engineering(
        dummy_df.copy(), dummy_features
    )

    # Validate outputs
    assert isinstance(processed_df, pd.DataFrame)
    assert "utilization_rate" in processed_df.columns
    assert "total_months_late" in processed_df.columns
    assert X_train.shape[1] > 0
    assert y_train.notnull().all()


# ------------------ ERROR HANDLING ------------------ #

def test_feature_engineering_raises_exception_on_invalid_df():
    with pytest.raises(CGAgentException):
        feature_engineering(None)
