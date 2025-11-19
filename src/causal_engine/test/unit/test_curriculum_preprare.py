import pytest
import pandas as pd
from pathlib import Path
from src.causal_engine.curricullum_preparer import read_data, curriculum_prep
from exception import CGAgentException


# FIXTURES
@pytest.fixture
def sample_data(tmp_path):
    """
    Creates temporary CSVs for X_train, y_train, X_val, y_val and cleanlab scores.
    Returns paths to these files.
    """

    # --- Training split ---
    X_train = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "feature1": [10, 20, 30, 40],
    })
    y_train = pd.DataFrame({"label": [1, 0, 1, 0]})

    # --- Validation split ---
    X_val = pd.DataFrame({
        "ID": [5, 6],
        "feature1": [50, 60],
    })
    y_val = pd.DataFrame({"label": [1, 0]})

    # --- Cleanlab scores ---
    cl_scores = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "quality_score": [0.95, 0.80, 0.92, 0.50]
    })

    # Create files
    X_train_path = tmp_path / "X_train.csv"
    y_train_path = tmp_path / "y_train.csv"
    X_val_path   = tmp_path / "X_val.csv"
    y_val_path   = tmp_path / "y_val.csv"
    cl_path      = tmp_path / "cl_scores.csv"

    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_val.to_csv(X_val_path, index=False)
    y_val.to_csv(y_val_path, index=False)
    cl_scores.to_csv(cl_path, index=False)

    return {
        "X_train": X_train_path,
        "y_train": y_train_path,
        "X_val": X_val_path,
        "y_val": y_val_path,
        "cl": cl_path
    }


# TESTS FOR read_data()


def test_read_data_success(tmp_path):
    """read_data should correctly read a CSV and return a DataFrame."""
    file_path = tmp_path / "dummy.csv"
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_csv(file_path, index=False)

    loaded_df = read_data(str(file_path))

    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.shape == (3, 1)


def test_read_data_raises_exception_on_bad_path():
    """read_data should raise CGAgentException when file does not exist."""
    with pytest.raises(CGAgentException):
        read_data("non_existent_file.csv")


# TESTS FOR curriculum_prep()

def test_curriculum_prep_success(sample_data):
    """Verify curriculum_prep loads all files and returns correct outputs."""
    
    outputs = curriculum_prep(
        str(sample_data["X_train"]),
        str(sample_data["y_train"]),
        str(sample_data["X_val"]),
        str(sample_data["y_val"]),
        str(sample_data["cl"])
    )

    (high_conf, low_conf, X_train, y_train, X_val, y_val, cl_scores) = outputs

    # Check segmentation logic
    assert len(high_conf) == 2  # quality >= 0.90 (IDs 1, 3)
    assert len(low_conf) == 2   # IDs 2, 4

    # Check merge happened
    assert "quality_score" in high_conf.columns
    assert "quality_score" in low_conf.columns

    # Check shapes of returned original splits
    assert X_train.shape == (4, 2)
    assert y_train.shape == (4, 1)
    assert X_val.shape == (2, 2)
    assert y_val.shape == (2, 1)


def test_curriculum_prep_missing_cleanlab_scores(tmp_path, sample_data):
    """Ensure missing cleanlab scores do not break the function."""
    
    # Modify cleanlab file to remove one ID
    cl_df = pd.read_csv(sample_data["cl"])
    cl_df = cl_df[cl_df["ID"] != 3]  # Remove ID 3
    cl_df.to_csv(sample_data["cl"], index=False)

    outputs = curriculum_prep(
        str(sample_data["X_train"]),
        str(sample_data["y_train"]),
        str(sample_data["X_val"]),
        str(sample_data["y_val"]),
        str(sample_data["cl"])
    )

    high_conf, low_conf, *_ = outputs

    # ID 3 should have NaN -> goes to low_conf by default (<0.90)
    assert 3 in low_conf["ID"].values


def test_curriculum_prep_throws_exception_on_bad_input(sample_data):
    """Expect CGAgentException if any file path is wrong."""
    with pytest.raises(CGAgentException):
        curriculum_prep(
            "invalid_path.csv",
            str(sample_data["y_train"]),
            str(sample_data["X_val"]),
            str(sample_data["y_val"]),
            str(sample_data["cl"])
        )
