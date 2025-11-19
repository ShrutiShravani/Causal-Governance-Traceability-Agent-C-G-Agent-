import pytest
import pandas as pd
import hashlib
from src.data_pipeline.data_hashing import dataframe_to_stable_bytes,generate_sha256_hash

@pytest.fixture
def sample_df():
    """fixture: Create a deterministic dataframe for testing hashing logic """
    return pd.DataFrame({
        "A":[1,2,3],
        "B":["x","y","z"]
    })

def test_dataframe_to_stable_bytes_consistency(sample_df):
    """ensure dataframe_to_stable_bytes() returns stable byte stream for same dataframe"""

    bytes1 = dataframe_to_stable_bytes(sample_df)
    bytes2 = dataframe_to_stable_bytes(sample_df.copy())

    # Should produce identical byte stream if data is same
    assert bytes1 == bytes2
    assert isinstance(bytes1, bytes)
    assert b"A,B" in bytes1  # Header should be present in encoded CSV


def test_dataframe_to_stable_bytes_variation(sample_df):
    """Ensure that small dataframe differences produce different byte streams."""
    df_modified = sample_df.copy()
    df_modified.loc[0, "A"] = 999  # Modify one cell

    bytes_orig = dataframe_to_stable_bytes(sample_df)
    bytes_mod = dataframe_to_stable_bytes(df_modified)

    assert bytes_orig != bytes_mod  # Changing data should change byte output


def test_generate_sha256_hash(sample_df):
    """Verify SHA256 hash generation from dataframe bytes."""
    data_bytes = dataframe_to_stable_bytes(sample_df)
    hash_value = generate_sha256_hash(data_bytes)

    expected_hash = hashlib.sha256(data_bytes).hexdigest()
    assert hash_value == expected_hash
    assert len(hash_value) == 64  # Standard SHA-256 hex length


def test_generate_sha256_hash_different_inputs():
    """Different inputs must yield different SHA256 values."""
    data1 = b"foo"
    data2 = b"bar"
    hash1 = generate_sha256_hash(data1)
    hash2 = generate_sha256_hash(data2)

    assert hash1 != hash2
    assert isinstance(hash1, str)
    assert isinstance(hash2, str)


def test_integration_hash_consistency(sample_df):
    """Integration: Same dataframe -> same hash across calls."""
    bytes_1 = dataframe_to_stable_bytes(sample_df)
    bytes_2 = dataframe_to_stable_bytes(sample_df.copy())

    hash_1 = generate_sha256_hash(bytes_1)
    hash_2 = generate_sha256_hash(bytes_2)

    assert hash_1 == hash_2