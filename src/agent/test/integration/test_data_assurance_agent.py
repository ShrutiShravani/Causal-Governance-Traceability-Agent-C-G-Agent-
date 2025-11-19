import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.agent.data_assurance_agent import DataAssuranceAgent


@pytest.fixture
def fake_df():
    """A small dataset to simulate real loaded data."""
    return pd.DataFrame({
        "LIMIT_BAL": [20000, 30000, 50000],
        "SEX": [1, 2, 2],
        "EDUCATION": [2, 2, 3],
        "MARRIAGE": [1, 2, 2],
        "AGE": [24, 32, 45],
        "default.payment.next.month": [0, 1, 0]
    })


@pytest.fixture
def fake_paths(tmp_path):
    """Simulated output folder structure for the agent."""
    p = {
        "interim_path": str(tmp_path / "interim.csv"),
        "confident_learning_path": str(tmp_path / "cl_report.csv"),
        "mislabelled_report_path": str(tmp_path / "mislabelled.json"),
        "target_path": str(tmp_path / "final.csv"),
        "X_train_path": str(tmp_path / "X_train.csv"),
        "y_train_path": str(tmp_path / "y_train.csv"),
        "X_test_path": str(tmp_path / "X_test.csv"),
        "y_test_path": str(tmp_path / "y_test.csv"),
        "X_val_path": str(tmp_path / "X_val.csv"),
        "y_val_path": str(tmp_path / "y_val.csv"),
        "X_train_val_hash": str(tmp_path / "train_val_hash.json"),
        "X_test_hash": str(tmp_path / "test_hash.json"),
        "pac_metadata": str(tmp_path / "pac_metadata.yaml"),
    }
    return p


@pytest.fixture
def mock_features():
    return ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]


def test_data_assurance_agent_full_integration(fake_df, fake_paths, mock_features,tmp_path):

    # 1. MOCK ALL DEPENDENCIES
   
    with patch("src.agent.data_assurance_agent.Data_Reader.load_data") as mock_load, \
         patch("src.agent.data_assurance_agent.categorical_consolidation", return_value=fake_df) as m1, \
         patch("src.agent.data_assurance_agent.clip_outliers", return_value=fake_df) as m2, \
         patch("src.agent.data_assurance_agent.handle_missing_values", return_value=fake_df) as m3, \
         patch("src.agent.data_assurance_agent.impute_missing_values", return_value=fake_df) as m4, \
         patch("src.agent.data_assurance_agent.check_data_imbalance", return_value=False) as m5, \
         patch("src.agent.data_assurance_agent.prepare_data", return_value=(fake_df, ["id1","id2","id3"])) as m6, \
         patch("src.agent.data_assurance_agent.run_confident_learning_audit") as m7, \
         patch("src.agent.data_assurance_agent.run_feature_engineering") as m8, \
         patch("src.agent.data_assurance_agent.generate_sha256_hash", return_value="FAKE_HASH") as m9, \
         patch("src.agent.data_assurance_agent.dataframe_to_stable_bytes", return_value=b"123") as m10, \
         patch("src.agent.data_assurance_agent.log_event") as mock_log_event, \
         patch("src.agent.data_assurance_agent.register_artifact", return_value="ART-ID-123") as mock_register_artifact, \
         patch("src.agent.data_assurance_agent.DataAssuranceAgent.get_git_revision", return_value="FAKE_GIT_HASH"):

      
        # 2. SET RETURN VALUES
        
        mock_load.return_value = (fake_df, fake_paths, mock_features)

        # Confident learning output
        m7.return_value = (fake_df, 0.5)

        # Feature engineering output
        m8.return_value = (
            fake_df, fake_df["default.payment.next.month"],
            fake_df, fake_df["default.payment.next.month"],
            fake_df, fake_df["default.payment.next.month"],
            fake_df
        )

   
        # 3. RUN THE AGENT
       
        agent = DataAssuranceAgent()
        result = agent.run()

       
        # 4. VALIDATIONS
        print(f"Debug reuslt:{result}")
        assert result["status"]=="success"
  
        assert result['result']["interim_path"] == fake_paths["interim_path"]
        assert result['result']["confident_learning_path"] == fake_paths["confident_learning_path"]

        # Evidence & policy metadata saved
        assert result["result"]["evidence_id"] == "ART-ID-123"
        assert result["result"]["policy_id"] == "ART-ID-123"

        # Ensure hashing happened
        assert result["result"]["dataset_hashes"]["train_val_hash"] == "FAKE_HASH"
        assert result["result"]["dataset_hashes"]["test_hash"] == "FAKE_HASH"

        # Ensure all major pipeline steps were called
        
        mock_load.assert_called_once()
        m1.assert_called_once()
        m7.assert_called_once()
        m8.assert_called_once()
        mock_log_event.assert_called()
        mock_register_artifact.assert_called()

        # 5A — Verify total artifact registrations (Expected: 4)
        assert mock_register_artifact.call_count == 4, \
            "Expected 4 artifacts to be registered (Report, 2 datasets, 1 Policy)."

        # 5B — Validate dataset_version artifact metadata
        dataset_call = mock_register_artifact.call_args_list[1]
        args, kwargs = dataset_call

        # table_name check
        assert args[0] == "dataset_version", "Must register dataset_version for train+val split."

        # metadata checks
        assert "split_group" in args[1], "dataset_version metadata missing split_group"
        assert args[1]["split_group"] == "train+val", "split_group should be 'train+val'"
        assert "row_count" in args[1], "dataset_version metadata must include row_count"

        # 5C — Validate required log events
        log_event_types = [c.kwargs['event_type'] for c in mock_log_event.call_args_list]

        assert "calculating_confidence_learning" in log_event_types, \
            "Missing log event: calculating_confidence_learning"

        assert "DATA_HASH_SAVED" in log_event_types, \
            "Missing log event: DATA_HASH_SAVED"

        assert "POLICY_METADATA_SAVED" in log_event_types, \
            "Missing log event: POLICY_METADATA_SAVED"

        # 5D — Verify DATA_HASH_SAVED payload correctness
        hash_calls = [
            c for c in mock_log_event.call_args_list
            if c.kwargs.get("event_type") == "DATA_HASH_SAVED"
        ]

        assert len(hash_calls) >= 1, "DATA_HASH_SAVED must be logged at least once."

        # extract payload from the call
        payload = hash_calls[0].kwargs['extra_payload']

        assert "train_hash" in payload, "DATA_HASH_SAVED payload missing train_hash"
        assert "test_hash" in payload, "DATA_HASH_SAVED payload missing test_hash"

        print("\nINTEGRATION RESULT:", result)
   