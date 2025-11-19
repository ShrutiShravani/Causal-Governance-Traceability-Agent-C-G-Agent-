from typing import Dict,Any,List
import yaml
import os
import pandas as pd

def get_integrity_rules()->Dict[str,Any]:
    """Defines rules governing data cleaning and imputation strategies."""
    return {
        "categorical_consolidation":{
            "EDUCATION":"5/6/0 combined to 4(Unknown/Others)",
            "MARRIAGE":"0 consolidated by others" ,

        },
        "outlier_policy":{
            "PAY_COLS":"Range enforced [-1,9]"
        },
        "imputation_strategy_missing_values":{
            "PAY_STATUS": "MODE (Most Frequent Status)",
            "NUMERIC_FINANCIAL": "ZERO_IMPUTATION (BILL_AMT, PAY_AMT)",
            "NUMERIC_OTHER": "MEDIAN_IMPUTATION (AGE, LIMIT_BAL)"
        },
        
        "drop_cols":["ID"]
        
    }

def get_feature_audit_rules()->Dict[str,Any]:
    """Defines the features created and the attributes retained for policy audit."""
    return {
        "derived_features": [
            "total_months_late", 
            "utilization_rate"
        ],
        "encoding_method": "One-Hot Encoding",
        "scaling_method":"StandardScaler",
        "protected_attributes": ["SEX", "AGE", "EDUCATION", "MARRIAGE"]
    }



#master function for pac generation
def generate_pac_metadata(train_val_hashes,test_hashes,split_ratio="80/10/10",data_version:str="1.0")->Dict[str,Any]:
    """Aggregates all policies into the final, versionable metadata dictionary."""
    metadata={
        "governance_context":{
        "policy_id": f"DATA-ASSURANCE-{data_version}",
        "creation_timestamp": pd.Timestamp.now().isoformat(),
        "daa_agent_version": data_version,
        "split_strategy": split_ratio,
        "data_version_hash_key": train_val_hashes ,# Linkage to the training data hash
        "test_data_version_hash_key":test_hashes
    },
    "data_integrity_rules": get_integrity_rules(),
    "feature_engineering_rules": get_feature_audit_rules(),
    "governance_notes": [
            "All preprocessing choices are recorded for reproducibility.",
            "Hash values ensure deterministic dataset lineage.",
            "Protected attributes maintained for runtime policy auditing.",
            "Dataset is prepared for causal inference and XAI-based auditing."
        ]
    }
    return metadata

