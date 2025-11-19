import pytest
import pandas as pd
from src.data_pipeline.data_cleaning import categorical_consolidation,clip_outliers,prepare_data,handle_imbalance_data,handle_missing_values,check_data_imbalance,impute_missing_values
from exception import CGAgentException
import yaml

@pytest.fixture
def temp_features_path(tmp_path):
    content={
        "raw_features": ["EDUCATION", "MARRIAGE", "AGE"],
        "dtypes": {
            "EDUCATION": "int64",
            "MARRIAGE": "int64",
            "AGE": "int64"
        }
    }
    features_path= tmp_path/"feature_schema.yaml"
    with open(features_path, "w") as f:
        yaml.safe_dump(content, f)
    return features_path, content

def test_categorical_consolidation(temp_features_path):
    features_path,content= temp_features_path
    #sample dataframe 
    df=pd.DataFrame({
        "EDUCATION":[1,5,6,0],
        "MARRIAGE":[1,0,2,3],
        "AGE":[25,30,35,40]
    })

    
    res= categorical_consolidation(df.copy(),content)

    #assert trasnformations
    assert all(res["EDUCATION"].isin([1,4]))
    assert all(res["MARRIAGE"].isin([1,2,3]))

    for col,dtype in content['dtypes'].items():
        for col in df.columns:
            assert str(df[col].dtype)==dtype
  
def test_clip_outliers():
    df=pd.DataFrame({
    "PAY_0":[10,-5,2],
    "PAY_1":[0,12,-2]
})
    features={}

    res= clip_outliers(df.copy(),features)
    assert res["PAY_0"].max()<=9
    assert res["PAY_0"].min()>=-1


def test_check_data_imbalance_detected():
    df = pd.DataFrame({
        "default.payment.next.month": [0]*90 + [1]*10
    })
    result = check_data_imbalance(df)
    assert result is True  # imbalance should be detected


def test_check_data_imbalance_not_detected():
    df = pd.DataFrame({
        "default.payment.next.month": [0]*50 + [1]*50
    })
    result = check_data_imbalance(df)
    assert result is False



def test_handle_imbalance_data_upsample():
    df = pd.DataFrame({
        "default.payment.next.month": [0]*80 + [1]*20,
        "AGE": list(range(100))
    })
    balanced = handle_imbalance_data(df,method="upsample")
    counts = balanced["default.payment.next.month"].value_counts()
    assert counts[0] == counts[1]  # perfectly balanced

def test_handle_imbalance_data_downsample():
    df = pd.DataFrame({
        "default.payment.next.month": [0]*80 + [1]*20,
        "AGE": list(range(100))
    })
    balanced = handle_imbalance_data(df, method="downsample")
    counts = balanced["default.payment.next.month"].value_counts()
    assert counts[0] == counts[1]  # balanced after downsampling


def test_prepare_data_with_id():
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "AGE": [25, 30, 35]
    })
    result_df, ids = prepare_data(df)
    assert "ID" in result_df.columns
    assert ids.equals(df["ID"])

def test_prepare_data_without_id():
    df = pd.DataFrame({
        "AGE": [25, 30, 35]
    })
    result_df, ids = prepare_data(df)
    assert ids is None

def test_handle_missing_values_detects_missing(monkeypatch):
    df = pd.DataFrame({
        "A": [1, None, 3],
        "B": [4, 5, 6]
    })

    # Capture print output
    missing_cols = handle_missing_values(df)
    assert "A" in missing_cols
    assert isinstance(missing_cols, list)
    assert len(missing_cols) == 1


def test_handle_missing_values_no_missing():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    missing_cols = handle_missing_values(df)
    assert missing_cols == []

def test_impute_missing_values_replaces_nans():
    df = pd.DataFrame({
        "BILL_AMT": [1000, None, 2000],
        "PAY_AMT": [None, 500, 600],
        "AGE": [25, None, 30],
        "LIMIT_BAL": [None, 20000, 30000],
        "PAY_0": [None, 1, 0]
    })

    df_filled = impute_missing_values(df)

    # Check all NaNs filled
    assert not df_filled.isnull().any().any(), "Some NaN values remain after imputation"

def test_impute_missing_values_handles_invalid_df():
    with pytest.raises(CGAgentException):
        impute_missing_values(None)
 