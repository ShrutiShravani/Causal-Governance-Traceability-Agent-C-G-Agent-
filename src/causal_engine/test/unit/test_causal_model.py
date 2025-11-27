from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from exception import CGAgentException
import pytest
from src.causal_engine.causal_model import CausalModel
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import exception


@pytest.fixture
def sample_data():
    """sample synthetic data to test prdiction"""
    df=pd.DataFrame({
     "LIMIT_BAL": [10000, 20000, 30000, 40000, 15000],
     "AGE":        [25, 45, 35, 50, 29],
     "PAY_1":      [0, -1, 1, 0, 0],
     "PAY_2":      [0, -1, 1, 0, 0],
        })
    
    y = np.array([0, 1, 0, 1, 0])
    return df,y

def test_model_instance() :
    """
    Tests that the self.model attribute is an instance of 
    GradientBoostingClassifier after initialization.
    """
    handler_instance= CausalModel(features=["LIMIT_BAL", "AGE"],model=None)
    #ASSERT that the model  attribute exist and is istance of expected class
    assert hasattr(handler_instance,'model')
    assert isinstance(handler_instance.model, GradientBoostingClassifier)

    
def test_model_shape_mismatch():
    model = CausalModel(features=["LIMIT_BAL"], model=None)
    X_train_mismatch= pd.DataFrame({"LIMIT_BAL":np.random.rand(100)})
    y_train_mismatch= np.random.rand(50)
    
    with pytest.raises(exception.CGAgentException):
        model.fit(X_train_mismatch, y_train_mismatch)

def test_model_predict(sample_data):

    df, y = sample_data
    model = CausalModel(features=["LIMIT_BAL", "AGE", "PAY_1", "PAY_2"],model=None)

    # Train first
    model.fit(df, y)

    preds = model.predict(df)

    # Assertions
    assert len(preds) == len(df)
    assert set(preds).issubset({0, 1}), "Predictions must be binary"


def test_predict_proba_after_training(sample_data):
    df,y= sample_data
    model=CausalModel(features=["LIMIT_BAL", "AGE", "PAY_1", "PAY_2"],model=None)

    model.fit(df,y)
    proba=model.predict_proba(df)

    #assertions
    assert len(proba) == len(df)
    assert np.all(proba>=0) and np.all(proba<=1),"Probabilites must be in [0,1]"
    assert proba.ndim == 1  #because your code returns [:,1]

def test_predict_missing_feature_should_fail(sample_data):
    df, y = sample_data
    model = CausalModel(features=["LIMIT_BAL", "AGE", "PAY_1", "PAY_2", "MISSING"],model=None)

    # Training should fail due to missing feature
    with pytest.raises((CGAgentException)):
        model.fit(df,y)