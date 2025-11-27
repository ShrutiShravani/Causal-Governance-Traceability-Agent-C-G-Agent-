import pytest
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from src.causal_engine.explainer import XAIEngine

class DummyCausalModel:
    def __init__(self,model):
        self.model=model

@pytest.fixture
def sample_dataset(tmp_path):
    """
    Creates a tiny dataset for SHAP interpreation

    """
    df=pd.DataFrame({
        "ID": [1, 2],
        "feat1": [0.5, 0.2],
        "feat2": [1.0, 0.3],
        "default.payment.next.month": [1, 0]
    })
    
    data_path= tmp_path/"X_val.csv"
    df.to_csv(data_path,index=False)

    return data_path


@pytest.fixture
def trained_model(tmp_path):
    """
    Creates and saves a minimal trained model to be used by the XAI engine.
    """
    df = pd.DataFrame({
        "feat1": [0.5, 0.2, 0.8, 0.1],
        "feat2": [1.0, 0.3, 0.5, 0.2]
    })
    y = [1, 0, 1, 0]
    clf = GradientBoostingClassifier()
    clf.fit(df, y)
    
 
   
    causal_model= DummyCausalModel(clf)

    model_path= tmp_path/"trained_model.pkl"
    
    with open(model_path,"wb") as f:
        pickle.dump(causal_model, f)

    return model_path

def test_xai_engine(trained_model,sample_dataset,tmp_path,monkeypatch):
    """
    FULL INTEGRATION TEST:
    - Loads real model
    - Runs SHAP explainer
    - Produces global/local explanations
    - Generates shap summary plot
    """

    #Ensure SHAP artifacts to temp folder
    monkeypatch.setenv("SHAP_OUTPUT_DIR",str(tmp_path))

    result= XAIEngine.generate_explanations(
        model_artifact_path=str(trained_model),
        X_data_path=str(sample_dataset)
    )

    #validate output structure
    assert isinstance(result,dict)
    assert "global_feature_importance" in result
    assert "local_top_drivers" in result
    assert "risk_factors" in result
    assert "causal_graph" in result
    assert "expected_value" in result
    
    #validate contemnt
    assert len(result['global_feature_importance'])>0
    assert len(result["local_top_drivers"])>0
    assert len(result['risk_factors'])>0




