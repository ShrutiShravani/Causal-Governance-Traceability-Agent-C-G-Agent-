import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import logging
import os,sys
from exception import CGAgentException

#base causal model

class CausalModel:
    """
    A governance-aligned causal ML model for tabular risk prediction.
    Uses Gradient Boosting (stable, interpretable, SHAP-compatible).
    """

    def __init__(self,features:list,model):
        self.features=features

        #simple robust grdaient boostign classifier
        self.model=GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=200,
            max_depth=3,
            subsample=0.9
        )
    
    def validate_features(self,X):
         """Ensure all required features exist in X."""
         missing= [f for f in self.features if f not in X.columns]
         if missing:
            raise CGAgentException(f"Missing required features for model :{missing}",sys)

    def fit(self,X,y):
        try:
            self.validate_features(X)
            logging.info(f"training on {X.shape[0]} rows")
            return self.model.fit(X[self.features],y)
        except ValueError as e:
    # Keep sklearn message clean
           raise CGAgentException(str(e), sys)
        except Exception as e:
            raise CGAgentException(e, sys)

    def predict(self,X):
        try:
            self.validate_features(X)
            logging.info(f"Starting prediciton")
            return self.model.predict(X[self.features])
        except Exception as e:
            raise CGAgentException(e,sys)

    def predict_proba(self,X):
        try:
            self.validate_features(X)
            return self.model.predict_proba(X[self.features])[:,1]
        except Exception as e:
            raise CGAgentException(e, sys)

#curriculum leanring scheduler

class CurriculumScheduler:
    """
    Handles the 3-phase curriculum:
    Phase A: High-confidence data only
    Phase B: Mix high + low in increasing proportions
    Phase C: Full dataset
    """

    def __init__(self,high_df,low_df):
        self.high_df= high_df
        self.low_df=low_df

    def get_batches(self)->list:
        """
        Curriculum schedule:
        Phase A: 3 epochs → only high quality
        Phase B: 3 epochs → 70% high + 30% low
        Phase C: 4 epochs → 100% high + 100% low
        """
        try:
            batches=[]
            
            #phase 1 warmup
            #epochs 1-3 :Only high confidence
            for _ in range(3):
                batches.append(self.high_df)

            
            #phase b Graduated mixing
            #epochs 4-6 :30% low+high
            mixed_30=pd.concat([
                self.high_df,
                self.low_df.sample(frac=0.30,random_state=42)
            ],axis=0)

            
            for _ in range(3):
                batches.append(mixed_30)

            #epcohs 7-10 :full low+high
            mixed_full =pd.concat([self.high_df,self.low_df],axis=0)

            for _ in range(4):
                batches.append(mixed_full)
            return batches
        except Exception as e: 
            raise CGAgentException(e,sys)
    
  