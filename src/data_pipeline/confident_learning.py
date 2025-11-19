import os,sys
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from typing import Dict,Any,Tuple
import logging
from exception import CGAgentException
import uuid


BASE_CLASSIFIER=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)

K_FOLDS=5
CONFIDENCE_THRESHOLD=0.5


def run_confident_learning_audit(df:pd.DataFrame,features:dict,ids)->Tuple:
    """
    Executes Confident Learning (Cleanlab) to identify and flag mislabeled samples
    in the high-risk financial dataset.

    Args:
     df: The interim DataFrame (post-cleaning, pre-feature engineering).
        target_col: The label column ('default.payment.next.month').
        id_col: The unique client ID column ('ID').
        output_dir: Path to save the governance reports (/data/quality/).
    
    Returns:
    A tuple containing:
        - mislabeled_df: DataFrame of client IDs flagged for review.
        - audit_payload: Dictionary containing the governance log entry.
    """
    try:
        logging.info("Starting Confident learning Audit")

        
        #prepare data for statistical audit

        include_cols = features["train_features"]["include"]
        target_col = features["target"]
        

        X = df[include_cols]
        y = df[target_col].values
        logging.info(f"Feature matrix shape: {X.shape}")
        logging.info(f"Labels shape: {y.shape}")

        
        
        #check for binary classification 
        unique_labels= sorted(list(set(y)))
        if len(unique_labels)<2:
            logging.info("Target column must have at least two unique classes for classification")
            raise ValueError("Target column must have at least two unique classes for classification")
        

        #execute confident learning(baseline model training)
        #automatically trains simple classifier required for the statistical chcek
        logging.info("Running StratifiedKFold probability estimation...")
        skf= StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=42)
        pred_probs= np.zeros((len(y),len(unique_labels)))


        for train_idx,val_idx in skf.split(X,y):
            model=BASE_CLASSIFIER
            model.fit(X.iloc[train_idx],y[train_idx])
            pred_probs[val_idx] =model.predict_proba(X.iloc[val_idx])
        
        logging.info("Probability matrix computed.")

        #find label issus and quality scores
        label_issues_indices= find_label_issues(
            labels=y,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence")
        
        logging.info(f"Found {len(label_issues_indices)} potential label issues.")

        label_quality_scores = get_label_quality_scores(y, pred_probs)

        # Filter to get the mislabelled report
        mislabeled_df = pd.DataFrame({
            "ID": ids.iloc[label_issues_indices].values,
            "original_label": y[label_issues_indices],
            "quality_score": label_quality_scores[label_issues_indices]
        })
        
        
        return mislabeled_df,CONFIDENCE_THRESHOLD
    except Exception as e:
        logging.error("Error occurred in Cleanlab audit.")
        raise CGAgentException(e, sys)



    

    




    








