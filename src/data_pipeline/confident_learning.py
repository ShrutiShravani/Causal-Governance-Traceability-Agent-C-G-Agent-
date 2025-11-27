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


def run_confident_learning_audit(X_train,y_train,features)->Tuple:
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

    
        logging.info(f"Feature matrix shape: {X_train.shape}")
        logging.info(f"Labels shape: {y_train.shape}")
        logging.info(f"Training class distribution:{pd.Series(y_train).value_counts().to_dict()}")
        
        if "ID" not in X_train.columns:
            logging.info("ID column missing")
        ids=X_train['ID'].values
       
        
        cl_X_train=X_train.drop('ID',axis=1)
        cl_y_train=y_train.copy()
    
        #check for binary classification 
        unique_labels= sorted(list(set(y_train)))
        if len(unique_labels)<2:
            logging.info("Target column must have at least two unique classes for classification")
            raise ValueError("Target column must have at least two unique classes for classification")
        

        #execute confident learning(baseline model training)
        #automatically trains simple classifier required for the statistical chcek
        logging.info("Running StratifiedKFold probability estimation...")
        skf= StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=42)
        pred_probs= np.zeros((len(y_train),len(unique_labels)))
        print(pred_probs[:1])

        
        for train_idx,val_idx in skf.split(cl_X_train,cl_y_train):
            model=BASE_CLASSIFIER
            logging.info("model fit on train data")
            model.fit(cl_X_train.iloc[train_idx],cl_y_train.iloc[train_idx])
            logging.info("model fited on train data")
            pred_probs[val_idx] =model.predict_proba(cl_X_train.iloc[val_idx])

        
        logging.info("Training Probability matrix computed.")

        #find label issus and quality scores
       
        label_issues_indices= find_label_issues(
            labels=y_train,
            pred_probs=pred_probs,
            return_indices_ranked_by=None)
        
        logging.info(f"Found {len(label_issues_indices)} potential label issues.")
        
        label_quality_scores = get_label_quality_scores(cl_y_train,pred_probs)
        logging.info(f"Training quality scores range: {label_quality_scores.min():.3f} - {label_quality_scores.max():.3f}")
        """
        # Filter to get the mislabelled report
        mislabeled_df = pd.DataFrame({
            "ID":ids[label_issues_indices],
            "original_label": y_train[label_issues_indices],
            "predicted_labels":np.argmax(pred_probs[label_issues_indices],axis=1),
            "confidence_score":np.max(pred_probs[label_issues_indices],axis=1),
            "quality_score": label_quality_scores[label_issues_indices],
            "is_issue":True
        })
        """
        mislabeled_df = pd.DataFrame({
            "ID":ids,
            "original_label": y_train,
            "predicted_labels":np.argmax(pred_probs,axis=1),
            "confidence_score":np.max(pred_probs,axis=1),
            "quality_score": label_quality_scores,
            "is_issue":False
        })
        mislabeled_df.loc[label_issues_indices, "is_issue"] = True
        num_issue= len(mislabeled_df[mislabeled_df["is_issue"]==True])
        print(f"{num_issue}")
      
        return mislabeled_df
    except Exception as e:
        logging.error("Error occurred in Cleanlab audit.")
        raise CGAgentException(e, sys)



    

    




    








