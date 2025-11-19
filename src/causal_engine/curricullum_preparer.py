import pandas as pd
import logging
from exception import CGAgentException
import os,sys

def read_data(path:str)->pd.DataFrame:
    try:
        data=pd.read_csv(path)
        return data
    except Exception as e:
        raise CGAgentException(e,sys)


def curriculum_prep(X_train_path:str,y_train_path:str,X_val_path:str,y_val_path:str,confidence_learning_report:str):
    #load data
    try:
        logging.info("Loading Train/Val splits and Cleanlab scores...")
        X_train= read_data(X_train_path)
        y_train= read_data(y_train_path)
        X_val= read_data(X_val_path)
        y_val= read_data(y_val_path)
        cl_scores=read_data(confidence_learning_report)

 
        #Merge scores into training data using ID
        train=X_train.copy()
        train["y"]=y_train.values
        logging.info("merging quality scores with train data")
        train=train.merge(cl_scores[["ID","quality_score"]],on="ID",how="left")
        
        if train["quality_score"].isna().sum()>0:
            logging.info("Some rows have missing clean lab scores")


        #segment into high and low confidence batches
        #high confidence batches
        high_conf_train=train [train['quality_score']>=0.90]
        low_conf_train= train[train['quality_score']<0.90]
        
        logging.info(f"High confidence rows :{len(high_conf_train)} | low_confidence_rows:{len(low_conf_train)}")
        return high_conf_train,low_conf_train,X_train,y_train,X_val,y_val,cl_scores
    
    except Exception as e:
        raise CGAgentException(e,sys)




    




    
