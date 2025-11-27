import pandas as pd
from logger import logging
from exception import CGAgentException
import os,sys

def read_data(path:str)->pd.DataFrame:
    try:
        data=pd.read_csv(path)
        return data
    except Exception as e:
        raise CGAgentException(e,sys)


def curriculum_prep(X_train_path:str,y_train_path:str,confidence_learning_report:str,quality_threshold:float=0.9):
    #load data
    try:
        logging.info("Loading Train/Val splits and Cleanlab scores...")
        X_train= read_data(X_train_path)
        y_train= read_data(y_train_path)
        cl_scores=read_data(confidence_learning_report)
        logging.info(cl_scores.shape)

 
        #Merge scores into training data using ID
        train=X_train.copy()
        train["y"]=y_train.values
        print(train.shape)
        logging.info(f"using quality threshold for curriculum :{quality_threshold}")
        
        train_df=train.merge(cl_scores[["ID","quality_score"]],on="ID",how="left")
        logging.info(f"After CL merge: {train_df.shape}")

        
    
        #segment into high and low confidence batches
        #high confidence batches
        high_conf_train=train [train_df['quality_score']>=quality_threshold]
        low_conf_train= train[train_df['quality_score']<quality_threshold]
        
        
        logging.info(f"High confidence rows :{len(high_conf_train)} | low_confidence_rows:{len(low_conf_train)}")
        return high_conf_train,low_conf_train
    
    except Exception as e:
        raise CGAgentException(e,sys)




    




    
