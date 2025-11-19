import pandas as pd
import logging
from exception import CGAgentException
import sys
from typing import Dict
from sklearn.model_selection import train_test_split

def feature_engineering(df:pd.DataFrame)->pd.DataFrame:
    try:
        pay_cols= [f"PAY_{i} "for i in [0,1,2,3,4,5,6]]

        for col in pay_cols:
            if col not in df.columns:
                logging.warning(f"Missing expected PAY column: {col}")
        
        
        df["total_months_late"]=df[pay_cols].apply(lambda row:sum(row>0),axis=1)

        logging.info("Created feature: TOTAL_MONTHS_LATE")
        return df
    except Exception as e:
        logging.info("Error creating TOTAL_MONTHS_LATE")
        raise CGAgentException(e,sys)
    

def create_utlization_rate(df:pd.DataFrame)->pd.DataFrame:
    """
    Creates UTILIZATION_RATE = (Latest bill amount) / LIMIT_BAL.
    Captures true financial pressure the client is under.
    """
    try:
        Latest_bill= df["BILL_AMT1"]
        df["utilization_rate"]= Latest_bill/df["LIMIT_BAL"]
        logging.info("Created feature: latest_bill")
        return df
    except Exception as e:
        logging.info("Error creating UTILIZATION_RATE")
        raise CGAgentException(e,sys)

def encode_categorical_features(df:pd.DataFrame,features:Dict)->pd.DataFrame:
    """
    Apply One-Hot Encoding to these categorical features
    """
    try:
        CATEGORICALS=features["categorical_cols"]
        for col in CATEGORICALS:
             if col not in df.columns:
                logging.info(f"Missing categorical column: {col}")
        
        df = pd.get_dummies(df,columns=CATEGORICALS, drop_first=False)
        
        logging.info("One-Hot Encoded demographic attributes (SEX, EDUCATION, MARRIAGE)")
        return df
    except Exception as e:
        logging.info("Error encoding categorical variables")
        raise CGAgentException(e,sys)

def scale_numeric_features(df:pd.DataFrame,features:Dict)->pd.DataFrame:
    """
    Scaling for numerical financial + age features.
    Using standard z-score scaling.
    """
    try:
        logging.info("Starting numeric feature scaling...")

        # A) Collect numeric columns
        num_cols = features["numerical_cols"]
        engineered_cols = ["utilization_rate", "total_months_late"]

        all_numeric_cols = num_cols + engineered_cols

        # B) Check for missing columns
        for col in all_numeric_cols:
            if col not in df.columns:
                logging.warning(f"Missing numerical column: {col}")

        # Filter only existing columns to avoid KeyErrors
        existing_num_cols = [c for c in all_numeric_cols if c in df.columns]

        if not existing_num_cols:
            logging.error("No numeric columns found for scaling.")
            return df

        # C) Apply scaling safely
        df[existing_num_cols] = (
            df[existing_num_cols] - df[existing_num_cols].mean()
        ) / df[existing_num_cols].std()

        logging.info(f"Scaled numeric columns: {existing_num_cols}")

        return df

    except Exception as e:
        raise CGAgentException(e, sys)

def get_features(df:pd.DataFrame,features:dict):
    "split datset in train and validation and test set"

    try:
        include_cols= features['train_features']["include_cols"]
        target_col= features["target"]

        X= df[include_cols]
        y=df[target_col]

        return X,y
         
    except Exception as e:
        raise CGAgentException(e,sys)

def train_val_split(X,y):
    try:
        logging.info("spliting train,test and val data")
        X_temp_train, X_test, y_temp_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
        )
        logging.info("data split in temp and test")
        X_train, X_val, y_train, y_val = train_test_split(
        X_temp_train, y_temp_train, test_size=0.125, random_state=42, shuffle=True
        )
        logging.info("data split in train and val")
        return X_train,y_train,X_val,y_val,X_test,y_test
       
    except Exception as e:
        raise CGAgentException(e,sys)

def run_feature_engineering(df:pd.DataFrame,features:dict):
    try:
        logging.info("feature engineering started")
        data=feature_engineering(df)
        new_data=create_utlization_rate(data)
        new_data=encode_categorical_features(new_data,features)
        new_data=scale_numeric_features(new_data,features)
        X,y=get_features(new_data,features)
        X_train,y_train,X_val,y_val,X_test,y_test=train_val_split(X,y)

        return X_train,y_train,X_val,y_val,X_test,y_test,new_data

    except Exception as e:
        raise CGAgentException(e,sys)
