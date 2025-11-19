import logging
from exception import CGAgentException
import pandas as pd
import os,sys
from pathlib import Path
from sklearn.utils import resample

def categorical_consolidation(df:pd.DataFrame,features:dict) ->pd.DataFrame:
    """Handle inconsistent categorical codes and dtype validation."""
    try:
        #basic sanity check
        expected_cols= set(features["raw_features"])
        missing_cols=expected_cols-set(df.columns)
        if missing_cols:
            print(f"warning :missing columns in dataset")
            logging.info("warning :missing columns in dataset")
    

        #checking for correct dtypes
        if "dtypes" in features:
            for col,dtype in features["dtypes"].items():
                if col in df.columns:
                        current_dtype= str(df[col].dtype)
                        if current_dtype!=dtype:
                            logging.info(f"Column '{col}' has dtype {current_dtype}, expected {dtype}.")

        #categorical consolidation

        df["EDUCATION"]=df["EDUCATION"].replace({5:4,6:4,0:4})
        logging.info("5,6 replaced by unknown in education column")

        df["MARRIAGE"]=df["MARRIAGE"].replace({0:3})
        logging.info("zero replaced by others")
        logging.info("Categorical consolidation completed")
        return df
    
    except Exception as e:
        raise CGAgentException(e,sys)

                        
def clip_outliers(df:pd.DataFrame,features:dict)->pd.DataFrame:
    """Clip repayment history columns to valid range."""
    try:
        pay_cols= [f"PAY_{i}"for i in [0,1,2,3,4,5,6]]

        for col in pay_cols:
            if col in df.columns:
                df[col]=df[col].clip(lower=-1,upper=9)

        logging.info("Repayment history columns clipped to range [-1, 9]")
        return df
    except Exception as e:
        raise CGAgentException(e,sys)

def handle_missing_values(df:pd.DataFrame):
    """Detect and impute missing values."""
    try:
        missing_col=[col for col in df.columns if df[col].isnull().any()]
        if len(missing_col)>0:
            print(f"cols has missing values {missing_col}")
            logging.info(f"cols has missing values {missing_col}")
        return missing_col
    except Exception as e:
        raise CGAgentException(e,sys)
        

def impute_missing_values(df:pd.DataFrame):  
    try:
        pay_cols= [f"PAY_{i}"for i in [0,1,2,3,4,5,6]]
        #handle time series msising values cols for financial cols
        for col in df.columns:
            if col in ["BILL_AMT","PAY_AMT"]:
                if df[col].isnull().any().any():
                    logging.info(f"missing values found for {df[col]}")
                    df[col].fillna(0,inplace=True)
                else:
                    logging.info("no misisng values found in any column")  
            elif col in ["AGE", "LIMIT_BAL"]:
                if df[col].isnull().any().any():
                    logging.info(f"missing values found for {df[col]}")
                    df[col].fillna(df[col].median(),inplace=True)
        
        existing_pay_cols=[c for c in pay_cols if c in df.columns]
        if existing_pay_cols and df[existing_pay_cols].isnull().any().any():
            df[existing_pay_cols] = df[existing_pay_cols].fillna(df[existing_pay_cols].mode().iloc[0])
            logging.info(f"Missing values imputed for PAY columns: {existing_pay_cols}")      
        return df
    except Exception as e:
        raise CGAgentException(e,sys)

def check_data_imbalance(df:pd.DataFrame):
    count= df["default.payment.next.month"].value_counts()
    print(f"\n class_count: \n",count)

    #class percenteges
    class_per= df["default.payment.next.month"].value_counts(normalize=True)*100
    print("\nClass percentages:\n", class_per)

    imbalance_ratio= class_per.max()/class_per.min()
    print(f"\n significant clas simbalance detected" ,{imbalance_ratio})
    
    if imbalance_ratio >1.6:
        print("\nSignificant class imbalance detected.")
        return True
    else:
        print("\n No major imbalance detected.")
        return False


def handle_imbalance_data(df:pd.DataFrame,target_col:str="default.payment.next.month",method: str = "upsample") -> pd.DataFrame:

    majority_class= df[target_col].value_counts().idxmax()
    minority_class= df[target_col].value_counts().idxmin()

    df_majority=df[df[target_col]==majority_class]
    df_minority= df[df[target_col]==minority_class]

    if method=="upsample":
        df_minority_upsampled=resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        df_balanced= pd.concat([df_majority,df_minority_upsampled])

    elif method=="downsample":
        df_majority_downsampled= resample(
            df_majority,
            replace=True,
            n_samples=len(df_minority),
            random_state=42
        )
    
        df_balanced= pd.concat([df_majority_downsampled,df_minority])
    else:
        raise ValueError("Method must be 'upsample' or 'downsample'")

    print(f"\n Data balanced using {method}. New class distribution:\n{df_balanced[target_col].value_counts()}")
    return df_balanced


def prepare_data(df:pd.DataFrame):
    try:
        # 1ID Handling
        id_col = "ID"
        if id_col in df.columns:
            ids = df[id_col]
        else:
            ids = None
        return df,ids
    except Exception as e:
        raise CGAgentException(e,sys)

