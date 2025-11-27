import logging
from exception import CGAgentException
import pandas as pd
import os,sys
from pathlib import Path
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def check_duplicates(df:pd.DataFrame) ->pd.DataFrame:
    """chcek for duplicate values"""
    try:
        initial_rows= len(df)
        #chcek for duplicate ids
        duplicate_ids= df['ID'].duplicated().sum()

        if duplicate_ids>0:
            logging.info(f"found{duplicate_ids} out of {initial_rows}")
        
        #log samples of duplicates
        dupl_samples= df[df["ID"].duplicated(keep=False)]['ID'].unique()[:5]
        logging.info(f"Sample duplicate IDs: {dupl_samples.tolist()}")
        
        #drop duplicate ids
        # Remove duplicate IDs (keep first occurrence)
        df_clean = df.drop_duplicates(subset=['ID'], keep='first')
        logging.info(f"Removed {duplicate_ids} duplicate IDs. {initial_rows} → {len(df_clean)}")
        return df_clean
    except Exception as e:
        raise CGAgentException(e,sys)

def categorical_consolidation(df:pd.DataFrame,features:dict) ->pd.DataFrame:
    """Handle inconsistent categorical codes and dtype validation."""
    try:
        #basic sanity check
        df.columns = df.columns.str.strip()   
        logging.info(df.columns)
        expected_cols= set(features["raw_features"])
        missing_cols=expected_cols-set(df.columns)
        if missing_cols:
            print(f"warning :missing columns in dataset")
            logging.info(f"warning :missing columns in dataset{missing_cols}")
    

        #checking for correct dtypes
        if "dtypes" in features:
            for col,dtype in features["dtypes"].items():
                if col in df.columns:
                        current_dtype= str(df[col].dtype)
                        if current_dtype!=dtype:
                            logging.info(f"Column '{col}' has dtype {current_dtype}, expected {dtype}.")

        #categorical consolidation

        df["EDUCATION"]=df["EDUCATION"].replace({5:4,6:4,0:4})
        logging.info("5,6 replaced by 0 in education column")

        df["MARRIAGE"]=df["MARRIAGE"].replace({0:3})
        logging.info("zero replaced by 3 (3rd category)")
        logging.info("Categorical consolidation completed")
        return df
    
    except Exception as e:
        raise CGAgentException(e,sys)

                        
def clip_outliers(df:pd.DataFrame)->pd.DataFrame:
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
        missing_cols=[]
        for col in df.columns:
            if df[col].isnull().any():
                missing_cols.append(col)
                print(f"cols has missing values {col}")
                logging.info(f"cols has missing values {col}")
        
        if not missing_cols:
            logging.info("No misisng values found")
        return df
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
        else:
            logging.info("No missing values in any cols")    
        return df
    except Exception as e:
        raise CGAgentException(e,sys)

def check_data_imbalance(df: pd.DataFrame):
    try:
        logging.info("Checking data imbalance")

        class_per = df["default_payment_next_month"].value_counts(normalize=True) * 100
        print("\nClass percentages:\n", class_per)

        min_pct = class_per.min()
        max_pct = class_per.max()
        imbalance_ratio = max_pct / min_pct

        logging.info(f"\nImbalance ratio: {imbalance_ratio:.2f}")


        # RULE 2 — imbalance ratio > 1.6 → upsample
        if imbalance_ratio > 1.1:
            logging.info("Using SMOTE for balancing")
            return True
        else:
            logging.info("No balancing needed")
            return False
    except Exception as e:
        raise CGAgentException(e,sys)

def apply_smote_with_preserved_ids(df, target_col="default_payment_next_month"):
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        original_ids = df["ID"].values
        max_old_id = df["ID"].max()
        original_count = len(df)

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        total_after = len(X_res)

        new_ids = []
    

        for i in range(total_after):
            if i < original_count:
                # ORIGINAL ROW → KEEP ORIGINAL ID
                new_ids.append(original_ids[i])
               
            else:
                # SMOTE-GENERATED SAMPLE → NEW UNIQUE ID
                new_ids.append(max_old_id + (i - original_count) + 1)
        
        df_bal = pd.DataFrame(X_res, columns=X.columns)
        df_bal[target_col] = y_res
        df_bal["ID"] = new_ids
      

        return df_bal

    except Exception as e:
        raise CGAgentException(e,sys)

def prepare_data(df:pd.DataFrame):
    try:
        # 1ID Handling
        id_col = "ID"
        if id_col in df.columns:
            ids = df[id_col]
        else:
            ids = None
        return ids
    except Exception as e:
        raise CGAgentException(e,sys)

