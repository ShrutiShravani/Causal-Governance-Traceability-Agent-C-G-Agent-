from src.data_pipeline.data_loader import Data_Reader
from src.data_pipeline.data_cleaning import categorical_consolidation,clip_outliers,handle_missing_values,impute_missing_values,prepare_data,check_data_imbalance,apply_smote_with_preserved_ids,check_duplicates
from src.data_pipeline.confident_learning import run_confident_learning_audit
from src.data_pipeline.feature_engineering import feature_engineering,create_utlization_rate,scale_numeric_features,get_features,train_val_split
from src.data_pipeline.data_hashing import generate_sha256_hash,dataframe_to_stable_bytes
from pathlib import Path
from logger import logging
import json
from src.data_pipeline.preprocess_policies import generate_pac_metadata
import yaml
from src.traceability.audit_logger import log_event, register_artifact
import pandas as pd
import subprocess
import uuid
from exception import CGAgentException
import os,sys
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import joblib

class DataAssuranceAgent:
    agent_name="DataAssuranceAgent"
    
    def __init__(self,trace_id:str,client_transaction_id:str):
        self.pipeline_trace_id = trace_id #str(uuid.uuid4()).replace('-', '')[:32]
        self.client_transaction_id=client_transaction_id
    def get_git_revision(self):
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    logging.info("data_assurance_agent_started")
        
    def run(self,payload:dict=None)->dict:
        try:
            output=self._execute_data_assurance()
            return {
                "agent":self.agent_name,
                "status":"success",
                "result":output
            }
        except Exception as e:
            log_event(
                agent_name=self.agent_name,
                trace_id=self.pipeline_trace_id,
                client_transaction_id=self.client_transaction_id,
                phase="Phase1_DataAssurance",
                event_type="ERROR",
                summary=str(e),
                extra_payload=None
            )
            return {
                "agent":self.agent_name,
                "status":"failed",
                "error":str(e)
            }


    def _execute_data_assurance(self):
        logging.info("Data loader started")
        df,paths,features=Data_Reader.load_data()
       

        if not isinstance(df, pd.DataFrame):
                raise CGAgentException("Loaded object is not a DataFrame",sys)

        #chcek duplicates
        df=check_duplicates(df)

        #categorical consolidtarion
        df=categorical_consolidation(df,features)
        
        # Step 2: Clip outliers
        df = clip_outliers(df)

        
        # Step 3: check for missing values
        df = handle_missing_values(df)

        # Step 4: Impute missing values
        df= impute_missing_values(df)

         #save interim data
        interim_path= Path(paths["interim_path"])
        interim_path.parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(interim_path, index=False)
        logging.info(f"Saved interim dataset to {interim_path}")

        #Step 5 : apply fetaurr engineering 
        logging.info("feature engineering started")
        data=feature_engineering(df)

        #Step 6: create new columns
        new_data=create_utlization_rate(data)
        #new_data=encode_categorical_features(new_data,features)
        print(new_data.columns)

        #STep 7 :scale numerical columns
        new_data,scaling_params=scale_numeric_features(new_data,features)
         
        preprocessed_path= Path(paths["preprocessed_path"])
        preprocessed_path.parent.mkdir(parents=True,exist_ok=True)
        
        joblib.dump(scaling_params,preprocessed_path)
        
        # Step 8: Chcek for class imbalance
        is_imbalanced= check_data_imbalance(new_data)
        if is_imbalanced:
            new_data1=apply_smote_with_preserved_ids(new_data,target_col="default_payment_next_month")
        else:
           logging.info("Data is balanced. Skipping SMOTE.")
        logging.info(new_data1.shape)
        #chcek if it has duplicate values
        initial_rows=len(new_data1)
        duplicates= new_data1['ID'].duplicated().sum()
        if duplicates>0:
            logging.info(f"found{duplicates} out of {initial_rows}")
        
        
        # Step 6 :prepare train data
        ids=prepare_data(new_data1)

        X,y= get_features(new_data1,features)
        X_train,y_train,X_val,y_val=train_val_split(X,y)
    
        logging.info(X_train.shape)
        logging.info(X_val.shape)


        final_path= Path(paths["target_path"])
        X_train_path=Path(paths["X_train_path"])
        y_train_path=Path(paths["y_train_path"])
        X_val_path=Path(paths["X_val_path"])
        y_val_path= Path(paths["y_val_path"])

        for p in [final_path,X_train_path,y_train_path,X_val_path,y_val_path,]:
            p.parent.mkdir(parents=True,exist_ok=True)
            
        
    
        new_data1.to_csv(final_path,index=False)
        X_train.to_csv(X_train_path,index=False)
        X_val.to_csv(X_val_path,index=False)
        y_train.to_csv(y_train_path,index=False)
        y_val.to_csv(y_val_path,index=False)
      


        logging.info(f"Saved final processed dataset to {final_path}")
        logging.info(f"Saved training data to {X_train_path},{y_train_path}")
        logging.info(f"Saved validation data to {X_val_path},{y_val_path}")
       
        #calculate versoning
        X_train_val_hash_path=Path(paths["X_train_val_hash"])
       
        X_train_val_hash_path.parent.mkdir(parents=True, exist_ok=True)
    
        
        
        logging.info("concat train and val data")
        train_val_df = pd.concat(
        [
            X_train.reset_index(drop=True), 
            y_train.reset_index(drop=True), 
            X_val.reset_index(drop=True), 
            y_val.reset_index(drop=True)
        ],
        axis=1
    )
        print("train and val data concatenated")
        train_val_bytes = dataframe_to_stable_bytes(train_val_df)
        train_val_hash = generate_sha256_hash(train_val_bytes)

        with open(X_train_val_hash_path, "w") as f:
            json.dump({"train_val_hash":train_val_hash}, f, indent=4)

        logging.info(f"Saved TRAIN+VAL hash → {X_train_val_hash_path}")

        data_hash_event_id=log_event(
        agent_name=self.agent_name,
        trace_id=self.pipeline_trace_id ,
        client_transaction_id=self.client_transaction_id,
        phase="Phase1_DataAssurance",
        event_type="DATA_HASH_SAVED",
        summary="Final Data Version Certified",
        extra_payload={"train_hash":train_val_hash}
    )

        dataset_train_val_id=register_artifact("dataset_version",{
            "dataset_version":"DATA_V1",
            "split_group":"train+val",
            "sha256":train_val_hash,
            "dvc_rev":self.get_git_revision(),
            "dvc_path":str(X_train_val_hash_path),
            "row_count": len(X_train) + len(X_val)+len(y_train)+len(y_val),
            "col_count": X_train.shape[1]},
            self.client_transaction_id #add remote dvc path
            )

        # LINK: Data hash event to dataset (ONLY dataset_id)
        register_artifact("event_link", {
            "event_id": data_hash_event_id,
            "dataset_id": dataset_train_val_id
            # policy_id: NULL, evidence_id: NULL (automatically)
        },self.client_transaction_id)
  
        pac_metadata= generate_pac_metadata(train_val_hashes=train_val_hash,split_ratio="80/10/10",data_version="1.0")
        metadata_path=Path(paths["pac_metadata"])
        metadata_path.parent.mkdir(parents=True,exist_ok=True)
        with open(metadata_path,"w") as f:
            yaml.dump(pac_metadata,f,sort_keys=False)

        policy_id= register_artifact("policy_document",{
            "policy_name":"Data-Preprocess-Policy",
            "policy_version":"V1.0",
            "content_sha256":generate_sha256_hash(
                 open(metadata_path, "rb").read()),
            "storage_path":str(metadata_path)
        },self.client_transaction_id)

        policy_event_id=log_event(
            agent_name=self.agent_name,
            trace_id=self.pipeline_trace_id ,
            client_transaction_id=self.client_transaction_id,
            phase="Phase1_DataAssurance",
            event_type="POLICY_METADATA_SAVED",
            summary="PaC metadata stored",
            extra_payload={"policy_id": policy_id}
        )

        #LINK: Policy event to policy document (ONLY policy_id)
        register_artifact("event_link", {
            "event_id": policy_event_id, 
            "policy_id": policy_id
            # dataset_id: NULL, evidence_id: NULL (automatically)
        }, self.client_transaction_id)

        #Step 7 Confident learning
        df=pd.concat([X_train.reset_index(drop=True),y_train.reset_index(drop=True)],axis=1)
        mislabelled_df = run_confident_learning_audit(X_train,y_train,features)
        if mislabelled_df is None or not isinstance(mislabelled_df,pd.DataFrame):
                raise CGAgentException("Confident Learning returned invalid output",sys)

        #save governance report
        confident_learning_report= Path(paths["confident_learning_path"])
        confident_learning_report.parent.mkdir(parents=True,exist_ok=True)
        
        
        mislabelled_df.to_csv(confident_learning_report,index=False)
        logging.info(f"Saved Cleanlab report to {confident_learning_report}")
       
       #register evidence pointer
        confidence_evidence_id= register_artifact("evidence_pointer",{
            "kind":"CONFIDENT_LEARNING_REPORT",
            "dvc_rev":self.get_git_revision(),
            "dvc_path":str(confident_learning_report), #here add relative dvc path wher eit is versoined
            "sha256":generate_sha256_hash(
                dataframe_to_stable_bytes(mislabelled_df))
        },self.client_transaction_id)

        confidence_event_id=log_event(
            agent_name=self.agent_name,
            trace_id=self.pipeline_trace_id,
            client_transaction_id=self.client_transaction_id,
            phase="Phase1_DataAssurance",
            event_type="calculating_confidence_learning",
            summary=f"{len(mislabelled_df[mislabelled_df["is_issue"]==True])} potential label issues",
            extra_payload={"evidence_id":confidence_evidence_id}
        )

        #Confidence event to evidence (ONLY evidence_id)
        register_artifact("event_link", {
            "event_id": confidence_event_id,
            "evidence_id": confidence_evidence_id}
            ,self.client_transaction_id)
            # policy_id: NULL, dataset_id: NULL (automatically)
    
        #Preapre Agent alert payload(Governance Decision
        mislabelled_report_path=Path(paths['mislabelled_report_path'])
        mislabelled_report_path.parent.mkdir(parents=True,exist_ok=True)
        
        num_mislabelled=len(mislabelled_df[mislabelled_df["is_issue"]==True])
 
        if num_mislabelled>0:
            alert_action= "Esclate for human review"
        else:
            alert_action="Data Quality check passed"
        
        audit_payload = {
            "event_type": "DataQualityAssurance",
            "agent_name": "DataAssuranceLayer",
            "data_quality_status": "Flagged_Noise",
            "num_mislabeled_records": num_mislabelled,
            "action_required": alert_action,
            "governance_rationale": "High-risk financial labels checked for integrity defects.",
            "report_path": str(mislabelled_report_path)
        }

        with open(mislabelled_report_path,"w") as f:
            json.dump(audit_payload,f,indent=4)
        
        print(f"[INFO] Audit Payload generated for Traceability Layer at {mislabelled_report_path}")
        print(f"[INFO] {num_mislabelled} possible mislabeled samples found. Action: {alert_action}")
        
        logging.info(f"Audit payload saved at {mislabelled_report_path}")


        return {
            "interim_path":str(interim_path),
            "confident_learning_path":str(confident_learning_report),
            "dataset_hashes":{
                "train_val_hash_path":str(X_train_val_hash_path),
                "train_val_hash":train_val_hash
            },
            "dataset_id":dataset_train_val_id,
            "evidence_id":confidence_evidence_id,
            "policy_id":policy_id,
            "dataset_versions":{
                "final_dataset":str(final_path),
                "X_train":str(X_train_path),
                "y_train":str(y_train_path),
                "X_val":str(X_val_path),
                "y_val":str(y_val_path)
            },
            "mislabelled_report":str(mislabelled_report_path),
            "pac_metadata":str(metadata_path),

        }
