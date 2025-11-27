from atexit import register
from psycopg2 import extras
import json
from typing import Dict, Any, Optional
import logging
from exception import CGAgentException
import os,sys
import psycopg2
import uuid
from datetime import datetime
from src.traceability.cloud_storage import CloudStorageService

DB_CONFIG={
    "dbname":"audit_db",
    "user":"test",
    "password":"test",
    "host":"localhost",
    "port":"5432"
}
#intialize GRAS Service on startup (o[ertaional fallback)
try:
    from src.traceability.traceability_agent_service  import TraceabilityAgentService 
    from src.traceability.graph_config import GraphDBClient
    GRAPH_CLIENT=GraphDBClient(use_mock=True)
    GRAS_SERVICE= TraceabilityAgentService(GRAPH_CLIENT)
    GRAS_ENABLED=True
    logging.info("GRAS Service initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize GARS BService.Proceeding with SQL only:{e}")
    GRAS_ENABLED=False

def get_db_connection():
    """Establishes and returns a database connection"""

    try:
        conn=psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection failed:{e}")
        logging.info(f"Database connection failed")
        raise CGAgentException(e,sys)

def log_event(agent_name:str,event_type:str,phase:str,client_transaction_id:Optional[str]=None,summary:Optional[str]=None,extra_payload:Optional[Dict[str,Any]]=None,trace_id: Optional[str] = None)->str:
    """
    Logs a discrete audit event into the append-only audit_event table.
    Called by DAA, CA, PA, and OA agents.
    """
    conn=None
    # 1. GENERATE IMMUTABLE IDENTIFIERS IN PYTHON
    event_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    
    # 2. CREATE CANONICAL RAW EVENT DICTIONARY (Source of Truth for GRAS)
    raw_event_data = {
        "event_id": event_id,
        "created_at": created_at,
        "trace_id": trace_id,
        "client_transaction_id": client_transaction_id,
        "agent_name": agent_name,
        "phase": phase,
        "event_type": event_type,
        "summary": summary,
        "extra": extra_payload if extra_payload else {}
    }
    try:
        conn=get_db_connection()
        cursor=conn.cursor()

        #prepare the core data for insertion
        # Prepare data for insertion
        event_data = (
            raw_event_data["event_id"],
            raw_event_data["created_at"],
            raw_event_data["trace_id"],
            raw_event_data["client_transaction_id"],
            raw_event_data["agent_name"],
            raw_event_data["phase"],
            raw_event_data["event_type"],
            raw_event_data["summary"],
            json.dumps(raw_event_data["extra"])
        )

        #sql injection prevention: sue parameter substitution
        sql="""
        INSERT INTO audit_event
        (event_id, created_at, trace_id, client_transaction_id, agent_name, phase, event_type, summary, extra)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING event_id;
        """
        cursor.execute(sql,event_data)
        event_id=cursor.fetchone()[0]
        conn.commit()

        if GRAS_ENABLED:
            try:
                 # Pass the complete, canonical data object to the synthesis layer
                 GRAS_SERVICE.synthesize_and_ingest_event(raw_event_data)
            except Exception as e:
                   logging.info(f"GARS Synthesis failed for event {event_id}:{e}")
                   
             
        logging.info(f"logged {event_type} by {agent_name}.ID: {event_id}")
        return str(event_id)
    except CGAgentException:
        # Re-raise the connection error received from get_db_connection()
         return "ERROR_NO_DB_CONNECTION"
    except Exception as e:
        print(f"failed to log audit event:{e}")
        if conn:
            conn.rollback()
        return "ERROR_LOGGING_FAILED"
    finally:
        if conn:
            conn.close()


def register_artifact(table_name: str,data: Dict[str, Any],client_transaction_id:Optional[str]) -> str:
    """
    A utility function to insert data into the Policy, Dataset, or Evidence tables.
    Used primarily by the DAA in Phase 1 for logging certified artifacts.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build dynamic SQL query safely
        columns = list(data.keys())
          #cloud tsorage integration
        local_path = data.get('dvc_path') or data.get('storage_path')
        if local_path and os.path.exists(local_path):
            try:
                cloud_storage= CloudStorageService(use_real_s3=True)

                    #uplaod to cloud and get public url
                cloud_url= cloud_storage.upload_artifact(
                        local_path=local_path,
                        artifact_type= table_name,
                        client_transaction_id=client_transaction_id 
                )

                #add cloud url to artifact data
                data['s3_path']=cloud_url
                logging.info(f"Uploaded {local_path} to S3: {cloud_url}")
            except Exception as e:
                logging.warning(f"S3 upload failed for {local_path}: {e}")
        
        columns=list(data.keys())
        values = [data[column] for column in columns]
        
        # Use parameter substitution for safety
        sql = f"""
        INSERT INTO {table_name} ({', '.join(columns)}) 
        VALUES ({', '.join(['%s'] * len(columns))})
        RETURNING *;
        """

    
        cursor.execute(sql, values) 
        result = cursor.fetchone()
         # FIXED: Get the UUID primary key (first column)
        if table_name == "dataset_version":
            registered_id = result[0]  # dataset_id UUID
        elif table_name == "policy_document":
            registered_id = result[0]  # policy_id UUID  
        elif table_name == "evidence_pointer":
            registered_id = result[0]  # evidence_id UUID
        else:
            registered_id= result[0]
        conn.commit()

    
        if GRAS_ENABLED:
            try:
                # Pass the complete, canonical data object to the synthesis layer
                GRAS_SERVICE.synthesize_and_ingest_artifact(table_name,data)
            except Exception as e:
                logging.info(f"GARS Synthesis failed for event {table_name}:{e}")
        logging.info(f" Registered artifact in {table_name}. ID: {registered_id}")
        return str(registered_id)
    
    except CGAgentException:
        # Re-raise the connection error received from get_db_connection()
        return None
    except Exception as e:
        print(f" Failed to register artifact in {table_name}: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def get_model_version_info(self):
    """get model info from evidence pointer table"""
    try:
        conn= get_db_connection()
        cursor= conn.cursor()

        cursor.execute("""
         SELECT evidence_id,mlflow_model_name,mlflow_model_version, mlflow_run_id,dvc_path
            FROM evidence_pointer 
            WHERE kind = 'TRAINED_MODEL'
            ORDER BY created_at DESC 
            LIMIT 1
        """)

        result= cursor.fetchone()
        conn.close()

        if result:
            return {
                "evidence_id":result[0],
                "mlflow_model_name": result[1],
                "mlflow_model_version": result[2], 
                "mlflow_run_id": result[3],
                "model_path": result[4]

            }
        return None
    except Exception as e:
        logging.warning(f"Could not fetch model info from DB: {e}")
        return None