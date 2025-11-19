from psycopg2 import extras
import json
from typing import Dict, Any, Optional
import logging
from exception import CGAgentException
import os,sys
import psycopg2
from src.traceability.traceability_agent_service  import TraceabilityAgentService 
from src.traceability.graph_config import GraphDBClient
import uuid
from datetime import datetime

DB_CONFIG={
    "dbname":"audit_db",
    "user":"audit_user",
    "password":"secure_password",
    "host":"localhost",
    "port":"5432"
}
#intialize GRAS Service on startup (o[ertaional fallback)
try:
    GRAPH_CLIENT=GraphDBClient()
    GRAS_SERVICE= TraceabilityAgentService(GRAPH_CLIENT)
    GRAS_ENABLED=True
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


def register_artifact(table_name: str,data: Dict[str, Any]) -> str:
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
        values = [data[column] for column in columns]
        
        # Use parameter substitution for safety
        sql = f"""
        INSERT INTO {table_name} ({', '.join(columns)}) 
        VALUES ({', '.join(['%s'] * len(columns))})
        RETURNING *;
        """
        
        cursor.execute(sql, values) 
        registered_id = cursor.fetchone()[0]
        conn.commit()
        
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