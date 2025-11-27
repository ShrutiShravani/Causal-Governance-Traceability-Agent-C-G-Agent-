import uuid
import json
from datetime import datetime
from typing import Dict, Any, List
from src.traceability.graph_config import GraphDBClient
from logger import logging


import uuid
import json
from datetime import datetime
from typing import Dict, Any, List
from src.traceability.graph_config import GraphDBClient
from logger import logging

class TraceabilityAgentService:
    def __init__(self, graph_client: GraphDBClient):
        self.graph_client = graph_client

    def synthesize_and_ingest_event(self, raw_event_data: Dict[str, Any]):
        """
        SIMPLIFIED: Generate graph from events - works with mock client
        """
        try:
            event_id = raw_event_data['event_id']
            timestamp = raw_event_data.get('created_at', datetime.utcnow().isoformat()) 
            txn_id = raw_event_data.get('client_transaction_id', 'default_txn')
            agent_name = raw_event_data['agent_name']
            event_type = raw_event_data['event_type']
            extra = raw_event_data.get('extra', {})
            
            # SIMPLE Cypher queries that mock can parse
            cypher_parts = []
            
            # 1. Core structure - always create these
            cypher_parts.append(f"""
            MERGE (a:Agent {{name: '{agent_name}'}})
            MERGE (t:Transaction {{id: '{txn_id}'}})
            CREATE (e:Event {{id: '{event_id}', type: '{event_type}', timestamp: '{timestamp}'}})
            CREATE (a)-[:PERFORMED]->(e)
            CREATE (e)-[:PART_OF]->(t)
            """)
            
            # 2. Event-specific nodes - SIMPLIFIED for mock
            if event_type == 'DATA_HASH_SAVED':
                train_hash = extra.get('train_hash') or extra.get('train_val_hash')
                if train_hash:
                    cypher_parts.append(f"""
                    MERGE (ds:Dataset {{hash: '{train_hash}'}})
                    SET ds.type = 'ProcessedData'
                    CREATE (e)-[:CREATED]->(ds)
                    """)
                    
            elif event_type == 'POLICY_METADATA_SAVED':
                policy_id = extra.get('policy_id')
                if policy_id:
                    cypher_parts.append(f"""
                    MERGE (p:Policy {{id: '{policy_id}'}})
                    SET p.name = 'DataPreprocessPolicy'
                    CREATE (e)-[:DEFINED]->(p)
                    """)
                    
            elif event_type == 'calculating_confidence_learning':
                evidence_id = extra.get('evidence_id')
                if evidence_id:
                    cypher_parts.append(f"""
                    MERGE (ev:Evidence {{id: '{evidence_id}'}})
                    SET ev.type = 'ConfidenceReport'
                    CREATE (e)-[:GENERATED]->(ev)
                    """)
                    
            elif event_type == 'MODEL_TRAINING_COMPLETED':
                model_evidence_id = extra.get('model_evidence_id')
                if model_evidence_id:
                    cypher_parts.append(f"""
                    MERGE (m:Model {{id: '{model_evidence_id}'}})
                    SET m.type = 'TrainedModel'
                    CREATE (e)-[:TRAINED]->(m)
                    """)
                    
            elif event_type == 'XAI_EXPLANATIONS_GENERATED':
                xai_evidence_id = extra.get('xai_evidence_id')
                if xai_evidence_id:
                    cypher_parts.append(f"""
                    MERGE (xai:XAI {{id: '{xai_evidence_id}'}})
                    SET xai.type = 'Explanation'
                    CREATE (e)-[:EXPLAINED]->(xai)
                    """)
                    
            elif event_type == 'PREDICTION_EXECUTED':
                prediction_evidence_id = extra.get('prediction_evidence_id')
                if prediction_evidence_id:
                    cypher_parts.append(f"""
                    MERGE (pred:Prediction {{id: '{prediction_evidence_id}'}})
                    SET pred.type = 'PredictionResult'
                    CREATE (e)-[:PREDICTED]->(pred)
                    """)
                    
            elif event_type == 'POLICY_EVALUATION_COMPLETED':
                governance_evidence_id = extra.get('governance_evidence_id')
                if governance_evidence_id:
                    cypher_parts.append(f"""
                    MERGE (gov:Governance {{id: '{governance_evidence_id}'}})
                    SET gov.type = 'PolicyDecision'
                    CREATE (e)-[:EVALUATED]->(gov)
                    """)
                    
            elif event_type == 'ERROR':
                summary = raw_event_data.get('summary', 'Unknown error')[:100]
                cypher_parts.append(f"""
                CREATE (err:Error {{summary: '{summary}'}})
                CREATE (e)-[:CAUSED]->(err)
                """)
            
            # Execute the query
            full_cypher = "\n".join(cypher_parts)
            result = self.graph_client.execute_cypher(full_cypher, {"event_data": raw_event_data})
            return result
            
        except Exception as e:
            logging.error(f"Event synthesis failed: {e}")
            return {"status": "error", "error": str(e)}

    def synthesize_and_ingest_artifact(self, table_name: str, artifact_data: Dict[str, Any]):
        """SIMPLIFIED: Create artifact nodes"""
        try:
            if table_name == "dataset_version":
                dataset_id = artifact_data.get('dataset_id')
                dataset_version = artifact_data.get('dataset_version', 'unknown')
                cypher_query = f"""
                MERGE (ds:Dataset {{
                    id: '{dataset_id}',
                    version: '{dataset_version}',
                    type: 'DatasetVersion'
                }})
                """
                
            elif table_name == "policy_document":
                policy_id = artifact_data.get('policy_id')
                policy_name = artifact_data.get('policy_name', 'unknown')
                cypher_query = f"""
                MERGE (p:Policy {{
                    id: '{policy_id}',
                    name: '{policy_name}',
                    type: 'PolicyDocument'
                }})
                """
                
            elif table_name == "evidence_pointer":
                evidence_id = artifact_data.get('evidence_id')
                kind = artifact_data.get('kind', 'unknown')
                
                if kind == 'TRAINED_MODEL':
                    cypher_query = f"""
                    MERGE (m:Model {{
                        id: '{evidence_id}',
                        kind: '{kind}',
                        type: 'TrainedModel'
                    }})
                    """
                elif kind in ['XAI_EXPLANATION_REPORT', 'PREDICTION_XAI_EXPLANATION']:
                    cypher_query = f"""
                    MERGE (xai:XAI {{
                        id: '{evidence_id}',
                        kind: '{kind}',
                        type: 'XAIExplanation'
                    }})
                    """
                elif kind == 'XAI_VISUALIZATION':
                    cypher_query = f"""
                    MERGE (viz:Visualization {{
                        id: '{evidence_id}',
                        kind: '{kind}',
                        type: 'XAIVisualization'
                    }})
                    """
                elif kind == 'PREDICTION_RESULT':
                    cypher_query = f"""
                    MERGE (pred:Prediction {{
                        id: '{evidence_id}',
                        kind: '{kind}',
                        type: 'PredictionResult'
                    }})
                    """
                elif kind == 'GOVERNANCE_DECISION':
                    cypher_query = f"""
                    MERGE (gov:Governance {{
                        id: '{evidence_id}',
                        kind: '{kind}',
                        type: 'GovernanceDecision'
                    }})
                    """
                else:
                    cypher_query = f"""
                    MERGE (ev:Evidence {{
                        id: '{evidence_id}',
                        kind: '{kind}',
                        type: 'Evidence'
                    }})
                    """
            else:
                return {"status": "skipped", "reason": f"Unknown table: {table_name}"}
            
            result = self.graph_client.execute_cypher(cypher_query, {"artifact_data": artifact_data})
            return result
            
        except Exception as e:
            logging.error(f"Artifact synthesis failed for {table_name}: {e}")
            return {"status": "error", "error": str(e)}

    def synthesize_and_ingest_event_links(self, event_link_data: Dict[str, Any]):
        """SIMPLIFIED: Create relationships from event links"""
        try:
            event_id = event_link_data.get('event_id')
            cypher_parts = []
            
            # Link to policy
            if event_link_data.get('policy_id'):
                policy_id = event_link_data['policy_id']
                cypher_parts.append(f"""
                MATCH (e:Event {{id: '{event_id}'}})
                MATCH (p:Policy {{id: '{policy_id}'}})
                CREATE (e)-[:USES_POLICY]->(p)
                """)
                
            # Link to dataset  
            if event_link_data.get('dataset_id'):
                dataset_id = event_link_data['dataset_id']
                cypher_parts.append(f"""
                MATCH (e:Event {{id: '{event_id}'}})
                MATCH (ds:Dataset {{id: '{dataset_id}'}})
                CREATE (e)-[:USES_DATA]->(ds)
                """)
                
            # Link to evidence
            if event_link_data.get('evidence_id'):
                evidence_id = event_link_data['evidence_id']
                cypher_parts.append(f"""
                MATCH (e:Event {{id: '{event_id}'}})
                MATCH (ev:Evidence {{id: '{evidence_id}'}})
                CREATE (e)-[:HAS_EVIDENCE]->(ev)
                """)
            
            if cypher_parts:
                full_cypher = "\n".join(cypher_parts)
                result = self.graph_client.execute_cypher(full_cypher, {"event_link_data": event_link_data})
                return result
            else:
                return {"status": "skipped", "reason": "No links to create"}
                
        except Exception as e:
            logging.error(f"Event link synthesis failed: {e}")
            return {"status": "error", "error": str(e)}

    def query_audit_trail(self, cypher_query: str) -> Dict[str, Any]:
        """
        Execute query against the graph DB
        """
        result_set = self.graph_client.execute_cypher(cypher_query, {"read_only": True})
        
        if not result_set or result_set.get('status') == 'error':
            return {"error": "Query failed", "status": "error"}

        return {
            "status": "success",
            "query_executed": cypher_query[:100] + "...",
            "raw_result": result_set,
            "mock": True
        }