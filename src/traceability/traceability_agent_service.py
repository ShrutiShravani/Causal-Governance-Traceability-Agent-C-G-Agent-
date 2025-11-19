import uuid
import json
from datetime import datetime
from typing import Dict, Any, List
from src.traceability.graph_config import GraphDBClient

class TraceabilityAgentService:
    def __init__(self, graph_client: GraphDBClient):
        self.graph_client = graph_client

    def synthesize_and_ingest_event(self,raw_event_data: Dict[str, Any]):
        """
        Synthesizes a raw log event into Nodes and Edges
        and ingests them into the Temporal Knowledge Graph, building the
        causal audit chain incrementally.
        """

        # 1. CORE EXTRACTION
        event_id = raw_event_data['event_id']
        timestamp = raw_event_data.get('created_at', datetime.utcnow().isoformat()) 
        txn_id = raw_event_data.get('client_transaction_id')
        agent_name = raw_event_data['agent_name']
        event_type = raw_event_data['event_type']
        extra = raw_event_data.get('extra', {})
        
        # Determine the artifact ID for generic evidence linking (used in logs 1, 7)
        artifact_id = extra.get('evidence_id') or extra.get('artifact_id')

        # --- 2. CORE GRAPH CREATION (Foundation) ---
        # A. Always create the unique Event Node and link to Agent and Transaction Anchor
        cypher_query = f"""
        // 1. Ensure Agent and Transaction Anchor Nodes exist
        MERGE (a:Agent {{name: '{agent_name}'}})
        MERGE (t:Transaction {{id: '{txn_id}'}})
        
        // 2. Create the unique Event Node with temporal property
        CREATE (e:Event {{id: '{event_id}', type: '{event_type}', timestamp: '{timestamp}'}})
        
        // 3. Create foundational Edges (Relationships)
        CREATE (a)-->(e)
        CREATE (e)-->(t)
        """

        # --- 3. CONDITIONAL SYNTHESIS (Relational Facts) ---
        
        # B1. Synthesis: Data Evidence Link (Example 1: calculating_confidence_learning)
        if event_type == 'calculating_confidence_learning':
            # This links the confidence event to the confidence report artifact
            if artifact_id:
                cypher_query += f"""
                MERGE (art:Artifact {{id: '{artifact_id}'}})
                CREATE (e)-->(art) 
                """

        # B2. Synthesis: Data Version Provenance (Example 2: DATA_HASH_SAVED)
        elif event_type == 'DATA_HASH_SAVED':
            train_hash = extra.get('train_hash') 
            if train_hash:
                # Link this event to the certified immutable DatasetVersion Node
                cypher_query += f"""
                MERGE (ds:DatasetVersion {{hash: '{train_hash}'}})
                CREATE (e)-->(ds)
                """

        # B3. Synthesis: Policy Document Link (Example 3: POLICY_METADATA_SAVED)
        elif event_type == 'POLICY_METADATA_SAVED':
            policy_id = extra.get('policy_id') 
            if policy_id:
                # Links the event that defined the policy to the PolicyDocument Node
                cypher_query += f"""
                MERGE (pd:PolicyDocument {{id: '{policy_id}'}})
                CREATE (e)-->(pd)
                """

        # B4. Synthesis: Prediction Outcome (Example 4: PREDICTION_EXECUTED)
        elif event_type == 'PREDICTION_EXECUTED':
            # Creates a Prediction Metrics Node (holds quantitative outcome)
            cypher_query += f"""
            CREATE (m:PredictionMetrics {{
                label: {extra.get('prediction_label', 'null')},
                probability: {extra.get('prediction_probability', 0.0)},
                model_version: '{extra.get('model_version', 'unknown')}'
            }})
            CREATE (e)-->(m)
            """

        # B5. Synthesis: XAI Explanation Link (Example 5: LOCAL_EXPLANATION_GENERATED)
        elif event_type == 'LOCAL_EXPLANATION_GENERATED':
            # Creates a dedicated Explanation Artifact Node with key XAI features
            cypher_query += f"""
            CREATE (xai:ExplanationArtifact {{
                top_drivers_count: size({json.dumps(extra.get('top_drivers',))}),
                expected_value: {extra.get('expected_value', 0.0)}
            }})
            CREATE (e)-->(xai)
            """

        # B6. Synthesis: Policy Enforcement & Final Decision (Example 6: POLICY_ENFORCED)
        elif event_type == 'POLICY_ENFORCED':
            final_decision = extra.get('final_decision')
            policy_results_summary = extra.get('policy_results_summary', {})
            violations_list: List[str] = extra.get('violations',)
            
            # 1. Anchor the Final Decision Node (MERGE ensures it's updated if policy check reruns)
            cypher_query += f"""
            MERGE (d:FinalDecision {{transaction_id: '{txn_id}'}})
            SET d.result = '{final_decision}', d.timestamp = '{timestamp}',
                d.confidence_status = '{extra.get('confidence_status', 'unknown')}'
            CREATE (e)-->(d)
            """
            
            # 2. Link Decision to specific Policy Check Outcomes (e.g., ProhibitedAttributesCheck)
            for policy_name in policy_results_summary.keys():
                cypher_query += f"""
                MERGE (pc:PolicyCheck {{name: '{policy_name}'}})
                CREATE (d)-->(pc)
                """
            
            # 3. Link Decision to specific Violations (The causal link)
            for violation_name in violations_list:
                cypher_query += f"""
                MERGE (v:Violation {{attribute: '{violation_name}'}})
                CREATE (d)-->(v)
                """
            
        # B7. Synthesis: Human Escalation (Example 7: HUMAN_ESCALATION)
        elif event_type == 'HUMAN_ESCALATION':
            escalation_id = extra.get('escalation_id')
            confidence = extra.get('confidence')
            
            # 1. Create Human Oversight Node (The mandatory compliance step)
            cypher_query += f"""
            MERGE (h:HumanOversight {{id: '{escalation_id}'}})
            SET h.action = '{extra.get('action', 'Human review required')}', 
                h.confidence_score = {confidence}
            CREATE (e)-->(h) 
            """
            
            # 2. Link Human Oversight to the Final Decision (The established causal link)
            # Find the existing FinalDecision Node anchored to the transaction
            cypher_query += f"""
            MATCH (d:FinalDecision {{transaction_id: '{txn_id}'}})
            CREATE (h)-->(d)
            """

        # B8. Synthesis: General Error (Example 8: ERROR)
        elif event_type == 'ERROR':
            # Create a dedicated Failure Node and link the event to it
            cypher_query += f"""
            CREATE (f:OperationalFailure {{summary: '{raw_event_data.get('summary', 'Unknown')}'}})
            CREATE (e)-->(f)
            """
            
        # --- 4. GENERIC ARTIFACT/EVIDENCE LINK (Links any report/file pointer) ---
        # This handles linking the event to any report or file pointer (Artifacts from register_artifact)
        if artifact_id:
            # We explicitly check the type of event to ensure we create the correct edge relationship
            cypher_query += f"""
            MATCH (e:Event {{id: '{event_id}'}})
            MERGE (art:Artifact {{id: '{artifact_id}'}})
            CREATE (e)-->(art) 
            """
            
        # 5. GRAPH INGESTION (Final Step)
        self.graph_client.execute_cypher(cypher_query, raw_event_data)

    def query_audit_trail(self, cypher_query: str) -> Dict[str, Any]:
        """
        Phase 6: Executes a complex multi-hop query against the Graph DB.
        This method retrieves the structured subgraph (the audit story).
        """
        """
        Phase 6: Executes a complex multi-hop query against the Graph DB.
        Processes the raw result set into a canonical dictionary of verifiable facts.
        """
        
        result_set = self.graph_client.execute_cypher(cypher_query, {"read_only": True})
        
        if not result_set or result_set.get('status') == 'unknown':
            return {}

        # --- CANONICAL DATA MAPPING ---
        
        canonical_facts = {}
        
        # A. Causal Chain Query Mapping (FinalDecision)
        if 'd' in result_set: 
            final_decision_node = result_set.get('d', {})
            
            # 1. Final Decision Facts (Directly from FinalDecision Node properties)
            canonical_facts['final_result'] = final_decision_node.get('result')
            canonical_facts['confidence_status'] = final_decision_node.get('confidence_status')
            
            # 2. Causal Facts (Aggregated lists)
            canonical_facts['causal_violations'] = result_set.get('violations', )
            canonical_facts['policies_enforced'] = result_set.get('policies', )
            
        # B. Provenance Query Mapping (DatasetVersion)
        elif 'ds.hash' in result_set:
            canonical_facts['data_provenance_hash'] = result_set.get('ds.hash')

        # C. Evidence/Report Query Mapping (Artifact)
        elif 'art' in result_set:
            artifact_node = result_set.get('art', {})
            canonical_facts['evidence_artifact_id'] = artifact_node.get('id')
            canonical_facts['evidence_kind'] = artifact_node.get('kind')
            
        
        # 3. RETURN CANONICAL FACTS
        return canonical_facts
        
        
    