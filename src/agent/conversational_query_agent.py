from typing import Dict, Any, List, Optional
from src.traceability.traceability_agent_service import TraceabilityAgentService
from logger import logging


from typing import Dict, Any, List, Optional
from src.traceability.traceability_agent_service import TraceabilityAgentService
from logger import logging


class QueryAgent:
    """
    Conversational Query Agent:
    - Accepts natural language audit questions from user
    - Converts to Cypher graph queries
    - Queries graph through TraceabilityAgentService
    - Converts structured results into human-readable conversation responses
    """

    def __init__(self, service: TraceabilityAgentService):
        self.service = service
        self.conversation_history: List[Dict[str, Any]] = []

    # 1. NATURAL LANGUAGE → CYPHER TRANSLATION 
    def _translate_to_cypher(self, user_query: str, txn_id: str) -> Optional[str]:
        """
        Converts natural language audit questions into multi-hop Cypher queries.
        FIXED: Uses correct node labels and relationship types from your graph
        """
        user_query_lower = user_query.lower()

        # WHY / ROOT-CAUSE QUERY
        if "why was" in user_query_lower or "what caused" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<-[:PART_OF]-(e:Event)
            OPTIONAL MATCH (e)-[:EVALUATED]->(gov:Governance)
            RETURN e.type as event_type, 
                   gov.final_decision as final_decision,
                   gov.confidence_status as confidence_status
            LIMIT 1
            """

        # DATA PROVENANCE HASH QUERY
        elif "show me data hash" in user_query_lower or "provenance" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<-[:PART_OF]-(e:Event {{type: 'DATA_HASH_SAVED'}})
            MATCH (e)-[:CREATED]->(ds:Dataset)
            RETURN ds.hash as data_hash, ds.type as data_type
            LIMIT 1
            """

        # EVIDENCE/REPORT QUERY
        elif "show me the report" in user_query_lower or "evidence" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<-[:PART_OF]-(e:Event)
            MATCH (e)-[:HAS_EVIDENCE]->(ev:Evidence)
            RETURN ev.id as evidence_id, ev.kind as evidence_kind, ev.type as evidence_type
            LIMIT 5
            """

        # PREDICTION SCORE QUERY
        elif "confidence" in user_query_lower or "score" in user_query_lower or "prediction" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<-[:PART_OF]-(e:Event {{type: 'PREDICTION_EXECUTED'}})
            MATCH (e)-[:PREDICTED]->(pred:Prediction)
            RETURN pred.id as prediction_id, pred.type as prediction_type
            LIMIT 1
            """

        # MODEL TRAINING LINEAGE QUERY
        elif "what data trained" in user_query_lower or "model lineage" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<-[:PART_OF]-(e:Event {{type: 'MODEL_TRAINING_COMPLETED'}})
            MATCH (e)-[:TRAINED]->(m:Model)
            RETURN m.id as model_id, m.type as model_type
            LIMIT 1
            """

        # ALL EVENTS FOR TRANSACTION
        elif "show all events" in user_query_lower or "timeline" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<-[:PART_OF]-(e:Event)
            RETURN e.id as event_id, e.type as event_type, e.timestamp as timestamp
            ORDER BY e.timestamp
            LIMIT 10
            """

        # AGENTS INVOLVED
        elif "which agents" in user_query_lower or "who processed" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<-[:PART_OF]-(e:Event)
            MATCH (a:Agent)-[:PERFORMED]->(e)
            RETURN DISTINCT a.name as agent_name, count(e) as event_count
            LIMIT 10
            """

        # FALLBACK - BASIC EVENT QUERY
        return f"""
        MATCH (t:Transaction {{id: '{txn_id}'}})<-[:PART_OF]-(e:Event)
        RETURN e.id as event_id, e.type as event_type, e.timestamp as timestamp
        ORDER BY e.timestamp
        LIMIT 5
        """

    # 2. MAIN PIPELINE: Process Query → Cypher → Structured Data → Text Response
    def process_query(self, user_query: str, txn_id: str) -> str:
        logging.info(f"User Query received for {txn_id}: '{user_query}'")

        # Translate NL → Cypher
        cypher_query = self._translate_to_cypher(user_query, txn_id)

        if not cypher_query:
            return "I could not generate an audit query for that request. Could you rephrase?"

        # Query graph via TraceabilityAgentService
        structured_results = self.service.query_audit_trail(cypher_query)

        if structured_results.get('status') == 'error':
            return f"Query failed: {structured_results.get('error', 'Unknown error')}"

        # FIXED: Handle mock graph response structure
        raw_result = structured_results.get('raw_result', {})
        
        # Mock graph returns data in different format
        if raw_result.get('mock'):
            # For mock, we need to process the actual graph data
            mock_data = raw_result.get('data', [])
            if not mock_data:
                return f"No audit records found for Transaction ID {txn_id}."
            
            # Use the first result for response generation
            result_data = mock_data[0] if isinstance(mock_data, list) and mock_data else {}
        else:
            # For real graph, use the structured results directly
            result_data = structured_results

        # 3. CONVERSATIONAL RESPONSE GENERATION
        lower_q = user_query.lower()
        response = ""

        try:
            # ROOT-CAUSE / WHY QUERY
            if "why was" in lower_q or "what caused" in lower_q:
                final_decision = result_data.get('final_decision', 'Unknown')
                confidence_status = result_data.get('confidence_status', 'Unknown')
                event_type = result_data.get('event_type', 'Unknown')
                
                response = (
                    f"**Transaction {txn_id} Analysis:**\n"
                    f"• **Event Type**: {event_type}\n"
                    f"• **Final Decision**: {final_decision}\n"
                    f"• **Confidence Status**: {confidence_status}\n\n"
                )
                
                if final_decision == "BLOCKED":
                    response += "The decision was blocked due to policy violations or low confidence."
                elif final_decision == "APPROVED":
                    response += "The system approved the request based on policy compliance."

            # DATA PROVENANCE HASH QUERY
            elif "show me data hash" in lower_q or "provenance" in lower_q:
                data_hash = result_data.get('data_hash', 'Not found')
                data_type = result_data.get('data_type', 'Unknown')
                response = f"**Data Provenance for {txn_id}:**\n• **Type**: {data_type}\n• **SHA-256**: {data_hash}"

            # EVIDENCE/REPORT QUERY
            elif "show me the report" in lower_q or "evidence" in lower_q:
                evidence_id = result_data.get('evidence_id', 'Not found')
                evidence_kind = result_data.get('evidence_kind', 'Unknown')
                evidence_type = result_data.get('evidence_type', 'Unknown')
                response = f"**Evidence Found:**\n• **ID**: {evidence_id}\n• **Kind**: {evidence_kind}\n• **Type**: {evidence_type}"

            # PREDICTION QUERY
            elif "confidence" in lower_q or "score" in lower_q or "prediction" in lower_q:
                prediction_id = result_data.get('prediction_id', 'Not found')
                prediction_type = result_data.get('prediction_type', 'Unknown')
                response = f"**Prediction Record:**\n• **ID**: {prediction_id}\n• **Type**: {prediction_type}"

            # MODEL LINEAGE QUERY
            elif "what data trained" in lower_q or "model lineage" in lower_q:
                model_id = result_data.get('model_id', 'Not found')
                model_type = result_data.get('model_type', 'Unknown')
                response = f"**Model Training Record:**\n• **Model ID**: {model_id}\n• **Type**: {model_type}"

            # ALL EVENTS QUERY
            elif "show all events" in lower_q or "timeline" in lower_q:
                if isinstance(mock_data, list):
                    events_text = "\n".join([
                        f"• {event.get('event_type', 'Unknown')} at {event.get('timestamp', 'Unknown')}"
                        for event in mock_data[:5]  # Show first 5 events
                    ])
                    response = f"**Event Timeline for {txn_id}:**\n{events_text}"
                else:
                    response = f"Found events for {txn_id}, but couldn't format timeline."

            # AGENTS QUERY
            elif "which agents" in lower_q or "who processed" in lower_q:
                if isinstance(mock_data, list):
                    agents_text = "\n".join([
                        f"• {agent.get('agent_name', 'Unknown')} ({agent.get('event_count', 0)} events)"
                        for agent in mock_data
                    ])
                    response = f"**Agents Involved in {txn_id}:**\n{agents_text}"
                else:
                    response = f"Found agent data for {txn_id}."

            # DEFAULT RESPONSE
            else:
                if isinstance(mock_data, list) and mock_data:
                    events_text = "\n".join([
                        f"• {event.get('event_type', 'Unknown')}"
                        for event in mock_data[:3]
                    ])
                    response = f"**Recent Events for {txn_id}:**\n{events_text}"
                else:
                    response = f"I found information for Transaction {txn_id}, but couldn't generate a detailed response."

        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            response = f"I encountered an error while processing your query: {str(e)}"

        # Save conversation history
        self.conversation_history.append({
            "user": user_query,
            "response": response,
            "txn_id": txn_id,
            "cypher_query": cypher_query
        })

        return response + "\n\nHow else may I assist with this audit?"

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []