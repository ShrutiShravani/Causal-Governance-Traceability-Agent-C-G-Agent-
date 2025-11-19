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

    # ---------------------------------------------------------------
    # 1. NATURAL LANGUAGE → CYPHER TRANSLATION (Brain of the Agent)
    # ---------------------------------------------------------------
    def _translate_to_cypher(self, user_query: str, txn_id: str) -> Optional[str]:
        """
        Converts natural language audit questions into multi-hop Cypher queries.
        This is where the NL → Graph reasoning happens.
        """
        user_query_lower = user_query.lower()

        # WHY / ROOT-CAUSE QUERY
        if "why was" in user_query_lower or "what caused" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<--(d:FinalDecision)
            OPTIONAL MATCH (d)--(v:Violation)
            OPTIONAL MATCH (d)--(pc:PolicyCheck)
            OPTIONAL MATCH (d)<--(e:Event {{type: 'POLICY_ENFORCED'}})<--(xai:ExplanationArtifact)
            RETURN d, collect(DISTINCT v.attribute) AS violations,
                   collect(DISTINCT pc.name) AS policies,
                   xai
            """

        # DATA PROVENANCE HASH QUERY
        elif "show me data hash" in user_query_lower or "provenance" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<--(e:Event {{type: 'DATA_HASH_SAVED'}})
            MATCH (e)-->(ds:DatasetVersion)
            RETURN ds.hash
            """

        # ARTIFACT / HUMAN REVIEW REPORT QUERY
        elif "show me the report" in user_query_lower or "evidence" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<--(d:FinalDecision)<--(h:HumanOversight)
            MATCH (h)<--(e:Event {{type: 'HUMAN_ESCALATION'}})
            OPTIONAL MATCH (e)-->(art:Artifact)
            RETURN h.id AS oversight_id, h.action AS oversight_action, art
            """

        # PREDICTION SCORE / CONFIDENCE QUERY
        elif "confidence" in user_query_lower or "score" in user_query_lower:
            return f"""
            MATCH (t:Transaction {{id: '{txn_id}'}})<--(e:Event {{type: 'PREDICTION_EXECUTED'}})
            MATCH (e)-->(m:PredictionMetrics)
            RETURN m.label, m.probability, m.model_version
            """

        # FALLBACK
        return None

    # ---------------------------------------------------------------
    # 2. MAIN PIPELINE: Process Query → Cypher → Structured Data → Text Response
    # ---------------------------------------------------------------
    def process_query(self, user_query: str, txn_id: str) -> str:
        logging.info(f"User Query received for {txn_id}: '{user_query}'")

        # Translate NL → Cypher
        cypher_query = self._translate_to_cypher(user_query, txn_id)

        if not cypher_query:
            return "I could not generate an audit query for that request. Could you rephrase?"

        # Query graph via TraceabilityAgentService
        structured_results = self.service.query_audit_trail(cypher_query)

        if not structured_results:
            return f"No audit records were found for Transaction ID {txn_id}."

        # ---------------------------------------------------------------
        # 3. CONVERSATIONAL RESPONSE GENERATION (Structured → Natural Language)
        # ---------------------------------------------------------------
        lower_q = user_query.lower()
        response = ""

        # ROOT-CAUSE / WHY QUERY
        if "why was" in lower_q or "what caused" in lower_q:
            final_result = structured_results.get('final_result', 'Unknown')
            confidence_status = structured_results.get('confidence_status', 'Unknown')

            violations = structured_results.get('causal_violations', [])
            policies = structured_results.get('policies_enforced', [])

            response = (
                f"The final outcome for Transaction **{txn_id}** was **{final_result}** "
                f"(Confidence Status: {confidence_status}).\n"
            )

            if final_result == "BLOCKED":
                formatted_violations = ", ".join(violations) if violations else "None"
                formatted_policies = ", ".join(policies) if policies else "None"
                response += (
                    f"The decision was blocked due to violations related to: **{formatted_violations}**.\n"
                    f"Policies involved: **{formatted_policies}**."
                )

            elif final_result == "APPROVED":
                formatted_policies = ", ".join(policies) if policies else "general model rules"
                response += f"The system approved the request. Policy checks referenced: **{formatted_policies}**."

        # DATA PROVENANCE HASH QUERY
        elif "show me data hash" in lower_q or "provenance" in lower_q:
            data_hash = structured_results.get('data_provenance_hash', 'Unknown')
            response = (
                f"The certified dataset fingerprint (SHA-256) used for model training is:\n"
                f"**{data_hash}**"
            )

        # ARTIFACT / REPORT QUERY
        elif "show me the report" in lower_q or "evidence" in lower_q:
            artifact_id = structured_results.get('evidence_artifact_id', 'Unknown')
            response = (
                f"A human oversight step was triggered for Transaction **{txn_id}**.\n"
                f"The corresponding review or evidence artifact is stored under ID: **{artifact_id}**."
            )

        # PREDICTION SCORE QUERY
        elif "confidence" in lower_q or "score" in lower_q:
            label = structured_results.get("label", "Unknown")
            prob = structured_results.get("probability", "Unknown")
            ver = structured_results.get("model_version", "Unknown")

            response = (
                f"The model predicted label **{label}** with probability **{prob}**.\n"
                f"Model version used: **{ver}**."
            )

        # Save conversation history
        self.conversation_history.append(
            {"user": user_query, "response": response, "data": structured_results}
        )

        return response + "\n\nHow else may I assist with this audit?"
