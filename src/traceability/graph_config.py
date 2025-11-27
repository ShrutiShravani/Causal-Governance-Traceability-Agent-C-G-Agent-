from  typing import Dict,Any
import logging
import os
from exception import CGAgentException
import os,sys
import re
from src.traceability.gars_client import MockGraphDBClient

"""
MOCK_DB_STATE = {
    # This state represents the data saved by the TraceabilityAgentService
    "TXN-101": {
        'd': {'result': 'APPROVED', 'confidence_status': 'human_review_required', 'timestamp': '2025-11-14T10:00:00'}, 
        'violations':"no_violation", 
        'policies': ['confidence_threshold', 'prohibited_attributes'],
        'ds.hash': 'SHA256-A4B7C9D2',
        'art.id': 'UUID-ESC-4321', 
        'art.kind': 'HUMAN_REVIEW_REPORT'
    },
    "TXN-102": {
        'd': {'result': 'BLOCKED', 'confidence_status': 'pass', 'timestamp': '2025-11-14T10:05:00'},
        'violations':"bais results based on prohibited attributes",
        'policies': ['confidence_threshold', 'prohibited_attributes'],
        #... (other successful provenance data)
    }
}
"""
class GraphDBClient:
    """
    Conceptual client for Graph Database interaction.
    In a production setting, this would wrap the official Neo4j Driver (or similar).
    """
    def __init__(self,use_mock:bool=True):
        if use_mock:
            self.client= MockGraphDBClient()
            logging.info("Uing mock grah db client")
        else:
            self.client=MockGraphDBClient()
            logging.info("real graph db not configured using mock")
    
    def execute_cypher(self,query: str, params: Dict[str, Any] = None):
        """
        Simulated execution of a Cypher query. Returns structured data 
        that would be retrieved from the database.
        """
        try:
            return self.client.execute_cypher(query,params)        
        except Exception as e:
            raise CGAgentException(e,sys)
    
    def get_graph_stats(self):
        return self.client.get_graph_stats()
    
    def visualize_graph(self):
        return self.client.visualize_graph()
