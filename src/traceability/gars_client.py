from logger import logging
from exception import CGAgentException
from typing import Dict,List,Any
import re

#mock GRAS client connection

# FIXED gars_client.py
from logger import logging
from exception import CGAgentException
from typing import Dict, List, Any
import re

class MockGraphDBClient:
    """
    Mock GRAS client that simulates graph database operations.
    FIXED: Better regex patterns and property parsing
    """
    def __init__(self) -> Dict[str, Any]:
        """Initialize mock graph data structure"""
        self.nodes = {}
        self.edges = []
        self.queries_executed = []
        logging.info("Mock Graph DB initialized")

    def execute_cypher(self, query: str, params: dict[str, Any] = None) -> Dict[str, Any]:
        """
        Mock Cypher query execution - parses and stores what WOULD be created
        """
        self.queries_executed.append({"query": query, "params": params})

        try:
            query_upper = query.upper()
            if "CREATE" in query_upper:
                self.handle_create_query(query)
            elif "MERGE" in query_upper:
                self.handle_merge_query(query)
            elif "MATCH" in query_upper:
                return self.handle_match_query(query, params)
            
            logging.info(f"Mock GRAS executed: {self._get_query_summary(query)}")
            return {
                "status": "success",
                "nodes_created": len([q for q in self.queries_executed if "CREATE" in q["query"]]),
                "edges_created": len(self.edges),
                "mock": True
            }
        except Exception as e:
            logging.error(f"Mock GRAS query failed: {e}")
            return {"status": "error", "error": str(e), "mock": True}

    def handle_create_query(self, query: str):
        """FIXED: Better CREATE query handling"""
        # Improved node pattern
        node_pattern = r'\((?:(\w+):)?(\w+)\s*({[^}]+})\)'
        node_matches = re.findall(node_pattern, query)

        for var_name, label, properties in node_matches:
            props = self._parse_properties_fixed(properties)
            node_id = self._generate_node_id(label, props, var_name)
            
            self.nodes[node_id] = {
                "id": node_id,
                "label": label,
                "properties": props,
                "variable": var_name or f"{label.lower()}_{len(self.nodes)}"
            }
            logging.info(f"Created node: {label}({node_id})")
        
        # Improved relationship pattern
        rel_pattern = r'\(([^)]+)\)\s*(?:-\[([^\]]*)\])?\s*->\s*\(([^)]+)\)'
        rel_matches = re.findall(rel_pattern, query)
        
        for from_node, rel_type, to_node in rel_matches:
            from_id = self._extract_node_id_fixed(from_node)
            to_id = self._extract_node_id_fixed(to_node)
            rel_type_clean = rel_type.strip('[]:') or "RELATES_TO"
            
            if from_id and to_id:
                edge_id = f"{from_id}-{rel_type_clean}->{to_id}-{len(self.edges)}"
                self.edges.append({
                    "id": edge_id,
                    "from": from_id,
                    "to": to_id,
                    "type": rel_type_clean
                })
                logging.info(f"Created edge: {from_id} -[{rel_type_clean}]-> {to_id}")

    def handle_merge_query(self, query: str):
        """Handle MERGE queries - create or get existing"""
        # For mock, treat MERGE same as CREATE but don't duplicate nodes
        node_pattern = r'\((?:(\w+):)?(\w+)\s*({[^}]+})\)'
        node_matches = re.findall(node_pattern, query)

        for var_name, label, properties in node_matches:
            props = self._parse_properties_fixed(properties)
            node_id = self._generate_node_id(label, props, var_name)
            
            # Only create if doesn't exist
            if node_id not in self.nodes:
                self.nodes[node_id] = {
                    "id": node_id,
                    "label": label,
                    "properties": props,
                    "variable": var_name or f"{label.lower()}_{len(self.nodes)}"
                }
                logging.info(f"Merged node: {label}({node_id})")

    def handle_match_query(self, query: str, params: Dict[str, Any]):
        """Handle MATCH queries - return mock results"""
        if "EVENT" in query.upper() and "type" in query.upper():
            event_type = params.get('type') if params else None
            matching_nodes = [node for node in self.nodes.values()
                            if node.get('label') == 'Event' and node['properties'].get('type') == event_type]
            return {"status": "success", "data": matching_nodes, "mock": True}
        
        return {"status": "success", "data": [], "mock": True}

    def _parse_properties_fixed(self, properties_str: str) -> Dict[str, Any]:
        """FIXED: Better property parsing"""
        try:
            properties = {}
            content = properties_str.strip('{}')
            
            # Simple split by comma - handle basic cases
            parts = [part.strip() for part in content.split(',') if part.strip()]
            
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes but be careful
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    
                    properties[key] = value
            
            return properties
        except Exception as e:
            logging.warning(f"Property parsing simplified: {e}")
            return {"raw": properties_str[:50]}

    def _generate_node_id(self, label: str, props: Dict, var_name: str) -> str:
        """Generate consistent node ID"""
        if props.get('id'):
            return props['id']
        elif props.get('name'):
            return f"{label}_{props['name']}"
        elif var_name:
            return f"{label}_{var_name}"
        else:
            return f"{label}_{len(self.nodes)}"

    def _extract_node_id_fixed(self, node_str: str) -> str:
        """FIXED: Better node ID extraction"""
        # Try to extract from properties first
        id_match = re.search(r"id:\s*'([^']+)'", node_str)
        if id_match:
            return id_match.group(1)
        
        name_match = re.search(r"name:\s*'([^']+)'", node_str)
        if name_match:
            return f"Agent_{name_match.group(1)}"
        
        # Try variable reference
        var_match = re.search(r'\((\w+)\)', node_str)
        if var_match:
            var_name = var_match.group(1)
            # Find existing node with this variable
            for node_id, node_data in self.nodes.items():
                if node_data.get('variable') == var_name:
                    return node_id
        
        return None

    def _get_query_summary(self, query: str) -> str:
        """Get short summary of query for logging"""
        if "CREATE" in query:
            return f"CREATE operation ({len(self.nodes)} total nodes, {len(self.edges)} total edges)"
        elif "MERGE" in query:
            return f"MERGE operation"
        elif "MATCH" in query:
            return f"MATCH query"
        else:
            return f"Unknown operation: {query[:50]}..."

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get current graph statistics"""
        node_counts = {}
        for node in self.nodes.values():
            label = node['label']
            node_counts[label] = node_counts.get(label, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": node_counts,
            "queries_executed": len(self.queries_executed)
        }

    def visualize_graph(self):
        """Print simple graph visualization"""
        print("\n" + "="*60)
        print("MOCK GRAPH VISUALIZATION")
        print("="*60)
        
        for node_id, node_data in self.nodes.items():
            print(f"{node_data['label']}: {node_id}")
            if 'properties' in node_data:
                for key, value in list(node_data['properties'].items())[:3]:
                    print(f"  {key}: {value}")
        
        print("\nRELATIONSHIPS:")
        for edge in self.edges:
            print(f"  {edge['from']} -[{edge['type']}]-> {edge['to']}")
        
        stats = self.get_graph_stats()
        print(f"\nSTATS: {stats['total_nodes']} nodes, {stats['total_edges']} edges, {stats['queries_executed']} queries")
        print("="*60)