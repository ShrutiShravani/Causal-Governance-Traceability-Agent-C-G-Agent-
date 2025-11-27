import yaml
from typing import Dict,Any
from src.governance_layer.opa_client import OPAClient
from exception import CGAgentException
from logger import logging
import os,sys

class PolicyEngine:
    """
    Centralized policy engine for Critic Agent.
    Loads policies from registry, queries OPA, and aggregates results.
    """

    def __init__(self,registry_path:str =r"src\governance_layer\policy_regsitry.yaml", opa_url: str ="http://localhost:8181"):
        self.registry_path = registry_path
        self.opa_client = OPAClient(opa_url=opa_url)
        self.policies = self._load_policy_registry()
        logging.info(f"PolicyEngine initialized with {len(self.policies)} policies.")

    def _load_policy_registry(self)->Dict[str,Any]:
        """Loads the active policy registry YAML."""
        try:
            if not os.path.exists(self.registry_path):
                raise FileNotFoundError(f"File not found at {self.registry_path}")
            with open(self.registry_path,"r",encoding="utf-8") as f:
                registry= yaml.safe_load(f)
            logging.info(f"Loaded policy registry from {self.registry_path}")
            return registry.get("policies",{})
        except Exception as e:
            raise CGAgentException(e,sys)
        except yaml.YAMLError as e:
            logging.info(f"Error parsing YAML Registry :{e}")
            return {}
    
    def evaluate(self,input_payload:Dict[str,Any])->Dict[str,Any]:
        """
        Evaluate all active policies against the input payload.
        Returns a dictionary of OPA results keyed by policy name.
        """
        results={}
        try:
            for policy_name,policy_meta in self.policies.items():
                policy_path= policy_meta.get("rego_path")
                if not policy_path:
                    logging.info(f"No rego path defined for policy {policy_path}")
                    continue
                logging.info(f"Evaluating poicy {policy_name} via OPA at {policy_path}")
                policy_results=self.opa_client.query_policy(policy_path,input_payload)
                results[policy_name]=policy_results
            print(results)
            logging.info(f"Policy evaluation completed. {len(results)} policies evaluated.")
            return results
        except Exception as e:
            logging.error(f"Policy evaluation failed: {e}")
            raise CGAgentException(e,sys)

    

