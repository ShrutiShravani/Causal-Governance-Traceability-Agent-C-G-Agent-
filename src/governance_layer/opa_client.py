import requests
import json
from logger import logging
from typing import Dict,Any,Optional
from exception import CGAgentException
import os,sys

class OPAClient:
    """
    A lightweight client to interact with the Open Policy Agent (OPA) REST API.
    """

    def __init__(self,opa_url:str="http://localhost:8181"):
        """
        :param opa_url: Base URL for the OPA REST API (data endpoint)
        Example: http://localhost:8181/v1/data/policies/bias_check
        """
        self.opa_url=opa_url.rstrip("/")
        self.policy_url = f"{self.opa_url}/v1/policies"

    def query_policy(self,policy_path:str,input_data:Dict[str,Any])->Dict[str,Any]:
        """
        Send a query to OPA for a specific policy.

        :param policy_path: Path to the OPA policy (relative to data/)
        :param input_data: Input payload (e.g., model explanation)
        :return: Response dict from OPA (with "result")
        """
        url=f"{self.opa_url}/{policy_path.lstrip('/')}"

        logging.info(f"Querying OPA policy at: {url}")
       
        try:
            response= requests.post(url,json={"input":input_data},timeout=5)
            response.raise_for_status()
            result=response.json()
            logging.info(f"OPA response:{result}")
            return result
        except requests.exceptions.RequestException as e:
            logging.info(f"OPA request failed for {policy_path}: {e}")
            raise CGAgentException(f"OPA request failed for {policy_path}: {e}", sys)
        except json.JSONDecodeError:
            logging.info(f"Invalid JSON from OPA for {policy_path}")
            raise CGAgentException(f"Invalid JSON from OPA for {policy_path}", sys)
    
    def load_policy(self,policy_file_path:str,policy_id:str=None):
        # Read the policy file
        try:
            if policy_id is None:
                policy_id=os.path.splitext(os.path.basename(policy_file_path))[0]
            
            with open(policy_file_path,'r',encoding='utf-8') as f:
                policy_content = f.read()
                
                # OPA API endpoint for policies
                url = f"{self.policy_url}/{policy_id}"
                headers = {"Content-Type": "text/plain"}
                response= requests.put(url,data=policy_content.encode("utf-8"),headers=headers,timeout=10)
                response.raise_for_status()

                logging.info(f"Policy '{policy_id}' loaded successfully into OPA")
                return True
                
        except FileNotFoundError:
            logging.error(f"Policy file not found: {policy_file_path}")
            return False
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to load policy '{policy_id}': {e}")
            return False

    def list_policies(self)->Dict[str,Any]:
        try:
          
            response = requests.get(self.policy_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise CGAgentException(e,sys)