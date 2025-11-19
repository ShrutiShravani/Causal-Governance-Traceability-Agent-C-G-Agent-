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

    def __init__(self,opa_url:str="http://localhost:8181/v1/data"):
        """
        :param opa_url: Base URL for the OPA REST API (data endpoint)
        Example: http://localhost:8181/v1/data/policies/bias_check
        """
        self.opa_url=opa_url.rstrip("/")

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
