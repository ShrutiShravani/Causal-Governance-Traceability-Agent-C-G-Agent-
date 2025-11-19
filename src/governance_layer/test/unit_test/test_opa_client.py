from unittest.mock import MagicMock,patch
import requests
import json
from logger import logging
from typing import Dict,Any,Optional
from exception import CGAgentException
import os,sys
from src.governance_layer.opa_client import OPAClient
import pytest
import requests


@pytest.fixture
def opa_client():
    return OPAClient(opa_url='http://localhost:8181/v1/data')

def test_query_policy_success(opa_client):
    """ OPA retruns valid JSON-> shoudl return same dictionary"""
    mock_response=MagicMock()
    mock_response.raise_for_status.return_value=None
    mock_response.json.return_value={"result":{"allow":True}}

    with patch("opa_client.requests.post",return_value=mock_response):
        result=opa_client.query_policy("policies/bias_check",{"x":1})
        assert result=={"result":{"allow":True}}


def test_query_policy_htpp_error(opa_client):
    """HTTP 4xx/5xx should raise request.RequestException->CGAgentException"""
    mock_response= MagicMock()
    mock_response.raise_for_status.side_effect=Exception("HTTP 400")

    with patch("opa_client.request.post",return_value=mock_response):
        with pytest.raises(CGAgentException):
            opa_client.query_polcy("policies/bias_chcek",{"x":1})


def test_query_polciy_invalid_json(opa_client):
    """invalid json returned->json.JSONDecodeError(simulated by value error)->CGAgentException"""
    mock_response= MagicMock()
    mock_response.raise_for_status.return_value=None
    mock_response.json.side_effect=ValueError("invalid json")
    
    with patch("opa_client.requests.post", return_value=mock_response):
        with pytest.raises(CGAgentException):
            opa_client.query_policy("policies/bias_check", {"x": 1})

def test_query_policy_network_error(opa_client):
    """Network issue->requests.exceptions->CGAgentException"""
    with patch("opa_client.requests.post",side_effect=Exception("network error")):
        with pytest.raises(CGAgentException):
            opa_client.query_policy("policies/bias_check", {"x": 1})


def test_query_polcy_timeout(opa_client):
    """timeout also raises Request exception->CGAgentException"""
    with patch("opa_client.requests.post",side_effet=Exception("timeout")):
        with pytest.raises(CGAgentException):
            opa_client.query_polcy("policies/bias_check", {"x": 1})