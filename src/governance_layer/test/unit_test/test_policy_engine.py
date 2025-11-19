import pytest
from unittest.mock import patch,MagicMock
from exception import CGAgentException
from src.governance_layer.policy_engine import PolicyEngine
import yaml

@pytest.fixture
def policy_engine(tmp_path):
    # create a temporary policy registry file
    content = """
    policies:
      p1:
        rego_path: "policies/check1"
      p2:
        rego_path: "policies/check2"
    """
    file_path = tmp_path / "policy_registry.yaml"
    file_path.write_text(content)

    return PolicyEngine(str(file_path), "http://localhost:8181/v1/data")



# 1. Test successful YAML load
def test_load_policy_registry(policy_engine):
    assert isinstance(policy_engine.policies, dict)
    assert "p1" in policy_engine.policies
    assert "p2" in policy_engine.policies


# 2. Test YAML file missing
def test_file_not_found():
    engine = PolicyEngine("nonexistent.yaml", "http://localhost:8181/v1/data")

    with pytest.raises(CGAgentException):
        engine._load_policy_registry()

# 3. Test invalid YAML content

def test_invalid_yaml(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: : bad")   # invalid YAML

    engine = PolicyEngine(str(bad_yaml), "http://localhost:8181/v1/data")

    with pytest.raises(CGAgentException):
        engine._load_policy_registry()


# 4. Test multi policy evaluation

def test_evaluate_multiple_policies(policy_engine):
    mock_result = {"result": {"allow": True}}

    with patch.object(policy_engine.opa_client, "query_policy", return_value=mock_result) as mock_q:
        output = policy_engine.evaluate({"x": 1})

        # Should call twice (p1 and p2)
        assert mock_q.call_count == 2
        assert output == mock_result


# 5. Missing rego_path → skip

def test_evaluate_skips_missing_rego_path(policy_engine):
    policy_engine.policies = {
        "p1": {"rego_path": "policies/check1"},
        "p2": {}   # missing rego_path
    }

    mock_result = {"result": {"allow": True}}

    with patch.object(policy_engine.opa_client, "query_policy", return_value=mock_result) as mock_q:
        output = policy_engine.evaluate({"x": 1})

        assert mock_q.call_count == 1
        assert output == mock_result

# 6. OPA network failure → CGAgentException

def test_opa_network_error(policy_engine):

    with patch.object(policy_engine.opa_client, "query_policy", side_effect=Exception("network down")):
        with pytest.raises(CGAgentException):
            policy_engine.evaluate({"x": 1})