from src.governance_layer.opa_client import OPAClient
from logger import logging


def setup_opa_policies():
    """Load all your governance policies"""
    opa = OPAClient("http://localhost:8181")
    
    policies = [
        (r"src/governance_layer/rego_policies/oversight_threshold.rego", "oversight_threshold"),
        (r"src/governance_layer/rego_policies/prohibited_attrs.rego", "prohibited_attrs"),
    ]
    
    # Load each policy
    for policy_file,policy_id in policies:
        try:
            with open(policy_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print("Policy content:")
                print("=" * 50)
                print(content)
                print("=" * 50)
        except Exception as e:
            print(f"Cannot read file: {e}")
            continue
        success = opa.load_policy(policy_file,policy_id)
        if not success:
            logging.info(f"Failed to load: {policy_file}")
    
    # List all loaded policies to verify
    logging.info("\nVerifying loaded policies:")
    loaded_policies = opa.list_policies()
    for policy in loaded_policies.get('result',[]):
        print(f"{policy['id']}")
    
    logging.info("All policies loaded and verified!")

if __name__ == "__main__":
    setup_opa_policies()