import sys
import os

# Add src to path
sys.path.append('src')

from governance_layer.policy_engine import PolicyEngine
from opa_client import OPAClient

def test_policy_engine_basic():
    """Test basic PolicyEngine functionality"""
    print("🧪 Testing PolicyEngine Basic Functionality")
    print("=" * 50)
    
    try:
        # Initialize PolicyEngine
        engine = PolicyEngine("governance_layer/policy_registry.yaml")
        
        print("✅ PolicyEngine initialized successfully")
        print(f"📋 Loaded {len(engine.policies)} policies:")
        for policy_name, policy_meta in engine.policies.items():
            print(f"   - {policy_name}: {policy_meta.get('name')}")
        
        return True
        
    except Exception as e:
        print(f"❌ PolicyEngine initialization failed: {e}")
        return False

def test_policy_evaluation():
    """Test policy evaluation with sample data"""
    print("\n🧪 Testing Policy Evaluation")
    print("=" * 50)
    
    try:
        engine = PolicyEngine("governance_layer/policy_registry.yaml")
        
        # Test case 1: Good prediction
        good_prediction = {
            "confidence": 0.85,
            "model_version": "gb_model_v1",
            "features_used": ["utilization_rate", "total_months_late"],
            "prediction_label": 1,
            "probability_class_1": 0.85
        }
        
        print("📊 Testing GOOD prediction:")
        print(f"   Input: {good_prediction}")
        
        results = engine.evaluate(good_prediction)
        print(f"   Results: {results}")
        
        # Test case 2: Problematic prediction  
        bad_prediction = {
            "confidence": 0.45,
            "model_version": "invalid_model",
            "features_used": ["age", "gender"],  # Prohibited attributes
            "prediction_label": 1,
            "probability_class_1": 0.45
        }
        
        print("\n📊 Testing BAD prediction:")
        print(f"   Input: {bad_prediction}")
        
        results = engine.evaluate(bad_prediction)
        print(f"   Results: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Policy evaluation failed: {e}")
        return False

def test_individual_policies():
    """Test each policy individually"""
    print("\n🧪 Testing Individual Policies")
    print("=" * 50)
    
    opa = OPAClient("http://localhost:8181/v1/data")
    
    test_cases = [
        {
            "policy": "policies/oversight_threshold",
            "input": {"confidence": 0.85},
            "description": "High confidence"
        },
        {
            "policy": "policies/oversight_threshold", 
            "input": {"confidence": 0.45},
            "description": "Low confidence"
        },
        {
            "policy": "policies/provenance_compliance_check",
            "input": {"model_version": "gb_model_v1"},
            "description": "Approved model"
        },
        {
            "policy": "policies/provenance_compliance_check",
            "input": {"model_version": "invalid_model"},
            "description": "Unapproved model"
        }
    ]
    
    for test in test_cases:
        print(f"\n🔍 Testing: {test['description']}")
        print(f"   Policy: {test['policy']}")
        print(f"   Input: {test['input']}")
        
        try:
            result = opa.query_policy(test["policy"], test["input"])
            print(f"   ✅ Result: {result.get('result', {})}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def test_policy_engine_integration():
    """Test full integration with Critic Agent data"""
    print("\n🧪 Testing Full Integration")
    print("=" * 50)
    
    try:
        engine = PolicyEngine("governance_layer/policy_registry.yaml")
        
        # Simulate data from Critic Agent (XAI artifact + prediction)
        critic_agent_data = {
            # From XAI artifact
            "confidence": 0.78,
            "model_version": "gb_model_v1", 
            "local_top_drivers": {
                "utilization_rate": 0.35,
                "total_months_late": 0.25
            },
            "risk_factors": ["utilization_rate"],
            
            # From prediction
            "prediction_label": 1,
            "probability_class_1": 0.78,
            
            # Additional governance data
            "features_used": ["utilization_rate", "total_months_late", "PAY_0"],
            "data_provenance": "approved_source",
            "prediction_timestamp": "2024-01-15T10:30:00Z"
        }
        
        print("📊 Testing with Critic Agent data:")
        for key, value in critic_agent_data.items():
            if key != "local_top_drivers":  # Skip large dict for readability
                print(f"   {key}: {value}")
        
        results = engine.evaluate(critic_agent_data)
        print(f"📋 Governance Results: {results}")
        
        # Interpret results
        print("\n🔍 Result Interpretation:")
        if results.get('result'):
            for policy_name, policy_result in results['result'].items():
                if isinstance(policy_result, dict):
                    allowed = policy_result.get('allow', False)
                    status = "✅ ALLOWED" if allowed else "❌ DENIED"
                    print(f"   {policy_name}: {status}")
                else:
                    print(f"   {policy_name}: {policy_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting PolicyEngine Tests")
    print("Make sure OPA is running and policies are loaded!")
    print()
    
    # Run tests
    test1 = test_policy_engine_basic()
    test2 = test_policy_evaluation() 
    test3 = test_individual_policies()
    test4 = test_policy_engine_integration()
    
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY:")
    print(f"   Basic Init: {'✅' if test1 else '❌'}")
    print(f"   Evaluation:  {'✅' if test2 else '❌'}")
    print(f"   Integration: {'✅' if test4 else '❌'}")