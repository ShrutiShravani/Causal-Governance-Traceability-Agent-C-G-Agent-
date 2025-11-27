package policies.prohibited_attrs

# Helper function to check violations
check_violations() = violations {
    prohibited_attributes := ["SEX", "AGE", "EDUCATION"]
    max_influence := 0.3
    
    violations = [violation |
        driver_pair := input.xai_artifact.local_top_drivers[_]
        prohibited_attr := driver_pair[0]
        influence_value := driver_pair[1]
        prohibited_attributes[_] == prohibited_attr
        abs(influence_value) > max_influence
        
        violation := {
            "attribute": prohibited_attr,
            "influence": abs(influence_value),
            "threshold": max_influence,
            "message": sprintf("Prohibited attribute %v has excessive influence: %v", [prohibited_attr, abs(influence_value)])
        }
    ]
}

# ONLY ONE root document
result := {
    "prohibited_attributes": ["SEX", "AGE", "EDUCATION"],
    "max_influence": 0.3,
    "violations": check_violations(),
    "allow": count(check_violations()) == 0,
    "checked_attributes": ["SEX", "AGE", "EDUCATION"]
}