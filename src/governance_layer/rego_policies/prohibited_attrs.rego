package policies.prohibited_attributes

# List of prohibited attributes
prohibited_attributes := ["SEX", "AGE", "RACE", "ETHNICITY"]

# Maximum allowed influence for prohibited attributes
max_influence := 0.1

# Main rule: evaluate the input top_drivers
violations := [v |
    input_top := input.top_drivers[_]
    attr := input_top.feature
    value := input_top.shap_value
    prohibited_attributes[_] == attr
    abs(value) > max_influence
    v := {
        "rule": "ProhibitedAttributeCheck",
        "attribute": attr,
        "influence": abs(value),
        "threshold": max_influence,
        "message": sprintf("%v influence exceeds permissible limit", [attr])
    }
]

# Allow if no violations
allow := count(violations) == 0
