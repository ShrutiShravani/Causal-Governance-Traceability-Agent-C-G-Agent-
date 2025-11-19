package policies.provenance_check

# List of approved/trusted model versions
approved_models := {
    "gb_model_v1",
    "rf_model_v2",
    "xgb_model_v3"
}

# Main rule: allow if model_version is in the approved list
allow := input.model_version != "" {
    input.model_version == approved_models[_]
}

# Optional message for audit
message := sprintf(
    "Model version '%v' is not approved for use",
    [input.model_version]
)
