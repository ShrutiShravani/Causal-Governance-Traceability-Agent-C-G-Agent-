package policies.provenance_compliance

approved_models := {"gb_model_v1", "rf_model_v2", "xgb_model_v3"}

default allow = false

allow {
    input.model_version != ""
    approved_models[input.model_version]
}

message := "Approved model" {
    allow
}

message := "Model not approved" {
    not allow
}