package policies.confidence_threshold

# Minimum confidence required for automatic approval
min_confidence := 0.7

# Main rule: allow if confidence >= threshold
allow := input.confidence >= min_confidence

# Optional: add message for transparency
message := sprintf(
    "Prediction confidence %v is below minimum threshold %v",
    [input.confidence, min_confidence]
)
