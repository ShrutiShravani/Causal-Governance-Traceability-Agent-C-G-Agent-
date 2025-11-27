package policies.oversight_threshold

# Minimum confidence required for automatic approval
min_confidence := 0.7

# SINGLE root document - will now return results
result := {
    "allow": input.confidence_scores >= min_confidence,
    "confidence_score": input.confidence_scores,
    "min_required_confidence": min_confidence,
    "requires_human_review": input.confidence_scores < min_confidence,
    "message": sprintf("Confidence: %v, Threshold: %v", [input.confidence_scores, min_confidence])
}