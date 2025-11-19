# ci_metrics_pusher.py (Executed by the CI/CD Pipeline)

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from typing import Dict, Any, List
import time
import logging
import os

# --- Configuration ---
# NOTE: In a real environment, this URL would point to the live Pushgateway service.
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
JOB_NAME = "cgo-agent-ci-build"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def push_ci_metrics(
    build_duration_seconds: float,
    total_tests: int,
    failed_tests: int,
    policy_check_pass_rate: float,
    vulnerability_count: int,
    agent_benchmark_latency_p95: float
):
    """
    Calculates and pushes critical CI/CD metrics to the Pushgateway.
    This runs at the end of the build job to record the final artifact quality.
    """
    
    # 1. Initialize a new registry for this push (mandatory for Pushgateway)
    registry = CollectorRegistry()

    # 2. Define Gauges (Metrics that hold a single value)
    
    # Code Integrity Metrics
    g_tests_failed = Gauge('ci_tests_failed_total', 'Total number of unit tests that failed.', registry=registry)
    g_build_duration = Gauge('ci_build_duration_seconds', 'Time taken to complete the CI job.', registry=registry)
    g_vulnerabilities = Gauge('sca_critical_vulnerabilities_total', 'Number of critical vulnerabilities found by the security scanner.', registry=registry)

    # Agentic Architecture & Compliance Metrics
    g_policy_pass_rate = Gauge('governance_policy_pass_rate', 'Overall success rate of agent policy checks in CI tests.', registry=registry)
    g_benchmark_latency = Gauge('agent_benchmark_latency_p95', '95th percentile internal agent execution time (ms).', registry=registry)

    # 3. Set the Metric Values
    g_tests_failed.set(failed_tests)
    g_build_duration.set(build_duration_seconds)
    g_vulnerabilities.set(vulnerability_count)
    g_policy_pass_rate.set(policy_check_pass_rate)
    g_benchmark_latency.set(agent_benchmark_latency_p95)

    # 4. Push to Gateway
    try:
        logging.info(f"Pushing metrics for job '{JOB_NAME}' to {PUSHGATEWAY_URL}...")
        push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
        logging.info("Metrics successfully pushed.")
    except Exception as e:
        # Crucial: Log the failure, but let the CI pipeline finish (metrics failure is non-critical)
        logging.error(f"Failed to connect to Pushgateway at {PUSHGATEWAY_URL}. Metrics not recorded: {e}")

if __name__ == "__main__":
    # --- SIMULATED EXECUTION AT END OF CI JOB ---
    
    start_time = time.time()

    # --- SIMULATE TEST EXECUTION AND RESULTS ---
    # Assume 100 tests ran, 5 failed, total build time was 45 seconds.
    # Assume Policy Enforcement test (the high-value unit test) passed 98.7% of its sub-checks.
    
    TEST_RESULTS = {
        "duration": 45.3,
        "total": 100,
        "failed": 5,
        "policy_pass_rate": 0.987,
        "vulnerabilities": 2,
        "p95_latency_ms": 120.5 # 120 milliseconds internal agent execution time
    }

    push_ci_metrics(
        build_duration_seconds=TEST_RESULTS["duration"],
        total_tests=TEST_RESULTS["total"],
        failed_tests=TEST_RESULTS["failed"],
        policy_check_pass_rate=TEST_RESULTS["policy_pass_rate"],
        vulnerability_count=TEST_RESULTS["vulnerabilities"],
        agent_benchmark_latency_p95=TEST_RESULTS["p95_latency_ms"]
    )