from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap, extract, inject
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import logging
from exception import CGAgentException
import os,sys
from typing import Dict,Any
from opentelemetry.propagators.tracecontext import TraceContextTextMapPropagator

SERVICE_NAME = "CausalGovernanceAgent-Service"
SERVICE_VERSION = "1.0.0"
OTLP_ENDPOINT = "localhost:4317"

def configure_tracer():
    """Sets up the global OpenTelemetry configuration"""
    if isinstance(trace.get_tracer_provider(),TracerProvider):
        return 
        
    resource = Resource.create({
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "environment":os.environ.get("ENV","development")
    })

    # Configure the OTLP exporter to send data to a collector
    otlp_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True) 

    # Create a BatchSpanProcessor to process and export spans efficiently
    span_processor = BatchSpanProcessor(otlp_exporter)

    # Initialize the TracerProvider with the resource and span processor
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(span_processor)

    # Set the global default tracer provider
    trace.set_tracer_provider(provider)

    # Set the global context propagator (CRITICAL for distributed tracing)
    set_global_textmap(TraceContextTextMapPropagator())
    RequestsInstrumentor().instrument()

def get_tracer():
    """ Returns the configured tracer object"""

    #ensure conifguration runs only once
    if not trace.get_tracer_provider():
        configure_tracer()
    return trace.get_tracer(__name__)

def start_agent_trace(span_name:str,client_id:str):
    """
    Called by the Orchestration Agent (OA) to start the root span for a transaction.
    Returns the root span object.
    """
    tracer= get_tracer()

    #start the root span and inject client context
    root_span= tracer.start_span(span_name)
    root_span.set_atrribute("client.id",client_id)
    return root_span

def inject_trace_context(payload:Dict[str,Any])->Dict[str,Any]:
    """
    Injects the current trace context into an outbound dictionary payload (e.g., HTTP headers).
    Used to pass context between agents (OA -> CA -> PA).
    """
    carrier={}
    inject(carrier)
    payload["trace_context"]=carrier
    return payload

def extract_trace_context(payload: dict):
    carrier = payload.get("trace_context", {})
    return extract(carrier)

