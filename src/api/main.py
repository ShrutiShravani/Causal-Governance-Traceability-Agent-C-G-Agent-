from fastapi import FastAPI, Request
from src.agent.orchestration_agent import OrchestratorAgent
from src.traceability.telemetry import get_tracer, inject_trace_context
import uuid

app=FastAPI()
tracer=get_tracer()

@app.post("/orchestrate")

async def orchestrate(request:Request):
    payload= await request.json()

    with tracer.start_as_current_span("orchestrator_api") as span:
        trace_id= format(span.get_span_context().trace_id,"032x")
        client_transaction_id = f"txn_{uuid.uuid4().hex[:8]}"

        #inject trace context in to pyaload for downstream agents
        payload= inject_trace_context(payload)

        orchestrator=OrchestratorAgent(trace_id=trace_id,client_trnsaction_id=client_transaction_id)
        result= orchestrator.run(payload)

        return {"trace_id":trace_id,"result":result}