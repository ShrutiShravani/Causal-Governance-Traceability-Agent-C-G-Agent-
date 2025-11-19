
from src.agent.conversational_query_agent import QueryAgent
from src.traceability.traceability_agent_service import TraceabilityAgentService
from src.traceability.graph_config import GraphDBClient
from logger import logging
import sys
from exception import CGAgentException

#intialization (setting up in environment)

def intialize_audit_system():
    try:
        #intilaize low level client
        graph_client= GraphDBClient()
        logging.info("graph db client initlaized")

        #intilaize the Traceability Service(the synthesize/retriever)
        traceability_agent_service=TraceabilityAgentService(graph_client)
   
        #initialize tjhe conversational query agent
        query_agent= QueryAgent(traceability_agent_service)

        logging.info("conversational query agent intialzied")
        logging.info("GRAS Accountability Loop Initialized.")
        return query_agent
    except Exception as e:
        raise CGAgentException(e,sys)


#interactive loop

def run_interactive_demo(query_agent:QueryAgent):
    print("--- CGO Audit Interface ---")
    print("Enter the client transaction ID you wish to audit (e.g., TXN-101).")
    
    # NOTE: In a real demo, you would ensure TXN-101 was run through the Orchestrator first.
    
    txn_id= input("Transaction ID: ").strip()

    while True:
        try:
            user_input= input(f"\nAudit for [{txn_id}]>").strip()
            if user_input.lower() in ['exit','quit']:
                print("Exiting Audit interface")
                break

            response=query_agent.process_query(user_input,txn_id)

            print(f"\n[AI Assistant]:{response}")

        except Exception as e:
            raise CGAgentException(e,sys)



if __name__=="__main__":
    agent= intialize_audit_system()
    run_interactive_demo(agent)