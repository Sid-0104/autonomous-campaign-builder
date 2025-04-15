from langgraph.graph import StateGraph
from core.state import CampaignState
from agents import AGENT_REGISTRY

def build_campaign_workflow():
    workflow = StateGraph(CampaignState)
    
    # Filter out the send_campaign_emails agent from the workflow
    # This is already done in your current code
    workflow_nodes = {k:v for k,v in AGENT_REGISTRY.items() if k != "send_campaign_emails"}
    
    # Add remaining nodes
    for name, func in workflow_nodes.items():
        workflow.add_node(name, func)
    
    # Create the workflow sequence
    nodes = list(workflow_nodes.keys())
    for i in range(len(nodes) - 1):
        workflow.add_edge(nodes[i], nodes[i+1])
    
    workflow.set_entry_point(nodes[0])
    workflow.set_finish_point(nodes[-1])
    
    return workflow.compile()
