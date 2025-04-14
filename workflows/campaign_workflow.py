from langgraph.graph import StateGraph
from core.state import CampaignState
from agents import AGENT_REGISTRY

def build_campaign_workflow():
    workflow = StateGraph(CampaignState)

    # Add all nodes from the registry
    for name, func in AGENT_REGISTRY.items():
        workflow.add_node(name, func)
    
    # Create the workflow sequence
    nodes = list(AGENT_REGISTRY.keys())
    for i in range(len(nodes) - 1):
        workflow.add_edge(nodes[i], nodes[i+1])

    workflow.set_entry_point(nodes[0])
    workflow.set_finish_point(nodes[-1])

    return workflow.compile()