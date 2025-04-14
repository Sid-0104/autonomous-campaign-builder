import os
import time
from dotenv import load_dotenv
from core.vector_db import load_mock_data, initialize_vector_db
from core.state import CampaignState
from workflows.campaign_workflow import build_campaign_workflow

# Load environment variables
load_dotenv()

# Configuration
STEP_DELAY = int(os.environ.get("STEP_DELAY", 10))

def run_autonomous_campaign_builder(goal: str):
    # Load data
    sales_data, campaign_data, customer_segments = load_mock_data()
    
    # Initialize vector database
    vector_db = initialize_vector_db(campaign_data, customer_segments)
    
    # Build workflow
    campaign_workflow = build_campaign_workflow()
    
    # Initialize state
    initial_state = CampaignState(
        goal=goal,
        vector_db=vector_db,
        sales_data=sales_data
    )
    
    # Run workflow with progress updates
    print(f"Starting campaign builder with goal: {goal}")
    print("=" * 50)
    
    for output in campaign_workflow.stream(initial_state):
        node = list(output.keys())[0] if output else None
        if node:
            print(f"Completed: {node}")
            time.sleep(STEP_DELAY)
    
    # Get final state
    final_state = campaign_workflow.invoke(initial_state)
    
    return final_state["final_report"]

if __name__ == "__main__":
    campaign_goal = "Boost Q2 SUV sales in the Western region by 15%"
    final_report = run_autonomous_campaign_builder(campaign_goal)
    print("\nFINAL REPORT:")
    print("=" * 50)
    print(final_report)