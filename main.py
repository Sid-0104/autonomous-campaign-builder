import os
import time
import argparse
from dotenv import load_dotenv
from core.vector_db import initialize_vector_db
from core.state import CampaignState
from workflows.campaign_workflow import build_campaign_workflow

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

# Configuration
STEP_DELAY = int(os.environ.get("STEP_DELAY", 5))

def display_domain_options():
    """Display available domain options"""
    print("\nAvailable domains:")
    print("  1. automotives - Automotive industry campaigns")
    print("  2. healthcare - Healthcare industry campaigns")
    print("  3. powerenergy - Power and energy sector campaigns")

def display_model_options():
    """Display available LLM providers"""
    print("\nAvailable LLM providers:")
    print("  1. gemini - Google's Gemini model (default)")
    print("  2. openai - OpenAI's GPT models")

def run_autonomous_campaign_builder(goal: str, domain: str = "automotives", model_provider: str = "gemini"):
    """
    Run the autonomous campaign builder with dynamic domain and model selection
    
    Args:
        goal: The campaign goal
        domain: The industry domain (automotives, healthcare, powerenergy)
        model_provider: The LLM provider to use (gemini, openai)
    
    Returns:
        The final campaign report and email status
    """
    print(f"\n🚀 Initializing campaign builder for {domain.upper()} domain using {model_provider.upper()} model")
    print(f"🎯 Campaign Goal: {goal}")
    print("=" * 80)
    
    # Initialize vector database with domain-specific data
    print(f"📊 Loading {domain} domain data and initializing vector database...")
    vector_db = initialize_vector_db(domain=domain, model_provider=model_provider)
    
    # Load domain-specific sales data
    from core.vector_db import load_documents_by_domain
    sales_data, _, _, _ = load_documents_by_domain(domain)
    
    # Build workflow
    campaign_workflow = build_campaign_workflow()
    
    # Initialize state with domain and model selection
    initial_state = CampaignState(
        goal=goal,
        vector_db=vector_db,
        sales_data=sales_data,
        selected_domain=domain,
        selected_llm=model_provider
    )
    
    # Run workflow with progress updates
    print("\n🔄 Starting campaign workflow execution:")

    for i, output in enumerate(campaign_workflow.stream(initial_state)):
        node = list(output.keys())[0] if output else None
        if node:
            print(f"\n🔄 Executing {node}...")
            time.sleep(STEP_DELAY)

    # Return final report and email status
    return initial_state.final_report, initial_state.email_status

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Autonomous Marketing Campaign Builder")
    parser.add_argument("--goal", type=str, help="Campaign goal", 
                        default="Boost Q2 SUV sales in the Western region by 15%")
    parser.add_argument("--domain", type=str, choices=["automotives", "healthcare", "powerenergy"], 
                        default="automotives", help="Industry domain")
    parser.add_argument("--model", type=str, choices=["gemini", "openai"], 
                        default="gemini", help="LLM provider")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Import random here to avoid potential issues with the mock data generation
    import random
    
    if args.interactive:
        print("\n🤖 Welcome to the Autonomous Marketing Campaign Builder 🤖")
        print("=" * 80)
        
        # Get campaign goal
        print("\nPlease enter your campaign goal:")
        campaign_goal = input("> ")
        if not campaign_goal.strip():
            campaign_goal = "Boost Q2 SUV sales in the Western region by 15%"
            print(f"Using default goal: {campaign_goal}")
        
        # Get domain selection
        display_domain_options()
        domain_choice = input("\nSelect domain (1-3) or enter name directly: ")
        
        domain_map = {
            "1": "automotives",
            "2": "healthcare", 
            "3": "powerenergy"
        }
        
        selected_domain = domain_map.get(domain_choice, domain_choice.lower())
        if selected_domain not in ["automotives", "healthcare", "powerenergy"]:
            selected_domain = "automotives"
            print(f"Invalid domain. Using default: {selected_domain}")
        
        # Get model selection
        display_model_options()
        model_choice = input("\nSelect LLM provider (1-2) or enter name directly: ")
        
        model_map = {
            "1": "gemini",
            "2": "openai"
        }
        
        selected_model = model_map.get(model_choice, model_choice.lower())
        if selected_model not in ["gemini", "openai"]:
            selected_model = "gemini"
            print(f"Invalid model. Using default: {selected_model}")
    else:
        # Use command line arguments
        campaign_goal = args.goal
        selected_domain = args.domain
        selected_model = args.model
    
    # Run the campaign builder
    final_report = run_autonomous_campaign_builder(
        campaign_goal, 
        domain=selected_domain,
        model_provider=selected_model
    )
    
    # Display results
    print("\n📋 FINAL CAMPAIGN REPORT:")
    print("=" * 80)
    print(final_report)