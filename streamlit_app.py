import streamlit as st
import time
import os
from dotenv import load_dotenv
from agents import AGENT_REGISTRY

# Load environment variables with explicit path
load_dotenv('c:\\Users\\user\\Desktop\\campaign builder\\.env')

# Verify API key is loaded
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

from core.vector_db import load_mock_data, initialize_vector_db
from core.state import CampaignState
from workflows.campaign_workflow import build_campaign_workflow

# Page configuration
st.set_page_config(
    page_title="Autonomous Campaign Builder",
    page_icon="🛴",
    layout="wide"
)

# Initialize session state if not already done
if 'state' not in st.session_state:
    st.session_state.state = None

# Header
st.title("🚀 Autonomous Campaign Builder")
st.markdown("Generate complete marketing campaigns with AI")

# Sidebar
st.sidebar.header("Configuration")
goal = st.sidebar.text_area(
    "Campaign Goal",
    value="Boost Q2 SUV sales in the Western region by 15%",
    height=100
)

# Main content
if st.sidebar.button("Generate Campaign"):
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load data
    status_text.text("Loading data...")
    sales_data, campaign_data, customer_segments = load_mock_data()
    
    # Initialize vector database
    status_text.text("Initializing knowledge base...")
    vector_db = initialize_vector_db(campaign_data, customer_segments)
    
    # Build workflow
    campaign_workflow = build_campaign_workflow()
    
    # Initialize state
    initial_state = CampaignState(
        goal=goal,
        vector_db=vector_db,
        sales_data=sales_data
    )
    
    # Create tabs for each section
    tabs = st.tabs([
        "Market Analysis", 
        "Target Audience", 
        "Campaign Strategy", 
        "Content", 
        "Simulation", 
        "Final Report",
        "Email Distribution"  # Added new tab for email distribution
    ])
    
    # Run workflow with progress updates
    steps = list(AGENT_REGISTRY.keys())  # Use the agent registry keys instead
    total_steps = len(steps)
    
    # Process the workflow stream with dictionary access
    for i, output in enumerate(campaign_workflow.stream(initial_state)):
        node = list(output.keys())[0] if output else None
        if node:
            # Update progress
            progress = (i + 1) / total_steps
            progress_bar.progress(progress)
            status_text.text(f"Step {i+1}/{total_steps}: Completed {node}")
            
            # Update corresponding tab
            state = output[node]
            if node == "research_market_trends" and "market_analysis" in state:
                with tabs[0]:
                    st.markdown(state["market_analysis"])
            elif node == "segment_audience" and "audience_segments" in state:
                with tabs[1]:
                    st.markdown(state["audience_segments"])
            elif node == "create_campaign_strategy" and "campaign_strategy" in state:
                with tabs[2]:
                    st.markdown(state["campaign_strategy"])
            elif node == "generate_content" and "campaign_content" in state:
                with tabs[3]:
                    st.markdown(state["campaign_content"])
            elif node == "simulate_campaign" and "simulation_results" in state:
                with tabs[4]:
                    st.markdown(state["simulation_results"])
            elif node == "generate_final_report" and "final_report" in state:
                with tabs[5]:
                    st.markdown(state["final_report"])
            elif node == "send_campaign_emails" and "email_status" in state:
                with tabs[6]:
                    st.markdown(state["email_status"])
            
            # Add a small delay for UI updates
            time.sleep(0.5)
    
    # Final update
    progress_bar.progress(1.0)
    status_text.text("Campaign generation complete!")
    
    # Get final state
    final_state = campaign_workflow.invoke(initial_state)
    
    # Store the final state in session state for later use
    st.session_state.state = final_state
    
    # Display download button for the report
    st.download_button(
        label="Download Report",
        data=final_state["final_report"],
        file_name="campaign_report.md",
        mime="text/markdown"
    )
    
    # Add email distribution button in the Email Distribution tab
    with tabs[6]:
        st.subheader("Campaign Distribution")
        
        if st.button("Send Campaign Emails"):
            with st.spinner("Sending emails to customers..."):
                # Create a progress bar for email sending
                email_progress = st.progress(0)
                email_status = st.empty()
                
                # Call the email sending agent
                email_state = AGENT_REGISTRY["send_campaign_emails"](st.session_state.state)
                
                # Update session state
                st.session_state.state = email_state
                
                # Display results
                if hasattr(email_state, 'email_status'):
                    if "failed" in email_state.email_status.lower():
                        st.error(email_state.email_status)
                    else:
                        st.success(email_state.email_status)
                        email_progress.progress(1.0)

else:
    # Display instructions when not running
    st.info("Enter your campaign goal in the sidebar and click 'Generate Campaign' to start.")
    
    # Sample output
    st.markdown("""
    ## Sample Output
    
    The campaign builder will generate:
    
    1. **Market Analysis** - Trends, opportunities, and competitive landscape
    2. **Target Audience** - Primary and secondary segments with messaging points
    3. **Campaign Strategy** - Timeline, channels, budget, and KPIs
    4. **Content Examples** - Email, social media, and landing page content
    5. **Performance Simulation** - Projected results and optimization recommendations
    6. **Final Report** - Complete campaign plan
    7. **Email Distribution** - Send campaign emails to target customers
    """)
