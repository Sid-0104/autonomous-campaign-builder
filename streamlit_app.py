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

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Autonomous Campaign Builder",
    page_icon="🛴",
    layout="wide"
)

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
        "Final Report"
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
            
            # Add a small delay for UI updates
            time.sleep(0.5)
    
    # Final update
    progress_bar.progress(1.0)
    status_text.text("Campaign generation complete!")
    
    # Get final state
    final_state = campaign_workflow.invoke(initial_state)
    
    # Display download button for the report
    st.download_button(
        label="Download Report",
        data=final_state["final_report"],
        file_name="campaign_report.md",
        mime="text/markdown"
    )

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
    """)
# Add this to your imports section
import streamlit as st
from agents import AGENT_REGISTRY
from core.state import CampaignState
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Your existing code...

# Add this in your sidebar or main UI section where appropriate
with st.sidebar.expander("Email Configuration"):
    st.write("Configure email settings for campaign distribution")
    
    # Email configuration inputs
    sender_email = st.text_input("Sender Email", value=os.getenv("SENDER_EMAIL", ""))
    email_password = st.text_input("Email Password", type="password", value=os.getenv("EMAIL_APP_PASSWORD", ""))
    smtp_server = st.text_input("SMTP Server", value=os.getenv("SMTP_SERVER", "smtp.gmail.com"))
    smtp_port = st.number_input("SMTP Port", value=int(os.getenv("SMTP_PORT", 465)))
    
    # Save configuration button
    if st.button("Save Email Configuration"):
        os.environ["SENDER_EMAIL"] = sender_email
        os.environ["EMAIL_APP_PASSWORD"] = email_password
        os.environ["SMTP_SERVER"] = smtp_server
        os.environ["SMTP_PORT"] = str(smtp_port)
        st.success("Email configuration saved!")

# Add this where you want the email sending functionality to appear
# For example, after the final report is generated
if hasattr(state, 'final_report') and state.final_report:
    st.subheader("Campaign Distribution")
    
    if st.button("Send Campaign Emails"):
        with st.spinner("Sending emails to customers..."):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Call the email sending agent
            state = AGENT_REGISTRY["send_campaign_emails"](state)
            
            # Display results
            if hasattr(state, 'email_status'):
                if "failed" in state.email_status.lower():
                    st.error(state.email_status)
                else:
                    st.success(state.email_status)
