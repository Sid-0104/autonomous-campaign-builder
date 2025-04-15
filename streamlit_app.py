import streamlit as st
import time
import os
import re
import pandas as pd
import traceback
from dotenv import load_dotenv
from core.vector_db import initialize_vector_db, load_documents_by_domain
from core.state import CampaignState
from workflows.campaign_workflow import build_campaign_workflow
from fpdf import FPDF
import logging
from logging.handlers import RotatingFileHandler
from agents.send_emails import send_campaign_emails


# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

# Configuration
STEP_DELAY = int(os.environ.get("STEP_DELAY", 5))

# Configure logging
def setup_logging():
    log_dir = os.path.join(script_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'campaign_builder.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- PDF Generator ---
def strip_emojis(content):
    return re.sub(r'[^\x00-\x7F]+', '', content)

def generate_pdf(section_title, content):
    clean_content = strip_emojis(content)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"{section_title}\n\n{clean_content}")
    return pdf.output(dest='S').encode('latin-1')

# ===== Section Renderer + Feedback =====
def render_section(title, content, filename, node_name, key_prefix):
    logger.debug(f"Rendering section: {title}")
    try:
        st.markdown(f"### {title}")
        st.write(content)
        
        # Download button
        download_key = f"{key_prefix}_pdf_button"
        st.download_button(
            "📥 Download PDF", 
            data=generate_pdf(title, content), 
            file_name=filename,
            key=download_key
        )
        
        # Feedback system
        st.markdown("#### 🗳️ How would you rate this section?")
        feedback_key = f"{key_prefix}_{node_name}"
        
        # Create columns for feedback buttons with proper spacing
        cols = st.columns([1, 1, 4])
        
        # Use unique keys for each feedback button
        positive_key = f"{feedback_key}_positive"
        negative_key = f"{feedback_key}_negative"
        
        # Check if feedback already given
        if feedback_key in st.session_state.feedback:
            st.info("✅ Feedback already recorded for this section")
        else:
            # Add feedback buttons with unique keys
            with cols[0]:
                if st.button("👍", key=positive_key):
                    st.session_state.feedback[feedback_key] = "positive"
                    st.rerun()  # Refresh to update the feedback status
            
            with cols[1]:
                if st.button("👎", key=negative_key):
                    st.session_state.feedback[feedback_key] = "negative"
                    st.rerun()  # Refresh to update the feedback status

    except Exception as e:
        logger.error(f"Failed to render section {title}: {str(e)}\n{traceback.format_exc()}")
        st.error("Error generating this section. Please check logs.")
        
# Function for vector db initialization with caching
@st.cache_resource
def cached_initialize_vector_db(domain, model_provider):
    """Cache the vector db initialization to avoid redundant loading"""
    return initialize_vector_db(domain=domain, model_provider=model_provider)

def run_autonomous_campaign_builder(goal: str, domain: str = "automotives", model_provider: str = "gemini"):
    logger.info(f"Initializing campaign builder for {domain} domain using {model_provider}")
    try:
        # Add retry logic for API quota errors
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                vector_db = cached_initialize_vector_db(domain=domain, model_provider=model_provider)
                logger.debug("Vector DB initialized successfully")
                break
            except Exception as e:
                retry_count += 1
                if "ResourceExhausted" in str(e) and retry_count < max_retries:
                    logger.warning(f"Gemini API quota exceeded, retrying ({retry_count}/{max_retries}): {str(e)}")
                    model_provider = "openai"
                    time.sleep(5)  # Short delay before retry
                elif retry_count >= max_retries:
                    logger.error(f"Failed to initialize vector DB after {max_retries} attempts")
                    raise
                else:
                    logger.error(f"Error initializing vector DB: {str(e)}")
                    raise

        sales_data, _, _, _ = load_documents_by_domain(domain)
        logger.debug(f"Loaded {len(sales_data)} sales records")
        
        campaign_workflow = build_campaign_workflow()
        logger.debug("Campaign workflow built")
        
        # Update model provider in state if changed
        initial_state = CampaignState(
            goal=goal,
            vector_db=vector_db,
            sales_data=sales_data,
            selected_domain=domain,
            selected_llm=model_provider if model_provider == "gemini" else "openai"
        )
        
        return campaign_workflow, initial_state
        
    except Exception as e:
        logger.error(f"Failed to initialize campaign builder: {str(e)}")
        st.error(f"API Error: {str(e)}")
        raise

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = None
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = "gemini"
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = "automotives"
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'tab_contents' not in st.session_state:
    st.session_state.tab_contents = {}
if 'generated' not in st.session_state:
    st.session_state.generated = False

# Page configuration
st.set_page_config(page_title="Autonomous Campaign Builder", page_icon=":scooter:", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        :root {--background-color: rgba(255, 255, 255, 1.0); --text-color: #1c1c1c; --card-bg: white; --border-color: #e6e6e6; --heading-color: #2c3e50; --shadow-color: rgba(0, 0, 0, 0.1); --tab-bg: #f1f5f9; --tab-selected-bg: white;}
        @media (prefers-color-scheme: dark) {:root {--background-color: rgba(30, 30, 30, 0.95); --text-color: #f1f1f1; --card-bg: #2d2d2d; --border-color: #444444; --heading-color: #8ab4f8; --shadow-color: rgba(0, 0, 0, 0.3); --tab-bg: #383838; --tab-selected-bg: #2d2d2d;}}
        body {background: linear-gradient(var(--background-color), var(--background-color)), url('https://images.unsplash.com/photo-1551836022-4c4c79ecde16?auto=format&fit=crop&w=1400&q=80') no-repeat center center fixed; background-size: cover; color: var(--text-color); font-family: 'Segoe UI', sans-serif;}
        #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
        .stApp {background-color: var(--background-color); padding: 0 !important; border-radius: 0 !important; box-shadow: none !important; max-width: 100% !important; margin: 0 !important;}
        .stTabs [role="tab"] {font-size: 16px; padding: 10px 20px; margin-right: 5px; border: 1px solid var(--border-color); background-color: var(--tab-bg); border-radius: 6px 6px 0 0; color: var(--text-color);}
        .stTabs [aria-selected="true"] {background-color: var(--tab-selected-bg); border-bottom: none; font-weight: bold;}
        .tab-content {animation: fadein 0.6s ease-in; background-color: var(--card-bg); padding: 20px; border-radius: 0 0 10px 10px; box-shadow: 0 2px 4px var(--shadow-color); margin-top: -1px; border: 1px solid var(--border-color); border-top: none;}
        .stTextArea textarea {border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; font-size: 16px; background-color: var(--card-bg); color: var(--text-color);}
        .stButton button {border-radius: 8px; font-weight: 500; transition: all 0.3s ease;}
        .stButton button:hover {transform: translateY(-2px); box-shadow: 0 4px 8px var(--shadow-color);}
        h1, h2, h3 {color: var(--heading-color); font-weight: 600;}
        .stTabs [data-baseweb="tab-panel"] {background-color: var(--card-bg); padding: 15px; border-radius: 0 0 10px 10px; border: 1px solid var(--border-color); border-top: none; color: var(--text-color);}
        .stProgress > div > div {background-color: #4CAF50;}
        @keyframes fadein {from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: translateY(0);}}
        /* Custom style for feedback section */
        .feedback-header {font-size: 14px !important;} /* Reduced font size for "Was this section helpful?" */
    </style>
""", unsafe_allow_html=True)

def main():
    logger.info("Starting Streamlit application")
    
    try:
        # Sidebar configuration
        with st.sidebar:
            # Set plain white background and remove all top padding/margin
            st.markdown(
                """
                <style>
                [data-testid="stSidebar"] {
                    background-color: #FFFFFF;
                    margin: 0;
                    padding: 0;
                    height: 100vh; /* Ensure sidebar takes full height */
                }
                [data-testid="stSidebarContent"] {
                    padding-top: 0 !important;
                    margin-top: 0 !important;
                }
                [data-testid="stSidebarNav"] + div {
                    margin-top: -15px !important; /* Stronger adjustment for residual spacing */
                    padding-top: 0 !important;
                }
                /* Target any additional default padding/margin */
                .sidebar .sidebar-content {
                    margin-top: 0 !important;
                    padding-top: 0 !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            col1, col2 = st.columns([1, 3.5], gap="small")
            with col1:
                # Use absolute path for the image
                image_path = os.path.join(script_dir, "assets", "info.png")
                if os.path.exists(image_path):
                    st.image(image_path, width=50, use_container_width=False)
                else:
                    st.write("🤖")  # Fallback emoji if image not found
            with col2:
                st.markdown(
                    """
                    <div style="display: flex; align-items: flex-start; height: 80px; margin: 0; padding: 0;">
                        <h1 style='font-size: 26px; color: #4B8BBE; margin: 0; padding: 0; line-height: 1;'>
                            Autonomous<br>Campaign Builder
                        </h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("---")
            st.subheader("Switch LLM")
            for llm in ["Gemini", "OpenAI"]:
                if st.button(llm, type="primary" if st.session_state.selected_llm == llm.lower() else "secondary", use_container_width=True):
                    st.session_state.selected_llm = llm.lower()
                    st.rerun()
            st.markdown(f"**Using: {st.session_state.selected_llm.upper()}**")
            
            st.subheader("Select Industry")
            # Using domain options from main.py
            for industry in ["Automotives", "Healthcare", "Powerenergy"]:
                domain_value = industry.lower()
                if st.button(industry, type="primary" if st.session_state.selected_domain == domain_value else "secondary", use_container_width=True):
                    st.session_state.selected_domain = domain_value
                    st.rerun()
            st.markdown(f"**Industry: {st.session_state.selected_domain.upper()}**")
    
        # Main content
        col1, col2 = st.columns([2, 3])
        with col1:
            # Add variable assignment here
            goal = st.text_area(
                label="Campaign Goal",
                placeholder="Enter your request", 
                height=68,
                key="goal_input"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
    
            if st.button("Generate", use_container_width=True, key="generate_button"):
                st.session_state.generated = True
                st.session_state.stop_requested = False
                st.session_state.tab_contents = {}
                st.rerun()
            if st.button("Stop", use_container_width=True, key="stop_button"):
                st.session_state.stop_requested = True
                st.rerun()
            if st.button("Back", use_container_width=True, key="back_button"):
                st.session_state.generated = False
                st.session_state.stop_requested = False
                st.session_state.tab_contents = {}
                st.session_state.active_tab = 0
                st.session_state.feedback = {}
                st.rerun()
    
        if st.session_state.generated and goal.strip():
            with st.spinner("Generating campaign..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
    
                try:
                    if not st.session_state.tab_contents:
                        status_text.text("Initializing campaign builder...")
                        progress_bar.progress(0.1)
                        if st.session_state.stop_requested:
                            status_text.text("⛔ Process stopped by user.")
                            st.session_state.generated = False
                            st.stop()
    
                        # Use run_autonomous_campaign_builder from main.py
                        campaign_workflow, initial_state = run_autonomous_campaign_builder(
                            goal=goal,
                            domain=st.session_state.selected_domain,
                            model_provider=st.session_state.selected_llm
                        )
                        
                        progress_bar.progress(0.2)
                        if st.session_state.stop_requested:
                            status_text.text("⛔ Process stopped by user.")
                            st.session_state.generated = False
                            st.stop()
    
                        # Display tabs first so they can be updated during processing
                        tab_labels = ["📊 Market Analysis", "🎯 Target Audience", "📈 Campaign Strategy", "✍️ Content", "🔬 Simulation", "📄 Final Report", "📬 Email Distribution"]
                        tabs = st.tabs(tab_labels)
    
                        # Get steps from the workflow
                        from agents import AGENT_REGISTRY
                        steps = list(AGENT_REGISTRY.keys())
                        total_steps = len(steps)
                        
                        # Process the workflow stream with dictionary access
                        for i, output in enumerate(campaign_workflow.stream(initial_state)):
                            if st.session_state.stop_requested:
                                status_text.text("⛔ Process stopped by user.")
                                st.session_state.generated = False
                                st.stop()
                                
                            node = list(output.keys())[0] if output else None
                            if node:
                                # Update progress - calculate based on current step
                                # Using a range from 0.2 to 0.9 for the workflow steps
                                current_progress = 0.2 + ((i + 1) / total_steps * 0.7)
                                progress_bar.progress(current_progress)
                                step_name = node.replace("_", " ").title()
                                status_text.text(f"Step {i+1}/{total_steps}: {step_name}")
                                
                                # Update state
                                state = output[node]
                                for key, value in state.items():
                                    setattr(initial_state, key, value)
                                
                                # Update corresponding tab
                                if node == "research_market_trends" and "market_analysis" in state:
                                    with tabs[0]:
                                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                                        render_section("Research Market Trends", state["market_analysis"], "market_analysis.pdf", node, "tab0")
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        st.session_state.tab_contents[0] = {"node": node, "content": state["market_analysis"]}
                                        st.session_state.active_tab = 0
                                elif node == "segment_audience" and "audience_segments" in state:
                                    with tabs[1]:
                                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                                        render_section("Segment Audience", state["audience_segments"], "audience_segments.pdf", node, "tab1")
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        st.session_state.tab_contents[1] = {"node": node, "content": state["audience_segments"]}
                                        st.session_state.active_tab = 1
                                elif node == "create_campaign_strategy" and "campaign_strategy" in state:
                                    with tabs[2]:
                                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                                        render_section("Create Campaign Strategy", state["campaign_strategy"], "campaign_strategy.pdf", node, "tab2")
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        st.session_state.tab_contents[2] = {"node": node, "content": state["campaign_strategy"]}
                                        st.session_state.active_tab = 2
                                elif node == "generate_content" and "campaign_content" in state:
                                    with tabs[3]:
                                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                                        render_section("Generate Content", state["campaign_content"], "campaign_content.pdf", node, "tab3")
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        st.session_state.tab_contents[3] = {"node": node, "content": state["campaign_content"]}
                                        st.session_state.active_tab = 3
                                elif node == "simulate_campaign" and "simulation_results" in state:
                                    with tabs[4]:
                                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                                        render_section("Simulate Campaign", state["simulation_results"], "simulation_results.pdf", node, "tab4")
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        st.session_state.tab_contents[4] = {"node": node, "content": state["simulation_results"]}
                                        st.session_state.active_tab = 4
                                elif node == "generate_final_report" and "final_report" in state:
                                    with tabs[5]:
                                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                                        render_section("Final Report", state["final_report"], "final_report.pdf", node, "tab5")
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        st.session_state.tab_contents[5] = {"node": node, "content": state["final_report"]}
                                        st.session_state.active_tab = 5
                                elif node == "send_campaign_emails" and "email_status" in state:
                                    with tabs[6]:
                                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                                        render_section("Email Distribution", state["email_status"], "email_status.pdf", node, "tab6")
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        st.session_state.tab_contents[6] = {"node": node, "content": state["email_status"]}
                                        st.session_state.active_tab = 6
                                    
                                    # Store the final state in session state for later use
                                    st.session_state.state = initial_state
                                    
                                    # Update progress to 100% only after all nodes are processed
                                    progress_bar.progress(1.0)
                                    status_text.text("Campaign generation complete!")
                                
                                # Add a small delay for UI updates
                                time.sleep(STEP_DELAY)
                            
                        # Add Email Distribution tab content
                        with tabs[6]:
                            st.subheader("Campaign Distribution")
                            
                            # Show who will receive emails
                            customer_data_path = os.path.join(script_dir, 'data', 'filtered_customers.csv')
                            try:
                                customer_data = pd.read_csv(customer_data_path)
                                valid_emails = customer_data[customer_data['email'].notna()]
                                
                                st.markdown("### Recipients")
                                st.dataframe(valid_emails[['full_name', 'email']].head(10),
                                           column_config={
                                               "full_name": "Name",
                                               "email": "Email"
                                           })
                                
                                if len(valid_emails) > 10:
                                    st.caption(f"Showing first 10 of {len(valid_emails)} recipients")
                                    
                                # Email templates section
                                st.markdown("### Email Templates")
                                if hasattr(st.session_state.state, 'email_templates') and st.session_state.state.email_templates:
                                    for i, template in enumerate(st.session_state.state.email_templates):
                                        with st.expander(f"Template {i+1}"):
                                            st.write(template.get('content', 'No content available'))
                                else:
                                    st.info("Email templates will be generated when you send the campaign")
                                    
                                # Manual email sending button
                                if st.button("📧 Send Campaign Emails", key="send_emails_button"):
                                    with st.spinner("Sending emails to customers..."):
                                        if st.session_state.state and hasattr(st.session_state.state, 'campaign_strategy'):
                                            # Call send_campaign_emails directly
                                            updated_state = send_campaign_emails(st.session_state.state)
                                            st.session_state.state = updated_state
                                            
                                            if updated_state.email_status and "Email Campaign Summary" in updated_state.email_status:
                                                st.success("✅ Emails sent successfully!")
                                                st.markdown(updated_state.email_status)
                                                
                                                # Show sent emails
                                                if updated_state.sent_emails:
                                                    st.markdown("### Sent Emails")
                                                    sent_df = pd.DataFrame(updated_state.sent_emails)
                                                    st.dataframe(sent_df[['name', 'email', 'subject']])
                                            else:
                                                st.error(updated_state.email_status or "Failed to send emails")
                                        else:
                                            st.error("Campaign not fully generated yet")
                                            
                            except FileNotFoundError:
                                st.warning("No customer data found. Please ensure filtered_customers.csv exists in the data directory.")

                except Exception as e:
                    logger.error(f"Workflow error: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"An error occurred: {str(e)}")
                    st.session_state.generated = False
                    st.stop()

        else:
            st.info("Enter your campaign goal and click 'Generate' to start.")
            st.markdown("""
            ## Sample Output
            The campaign builder will generate:
            1. **📊 Market Analysis** - Trends, opportunities, and competitive landscape
            2. **🎯 Target Audience** - Primary and secondary segments with messaging points
            3. **📈 Campaign Strategy** - Timeline, channels, budget, and KPIs
            4. **✍️ Content Examples** - Email, social media, and landing page content
            5. **🔬 Performance Simulation** - Projected results and optimization recommendations
            6. **📄 Final Report** - Complete campaign plan
            7. **📬 Email Distribution** - Send campaign emails to target customers
            """)
            
        # Feedback section
        st.markdown("---")
        st.subheader("Feedback")   

    except Exception as e:
        logger.error(f"Error in main application: {str(e)}\n{traceback.format_exc()}")
        st.error("An error occurred. Please check the logs.")

if __name__ == "__main__":
    main()