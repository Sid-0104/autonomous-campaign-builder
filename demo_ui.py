import streamlit as st
import time
import os
import re
import pandas as pd
import traceback
import uuid
from datetime import datetime
from dotenv import load_dotenv
from core.vector_db import initialize_vector_db, load_documents_by_domain
from core.state import CampaignState
from workflows.campaign_workflow import build_campaign_workflow
from fpdf import FPDF
import logging
from logging.handlers import RotatingFileHandler
from agents.send_emails import send_campaign_emails
from utils.feedback import save_node_feedback
from agents import AGENT_REGISTRY

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

# Configuration
STEP_DELAY = int(os.environ.get("STEP_DELAY", 5))

# Default prompts for each industry
DEFAULT_PROMPTS = {
    "automotives": "Boost Q2 SUV sales in the Western region by 15%",
    "healthcare": "Increase patient enrollment in our preventative care program by 25% in Q3",
    "powerenergy": "Increase residential solar panel installations by 35% in the Southern region this summer"
}

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
# KEPT FROM feedback_regenerate_updated: This preserves your working feedback + regeneration functionality
def render_section(title, content, filename, node_name, key_prefix, state_obj):
    import uuid
    unique_id = str(uuid.uuid4())[:8]  # short unique suffix

    st.markdown(f"### {title}")
    st.write(content)

    # Download PDF
    st.download_button(
        label="📥 Download PDF",
        data=generate_pdf(title, content),
        file_name=filename,
        key=f"download_{key_prefix}_{node_name}_{unique_id}"
    )

    # --- Feedback Section ---
    st.markdown("#### 🗳️ How would you rate this section?")
    feedback_key = f"{node_name}_recorded"

    col1, col2 = st.columns(2)
    if feedback_key not in st.session_state:
        if col1.button("👍", key=f"{key_prefix}_positive_{unique_id}"):
            msg = save_node_feedback(node_name, "Positive Feedback")
            st.success(msg)
            st.session_state[feedback_key] = True

        if col2.button("👎", key=f"{key_prefix}_negative_{unique_id}"):
            msg = save_node_feedback(node_name, "Negative Feedback")
            st.success(msg)
            st.session_state[feedback_key] = True
    else:
        st.info("✅ Feedback already recorded.")

    # --- Regenerate Section ---
    st.markdown("#### 🔄 Want to improve this section?")
    if st.button("♻️ Regenerate", key=f"{key_prefix}_regen_{unique_id}"):
        with st.spinner("Regenerating..."):
            new_state = AGENT_REGISTRY[node_name](state_obj)
            node_to_content = {
                "research_market_trends": "market_analysis",
                "segment_audience": "audience_segments",
                "create_campaign_strategy": "campaign_strategy",
                "generate_content": "campaign_content",
                "simulate_campaign": "simulation_results",
                "generate_final_report": "final_report",
                "send_campaign_emails": "email_status"
            }
            if node_name in node_to_content:
                content_key = node_to_content[node_name]
                if content_key in new_state:
                    st.session_state.tab_contents[st.session_state.active_tab]["content"] = new_state[content_key]
                    st.session_state.state = new_state  # persist
                    st.success("✅ Section regenerated!")

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
                logger.info("Starting vector DB initialization...")
                vector_db = cached_initialize_vector_db(domain=domain, model_provider=model_provider)
                logger.info("Vector DB initialized successfully")
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

        logger.info("Loading documents by domain...")
        sales_data, _, _, _ = load_documents_by_domain(domain)
        logger.info(f"Loaded {len(sales_data)} sales records")
        
        logger.info("Building campaign workflow...")
        campaign_workflow = build_campaign_workflow()
        logger.info("Campaign workflow built successfully")
        
        # Update model provider in state if changed
        logger.info("Creating initial state...")
        initial_state = CampaignState(
            goal=goal,
            vector_db=vector_db,
            sales_data=sales_data,
            selected_domain=domain,
            selected_llm=model_provider if model_provider == "gemini" else "openai"
        )
        logger.info("Initial state created successfully")
        
        # Check API keys before returning
        logger.info("Checking API keys...")
        # Check OpenAI API key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        else:
            logger.info("OPENAI_API_KEY found in environment")
            
        # Check Tavily API key
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_key:
            logger.warning("TAVILY_API_KEY not found in environment variables")
        else:
            logger.info("TAVILY_API_KEY found in environment")
            
        # Check Gemini API key
        gemini_key = os.environ.get("GOOGLE_API_KEY")
        if not gemini_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables")
        else:
            logger.info("GOOGLE_API_KEY found in environment")
        
        logger.info("Campaign builder initialization complete, returning workflow and state")
        return campaign_workflow, initial_state
        
    except Exception as e:
        logger.error(f"Failed to initialize campaign builder: {str(e)}\n{traceback.format_exc()}")
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
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False

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
        # Sidebar
        with st.sidebar:
            
            
            st.subheader("Switch LLM")
            for llm in ["Gemini", "OpenAI"]:
                if st.button(llm, type="primary" if st.session_state.selected_llm == llm.lower() else "secondary", use_container_width=True):
                    st.session_state.selected_llm = llm.lower()
                    st.rerun()
            st.markdown(f"**Using: {st.session_state.selected_llm.upper()}**")
            st.subheader("Select Industry")
            for industry in ["Automotives", "Healthcare", "Powerenergy"]:
                domain_value = industry.lower()
                if st.button(industry, type="primary" if st.session_state.selected_domain == domain_value else "secondary", use_container_width=True):
                    st.session_state.selected_domain = domain_value
                    # Set the default prompt for the selected industry
                    if "goal_input" not in st.session_state or not st.session_state.goal_input:
                        st.session_state.goal_input = DEFAULT_PROMPTS.get(domain_value, "")
                    st.rerun()
            st.markdown(f"**Industry: {st.session_state.selected_domain.upper()}**")

        # Input
        col1, col2 = st.columns([1, 3.5], gap="small")
        with col1:
            image_path = os.path.join(script_dir, "assets", "info.png")
            if os.path.exists(image_path):
                st.image(image_path, width=50, use_container_width=False)
            else:
                st.write("🤖")
        with col2:
            st.markdown("<h1 style='font-size: 26px; color: #4B8BBE;'>Autonomous<br>Campaign Builder</h1>", unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns([2, 3])
        with col1:
            # Use the default prompt for the selected industry if no input yet
            default_goal = DEFAULT_PROMPTS.get(st.session_state.selected_domain, "")
            
            # Add variable assignment here
            goal = st.text_area(
                label="Campaign Goal",
                placeholder="Enter your request", 
                height=68,
                key="goal_input",
                value=st.session_state.get("goal_input", default_goal)
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate", use_container_width=True, key="generate_button"):
                st.session_state.generated = True
                st.session_state.generation_complete = False
                st.session_state.stop_requested = False
                st.session_state.tab_contents = {}
                st.rerun()
            if st.button("Stop", use_container_width=True, key="stop_button"):
                st.session_state.stop_requested = True
                st.rerun()
            if st.button("Back", use_container_width=True, key="back_button"):
                st.session_state.generated = False
                st.session_state.generation_complete = False
                st.session_state.stop_requested = False
                st.session_state.tab_contents = {}
                st.session_state.active_tab = 0
                st.session_state.feedback = {}
                st.rerun()

        # ✅ If feedback or reloaded state already exists, show only once
        if st.session_state.generated and st.session_state.tab_contents:
            tab_labels = ["📊 Market Analysis", "🎯 Target Audience", "📈 Campaign Strategy", "✍️ Content", "🔬 Simulation", "📄 Final Report", "📬 Email Distribution"]
            tabs = st.tabs(tab_labels)
            for tab_index, tab in enumerate(tabs):
                if tab_index in st.session_state.tab_contents:
                    with tab:
                        content_info = st.session_state.tab_contents[tab_index]
                        node = content_info["node"]
                        content = content_info["content"]
                        section_title = node.replace("_", " ").title()
                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                        render_section(section_title, content, f"{node}.pdf", node, f"tab{tab_index}", st.session_state.state)
                        st.markdown("</div>", unsafe_allow_html=True)
            return  # 🛑 Stop here — no need to regenerate

        # ✅ Start fresh generation if goal is set
        if st.session_state.generated and goal.strip():
            # Display tabs first so they can be updated during processing
            tab_labels = ["📊 Market Analysis", "🎯 Target Audience", "📈 Campaign Strategy", "✍️ Content", "🔬 Simulation", "📄 Final Report", "📬 Email Distribution"]
            tabs = st.tabs(tab_labels)
            
            # Only start or continue processing if generation is not complete
            if not st.session_state.generation_complete:
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
                                    
                                    # Update corresponding tab - content is saved to session state
                                    # but the rendering is done separately to avoid conflicts with feedback
                                    if node == "research_market_trends" and "market_analysis" in state:
                                        st.session_state.tab_contents[0] = {"node": node, "content": state["market_analysis"]}
                                        st.session_state.active_tab = 0
                                    elif node == "segment_audience" and "audience_segments" in state:
                                        st.session_state.tab_contents[1] = {"node": node, "content": state["audience_segments"]}
                                        st.session_state.active_tab = 1
                                    elif node == "create_campaign_strategy" and "campaign_strategy" in state:
                                        st.session_state.tab_contents[2] = {"node": node, "content": state["campaign_strategy"]}
                                        st.session_state.active_tab = 2
                                    elif node == "generate_content" and "campaign_content" in state:
                                        st.session_state.tab_contents[3] = {"node": node, "content": state["campaign_content"]}
                                        st.session_state.active_tab = 3
                                    elif node == "simulate_campaign" and "simulation_results" in state:
                                        st.session_state.tab_contents[4] = {"node": node, "content": state["simulation_results"]}
                                        st.session_state.active_tab = 4
                                    elif node == "generate_final_report" and "final_report" in state:
                                        st.session_state.tab_contents[5] = {"node": node, "content": state["final_report"]}
                                        st.session_state.active_tab = 5
                                    elif node == "send_campaign_emails" and "email_status" in state:
                                        st.session_state.tab_contents[6] = {"node": node, "content": state["email_status"]}
                                        st.session_state.active_tab = 6
                                        
                                        # Store the final state in session state for later use
                                        st.session_state.state = initial_state
                                        st.session_state.generation_complete = True
                                        
                                        # Update progress to 100% only after all nodes are processed
                                        progress_bar.progress(1.0)
                                        status_text.text("Campaign generation complete!")
                                    
                                    # Render the current tab content after updating session state
                                    # This ensures feedback doesn't interrupt the current step
                                    with tabs[st.session_state.active_tab]:
                                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                                        content_data = st.session_state.tab_contents[st.session_state.active_tab]
                                        render_section(
                                            title=content_data["node"].replace("_", " ").title(),
                                            content=content_data["content"],
                                            filename=f"{content_data['node']}.pdf",
                                            node_name=content_data["node"],
                                            key_prefix=f"tab{st.session_state.active_tab}",
                                            state_obj=initial_state  # Pass the state object for regeneration
                                        )
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    
                                    # Add a small delay for UI updates
                                    time.sleep(STEP_DELAY)
                    
                    except Exception as e:
                        logger.error(f"Workflow error: {str(e)}\n{traceback.format_exc()}")
                        st.error(f"An error occurred: {str(e)}")
                        st.session_state.generated = False
                        st.stop()
            
            # Display previously generated content when returning to the page or after providing feedback
            if st.session_state.tab_contents:
                for tab_index, content_data in st.session_state.tab_contents.items():
                    tab_index = int(tab_index)
                    with tabs[tab_index]:
                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                        render_section(
                            title=content_data["node"].replace("_", " ").title(), 
                            content=content_data["content"], 
                            filename=f"{content_data['node']}.pdf", 
                            node_name=content_data["node"], 
                            key_prefix=f"tab{tab_index}",
                            state_obj=st.session_state.state  # Pass the state object for regeneration
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Add Email Distribution tab content if generation is complete
                if st.session_state.generation_complete and st.session_state.state:
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

        else:
            st.info("Enter your campaign goal and click 'Generate' to start.")
            st.markdown("""
            ## Sample Output
            The campaign builder will generate:
            1. **📊 Market Analysis**  
            2. **🎯 Target Audience**  
            3. **📈 Campaign Strategy**  
            4. **✍️ Content Examples**  
            5. **🔬 Performance Simulation**  
            6. **📄 Final Report**  
            7. **📬 Email Distribution**
            """)

    except Exception as e:
        logger.error(f"Error in main application: {str(e)}\n{traceback.format_exc()}")
        st.error("An error occurred. Please check the logs.")

if __name__ == "__main__":
    main()
    st.markdown("")