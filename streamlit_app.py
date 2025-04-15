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
from core.feedback import save_node_feedback
from agents import AGENT_REGISTRY

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
def render_section(title, content, filename, node_name, key_prefix, state_obj):
    st.markdown(f"### {title}")
    st.write(content)

    # Download PDF
    st.download_button(
        label="📥 Download PDF",
        data=generate_pdf(title, content),
        file_name=filename,
        key=f"download_{key_prefix}_{node_name}"
    )

    # --- Feedback Section ---
    st.markdown("#### 🗳️ How would you rate this section?")
    feedback_key = f"{node_name}_recorded"

    col1, col2 = st.columns(2)
    if feedback_key not in st.session_state:
        if col1.button("👍", key=f"{key_prefix}_positive"):
            msg = save_node_feedback(node_name, "Positive Feedback")
            st.success(msg)
            st.session_state[feedback_key] = True

        if col2.button("👎", key=f"{key_prefix}_negative"):
            msg = save_node_feedback(node_name, "Negative Feedback")
            st.success(msg)
            st.session_state[feedback_key] = True
    else:
        st.info("✅ Feedback already recorded.")

    # --- Regenerate Section ---
    st.markdown("#### 🔄 Want to improve this section?")
    if st.button("♻️ Regenerate", key=f"{key_prefix}_regen"):
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
            goal = st.text_area("Campaign Goal", placeholder="Enter your request", height=68, key="goal_input")
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
            with st.spinner("Generating campaign..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("Initializing campaign builder...")
                    progress_bar.progress(0.1)

                    campaign_workflow, initial_state = run_autonomous_campaign_builder(
                        goal=goal,
                        domain=st.session_state.selected_domain,
                        model_provider=st.session_state.selected_llm
                    )
                    progress_bar.progress(0.2)

                    from agents import AGENT_REGISTRY
                    steps = list(AGENT_REGISTRY.keys())
                    total_steps = len(steps)

                    tab_labels = ["📊 Market Analysis", "🎯 Target Audience", "📈 Campaign Strategy", "✍️ Content", "🔬 Simulation", "📄 Final Report", "📬 Email Distribution"]
                    tabs = st.tabs(tab_labels)

                    index_map = {
                        "research_market_trends": (0, "market_analysis"),
                        "segment_audience": (1, "audience_segments"),
                        "create_campaign_strategy": (2, "campaign_strategy"),
                        "generate_content": (3, "campaign_content"),
                        "simulate_campaign": (4, "simulation_results"),
                        "generate_final_report": (5, "final_report"),
                        "send_campaign_emails": (6, "email_status")
                    }

                    for i, output in enumerate(campaign_workflow.stream(initial_state)):
                        if st.session_state.stop_requested:
                            st.session_state.generated = False
                            st.stop()

                        node = list(output.keys())[0] if output else None
                        if node:
                            progress = 0.2 + ((i + 1) / total_steps * 0.7)
                            progress_bar.progress(progress)
                            status_text.text(f"Step {i+1}/{total_steps}: {node.replace('_', ' ').title()}")

                            state = output[node]
                            for key, value in state.items():
                                setattr(initial_state, key, value)

                            if node in index_map:
                                tab_index, content_key = index_map[node]
                                if content_key in state:
                                    with tabs[tab_index]:
                                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                                        render_section(node.replace("_", " ").title(), state[content_key], f"{node}.pdf", node, f"tab{tab_index}", initial_state)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    st.session_state.tab_contents[tab_index] = {
                                        "node": node,
                                        "content": state[content_key]
                                    }
                                    st.session_state.active_tab = tab_index

                    st.session_state.state = initial_state
                    progress_bar.progress(1.0)
                    status_text.text("Campaign generation complete!")
                    with tabs[6]:
                        st.subheader("📬 Campaign Distribution")
                        customer_data_path = os.path.join(script_dir, 'data', 'filtered_customers.csv')
                        if os.path.exists(customer_data_path):
                            customer_df = pd.read_csv(customer_data_path)
                            email_df = customer_df[customer_df["email"].notna()]
                            if not email_df.empty:
                                st.markdown("### Recipients")
                                st.dataframe(email_df[['full_name', 'email']].head(10))
                                if len(email_df) > 7:
                                    st.caption(f"Showing first 10 of {len(email_df)} recipients")
                                else:
                                    st.warning("No valid email addresses found.")

                        # Show email templates (if previously generated)
                        st.markdown("### Email Templates")
                        if hasattr(initial_state, 'email_templates') and initial_state.email_templates:
                            for i, template in enumerate(initial_state.email_templates):
                                with st.expander(f"Template {i+1}"):
                                    st.markdown(template.get('content', 'No content available'))

                        else:
                            st.info("Templates will be generated once emails are sent.")

                        if st.button("📧 Send Campaign Emails", key="send_emails_btn"):
                            with st.spinner("Sending emails..."):
                                # from agents.send_emails import send_campaign_emails
                                # updated_state = send_campaign_emails(initial_state)
                                # st.session_state.state = updated_state
                                if st.session_state.state and hasattr(st.session_state.state, 'campaign_strategy'):
                                    updated_state = send_campaign_emails(st.session_state.state)
                                    st.session_state.state = updated_state

                                if hasattr(updated_state, "email_status") and "Email Campaign Summary" in updated_state.email_status:
                                    st.success("✅ Emails sent successfully!")
                                    st.markdown(updated_state.email_status)
                                    # ✅ Update Tab Content after sending emails
                                    st.session_state.tab_contents[6] = {
                                        "node": "send_campaign_emails",
                                        "content": updated_state.email_status
                                        }
                                    st.session_state.active_tab = 6
                                else:
                                    st.error("❌ Email sending failed.")
                                    st.markdown(updated_state.email_status or "No additional info available.")



                    

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
    
