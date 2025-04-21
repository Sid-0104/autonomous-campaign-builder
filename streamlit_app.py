import streamlit as st
st.set_page_config(page_title="Autonomous Campaign Builder", page_icon=":scooter:", layout="wide")
import time
from io import BytesIO
import os
import re
import pandas as pd
import traceback
import base64
from dotenv import load_dotenv
from core.vector_db import initialize_vector_db, load_documents_by_domain
from core.state import CampaignState
from workflows.campaign_workflow import build_campaign_workflow
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import logging
from logging.handlers import RotatingFileHandler
from agents.send_emails import send_campaign_emails
from core.feedback import save_node_feedback
from agents import AGENT_REGISTRY
from datetime import datetime

import json

# Load section titles from JSON
with open(r"assets\\pdf_titles.json", "r") as f:
    SECTION_TITLES = json.load(f)

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
    # Replace curly quotes and em-dashes with ASCII equivalents
    content = re.sub(r"[‘’]", "'", content)
    content = re.sub(r"[“”]", '"', content)
    content = content.replace("–", "-").replace("—", "-")
    # Remove remaining non-ASCII characters
    return re.sub(r'[^\x00-\x7F]+', '', content)

# Custom PDF class with logo header, footer, and watermark
class PDFWithLogo(FPDF):
    def header(self):
        # Dynamically get the absolute path of the image
        base_path = os.path.dirname(__file__)  # directory where script is located
        logo_path = os.path.join(base_path, "assets", "infologo1.png")
        # Check if the logo file exists
        if os.path.exists(logo_path):
            self.image(logo_path, x=180, y=10, w=10)  # Adjust the logo size and position as needed
        else:
            print(f"Warning: Logo file not found at {logo_path}")
        self.set_y(40)  # Adjust the position after the header

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.set_text_color(100)  # Light gray
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_text = f"© 2025 InfoObjects Software Pvt Ltd | Generated on: {timestamp}"
        self.cell(0, 10, footer_text, align="C")


def clean_markdown(text):
    # Remove Markdown headers (e.g. ####, ###, ##, #)
    text = re.sub(r'#+\s*', '', text)
    # Remove bold and italic markdown (e.g., **text**, *text*)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Standardize bullet points
    text = re.sub(r'^\s*[-*]\s+', '- ', text, flags=re.MULTILINE)
    return text.strip()
# --- Your existing emoji stripper ---

def strip_emojis_and_unicode(text):
    # Remove emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Remove characters not supported by latin-1
    text = ''.join(c for c in text if ord(c) < 256)
    return text
# --- PDF Generator with cleaned content ---
def generate_pdf(title, content, node_name):
    section_title = SECTION_TITLES.get(node_name, node_name.replace("_", " ").title())

    clean_content = strip_emojis_and_unicode(content)
    clean_content = clean_markdown(clean_content)

    pdf = PDFWithLogo()
    pdf.add_page()

    pdf.set_font("Helvetica", style='B', size=16)
    pdf.cell(0, 10, "Autonomous Campaign Builder", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.set_text_color(80)
    pdf.cell(0, 10, f"{section_title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(6)

    pdf.set_font("Helvetica", size=12)

    lines = clean_content.split('\n')
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+\.\s+[A-Z \-]+:?$', line):
            pdf.set_font("Helvetica", style='B', size=12)
        else:
            pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, line[:100], new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf_output = BytesIO()
    pdf_output.write(pdf.output())
    pdf_output.seek(0)
    return pdf_output.getvalue()

# ===== Section Renderer + Feedback =====
def render_section(title, content, filename, node_name, key_prefix, state_obj):
    st.markdown(f"### {title}")
    st.write(content)
    

    # Download PDF
    st.download_button(
        label="📥 Download PDF",
        data=generate_pdf(title, content,node_name=node_name),
        file_name=filename,
        key=f"download_{key_prefix}_{node_name}"
    )

    # --- Feedback Section ---
    st.markdown("#### 🗳️ How would you rate this section?")
    feedback_key = f"{node_name}_recorded"

    col1, col2 = st.columns([1, 3.5], gap="small")
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
    st.markdown("#### 🔄 Want to regenerate the entire campaign?")
    if st.button("♻️ Regenerate", key=f"{key_prefix}_regen"):
        # Reset session state to force complete regeneration
        st.session_state.tab_contents = {}
        # Keep the goal and generated flag
        current_goal = st.session_state.state.goal if st.session_state.state else ""
        st.session_state.generated = True
        st.session_state.stop_requested = False
        # Force rerun to trigger complete regeneration
        st.rerun()

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


# Custom CSS
st.markdown("""
    <style>
        .st-emotion-cache-t1wise {
            width: 100%;
            padding: 0rem 6rem 10px 6rem;
            max-width: initial;
            min-width: auto;
        }
        .st-emotion-cache-1779v62{
            background-color: #ED9121;
            border: #ED9121;!important
        }
        .st-emotion-cache-1779v62:hover{
            background-color: #B9755A;
            border-color: #B9755A;!important
        }
        .st-emotion-cache-qsto9u:hover{
            color: #ED9121;
            border-color: #ED9121;!important
        }

        .stTabs [data-baseweb="tab-panel"] {
    background-color: var(--card-bg);
    padding: 15px;
    border-radius: 0 0 10px 10px;
    border: 1px solid var(--border-color);
    border-top: none;
    color: var(--text-color);

    max-height: 65vh;         /* Enable scrolling within fixed height */
    overflow-y: auto;         /* Activate vertical scrollbar */
    padding-right: 10px;      /* Space for scrollbar */
}

/* Custom Scrollbar Styling */
.stTabs [data-baseweb="tab-panel"]::-webkit-scrollbar {
    width: 8px;
}

.stTabs [data-baseweb="tab-panel"]::-webkit-scrollbar-thumb {
    background-color: #888;
    border-radius: 4px;
}

.stTabs [data-baseweb="tab-panel"]::-webkit-scrollbar-thumb:hover {
    background-color: #555;
}
        .st-emotion-cache-iyz50i:hover{
            color: #ED9121;
            border-color: #ED9121;!important
        }
        .st-emotion-cache-iyz50i:active{
            background-color: #ED9121;
        }
        .st-emotion-cache-iyz50i:focus:not(:active){
            color: #ED9121;
            border-color: #ED9121;
        }
        .st-emotion-cache-iyz50i{
            width: 40%;
            margin-left: 25%;
            justify-content: normal;
        }
        
        .st-emotion-cache-1bd5s7o{
             background-color: #ED9121;
            border: #ED9121;!important
        }
        .st-emotion-cache-1bd5s7o:hover{
            background-color: #B9755A;
            border-color: #B9755A;!important
        }
        .st-emotion-cache-m2qe7r:hover{
            color: #ED9121;
            border-color: #ED9121;!important
        }
        .st-emotion-cache-f03grt:hover{
            color: #ED9121;
            border-color: #ED9121;!important
        }
        .st-emotion-cache-f03grt:active{
            background-color: #ED9121;!important
        }
        .st-emotion-cache-f03grt:focus:not(:active){
            color: #ED9121;
            border-color: #ED9121;!important
        }
        .st-emotion-cache-f03grt{
            width: 40%;
            margin-left: 25%;
            justify-content: normal;
        }
        :root {--background-color: rgba(255, 255, 255, 1.0); --text-color: #1C1C1C; --card-bg: white; --border-color: #E6E6E6; --heading-color: #2C3E50; --shadow-color: rgba(0, 0, 0, 0.1); --tab-bg: #F1F5F9; --tab-selected-bg: white;}
        @media (prefers-color-scheme: dark) {:root {--background-color: rgba(30, 30, 30, 0.95); --text-color: #F1F1F1; --card-bg: #2D2D2D; --border-color: #444444; --heading-color: #8AB4F8; --shadow-color: rgba(0, 0, 0, 0.3); --tab-bg: #383838; --tab-selected-bg: #2D2D2D;}}
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

def render_email_tab():
    """Function to render the email tab content"""
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    st.subheader("📬 Campaign Distribution")
    customer_data_path = os.path.join(script_dir, 'data', 'filtered_customers.csv')
    if os.path.exists(customer_data_path):
        customer_df = pd.read_csv(customer_data_path)
        email_df = customer_df[customer_df["email"].notna()]
        if not email_df.empty:
            st.markdown("### Recipients")
            st.dataframe(email_df[['full_name', 'email']].head(10))
            if len(email_df) > 10:
                st.caption(f"Showing first 10 of {len(email_df)} recipients")
        else:
            st.warning("No valid email addresses found.")

    # Show email templates (if previously generated)
    st.markdown("### Email Templates")
    if st.session_state.state and hasattr(st.session_state.state, 'email_templates') and st.session_state.state.email_templates:
        for i, template in enumerate(st.session_state.state.email_templates):
            with st.expander(f"Previous Template {i+1}"):
                st.markdown(template.get('content', 'No content available'))
    else:
        st.info("Templates will be generated once emails are sent.")

    # Modified version of the code in render_email_tab()
    if st.button("📧 Send Campaign Emails", key="send_emails_btn"):
        with st.spinner("Sending emails..."):
            if st.session_state.state and hasattr(st.session_state.state, 'campaign_strategy'):
                # Fix: Capture just the updated state - don't try to unpack multiple values
                updated_state = send_campaign_emails(st.session_state.state)
                
                # Update session state
                st.session_state.state = updated_state

                if hasattr(updated_state, "email_status"):
                    # Parse the failed_count from the email_status if needed
                    failed_count = 0
                    if "Failed to send:" in updated_state.email_status:
                        try:
                            failed_part = updated_state.email_status.split("Failed to send:")[1].strip()
                            failed_count = int(failed_part.split()[0])
                        except:
                            pass
                    
                    if failed_count != 0:
                        st.error("❌" + updated_state.email_status)
                    else:
                        st.success("✅ Emails sent successfully!" + updated_state.email_status)
            else:
                st.error("❌ Campaign data not available. Please generate a campaign first.")
    st.markdown("</div>", unsafe_allow_html=True)
def handle_zombie_logic():
    if st.session_state.get("zombie_trigger", False):
        # Run zombie logic only if all sections are generated
        if len(st.session_state.tab_contents) >= 7:
            logger.info("🧟 Running zombie logic after campaign completion")
            # 🔄 Add your zombie behavior here (e.g., auto-tab switch, preloading)
            st.toast("Zombie button triggered!")  # for testing
            st.session_state.zombie_trigger = False

def render_footer():
    st.markdown(
        """<hr style="margin-top: 50px;"/>
        <div style="text-align: center; color: gray; font-size: 0.9em;">
            © 2025 InfoObjects Software Pvt Ltd
        </div>""",
        unsafe_allow_html=True
    )

def main():
    logger.info("Starting Streamlit application")
    handle_zombie_logic()
    

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
            render_footer()

        # Input
        col1, col2 = st.columns([1, 15], gap="small")
        image_path = os.path.join(script_dir, "assets", "info.png")
        with col1:
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode()
                st.markdown(
                    f"""
                    <div style='display: flex; align-items: center; height: 100px;'>
                        <img src='data:image/png;base64,{encoded}' width='60'>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.write(":robot_face:")
        with col2:
            st.markdown(
                """
                <div style='display: flex; align-items: center; height: 100px;'>
                    <h1 style='font-size: 32px; color: #4B8BBE; margin: 0; line-height: 1.2;'>
                        Autonomous<br>Campaign Builder
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("---")
        col1, col2 = st.columns([6, 3])
        with col1:
            DEFAULT_PROMPTS = {
                    "automotives": "Boost Q2 SUV sales in the Western region by 15%",
                    "healthcare": "Increase patient enrollment in our preventative care program by 25% in Q3",
                    "powerenergy": "Increase residential solar panel installations by 35% in the Southern region this summer"
                    }
            # Preserve the goal input between regenerations
            default_goal = DEFAULT_PROMPTS.get(st.session_state.selected_domain, "")
            # Add variable assignment here
            goal = st.text_area(
                label="Campaign Goal",
                placeholder="Enter your request",
                height=180,
                key="goal_input",
                value=st.session_state.get("goal_input", default_goal)
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <style>
                .stButton > button {
                    display: inline-flex !important;
                    align-items: center !important;
                    gap: 8px !important; /* space between emoji and text */
                    font-size: 16px !important;
                    padding: 8px 44px !important;
                    min-width: 200px !important; /* Ensures all buttons align from same start */
                }
                </style>
            """, unsafe_allow_html=True)
            generate_clicked = st.button(":arrows_counterclockwise:  Generate", use_container_width=True, key="generate_button")
            stop_clicked = st.button(":black_square_for_stop:   Stop", use_container_width=True, key="stop_button")
            back_clicked = st.button(":back:  Back", use_container_width=True, key="back_button")
            if generate_clicked:
                st.session_state.generated = True
                st.session_state.stop_requested = False
                st.session_state.tab_contents = {}
                # st.session_state.show_tabs = False 
                st.rerun()
            if stop_clicked:
                st.session_state.stop_requested = True
                st.rerun()
            if back_clicked:
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
                with tab:
                    # Check if this tab has content in session state
                    if tab_index in st.session_state.tab_contents:
                        content_info = st.session_state.tab_contents[tab_index]
                        node = content_info["node"]
                        content = content_info["content"]
                        section_title = node.replace("_", " ").title()
                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                        render_section(section_title, content, f"{node}.pdf", node, f"tab{tab_index}", st.session_state.state)
                        st.markdown("</div>", unsafe_allow_html=True)
                    # Special handling for email tab if not populated yet
                    elif tab_index == 6 and 6 not in st.session_state.tab_contents:
                        render_email_tab()
            
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
                    time.sleep(0.5)
                    status_text.text("Campaign generation complete!")
                    time.sleep(2.0)
                    st.session_state.zombie_trigger = True  # 🔥 Trigger zombie on next run
                    st.rerun()
                    
                    # Email tab shown after workflow completion
                    with tabs[6]:
                        render_email_tab()

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
    