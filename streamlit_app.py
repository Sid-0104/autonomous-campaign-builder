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
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- PDF Generator ---
import re
import os
from fpdf import FPDF

# Function to clean text by stripping emojis and replacing certain characters
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
        logo_path = os.path.join(base_path, "assets", "info.png")
        
        # Check if the logo file exists
        if os.path.exists(logo_path):
            self.image(logo_path, x=180, y=10, w=10)  # Adjust the logo size and position as needed
        else:
            print(f"Warning: Logo file not found at {logo_path}")
        
        self.set_y(40)  # Adjust the position after the header

from io import BytesIO

def clean_markdown(text):
    # Remove Markdown headers (e.g. ####, ###, ##, #)
    text = re.sub(r'#+\s*', '', text)

    # Remove bold and italic markdown (e.g., **text**, *text*)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Standardize bullet points
    text = re.sub(r'^\s*[-*]\s+', '- ', text, flags=re.MULTILINE)

    return text.strip()

# --- PDF Generator with cleaned content ---
def generate_pdf(section_title, content):
    clean_content = strip_emojis(content)
    clean_content = clean_markdown(clean_content)

    # Use the custom class with the logo (assuming it's already defined)
    pdf = PDFWithLogo()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, f"{section_title}\n\n{clean_content}")

    # Return PDF as bytes for download
    try:
        pdf_bytes = pdf.output(dest='S').encode('latin1')  # Ensure 'latin1' encoding
        return pdf_bytes
    except UnicodeEncodeError:
        # Handle potential encoding issues and log them
        print("Unicode encoding error: Some characters are unsupported.")
        clean_content = clean_content.encode('latin1', 'ignore').decode('latin1')  # Ignore unsupported characters
        pdf = PDFWithLogo()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, f"{section_title}\n\n{clean_content}")
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return pdf_bytes


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

    st.markdown("**Give Your Feedback**")
    feedback_key = f"{key_prefix}_feedback"

    # Button container
    with st.container():
        st.markdown('<div class="feedback-btns">', unsafe_allow_html=True)
        # Use container instead of columns to control layout
        if feedback_key not in st.session_state:
            # Place buttons side by side with minimal gap
            col1, col2 = st.columns([1, 1])  # Keep columns but minimize their impact
            with col1:
                if st.button("👍", key=f"{key_prefix}_positive"):
                    msg = save_node_feedback(node_name, "Positive Feedback")
                    feedback_placeholder = st.empty()
                    feedback_placeholder.success(msg)
                    time.sleep(2)
                    feedback_placeholder.empty()
                    st.session_state[feedback_key] = True
            with col2:
                if st.button("👎", key=f"{key_prefix}_negative"):
                    msg = save_node_feedback(node_name, "Negative Feedback")
                    feedback_placeholder = st.empty()
                    feedback_placeholder.success(msg)
                    time.sleep(2)
                    feedback_placeholder.empty()
                    st.session_state[feedback_key] = True
        else:
            st.info("✅ Feedback already recorded.")
        st.markdown('</div>', unsafe_allow_html=True)




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
        :root {
            --background-color: rgba(255, 255, 255, 1.0);
            --text-color: #1c1c1c;
            --card-bg: white;
            --border-color: #e6e6e6;
            --heading-color: #2c3e50;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --tab-bg: #f1f5f9;
            --tab-selected-bg: white;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --background-color: rgba(30, 30, 30, 0.95);
                --text-color: #f1f1f1;
                --card-bg: #2d2d2d;
                --border-color: #444444;
                --heading-color: #8ab4f8;
                --shadow-color: rgba(0, 0, 0, 0.3);
                --tab-bg: #383838;
                --tab-selected-bg: #2d2d2d;
            }
        }
        body {
            background: linear-gradient(var(--background-color), var(--background-color)),
                        url('https://images.unsplash.com/photo-1551836022-4c4c79ecde16?auto=format&fit=crop&w=1400&q=80') no-repeat center center fixed;
            background-size: cover;
            color: var(--text-color);
            font-family: 'Segoe UI', sans-serif;
        }
        #MainMenu, footer, header, .stDeployButton { visibility: hidden; }
        .stApp { background-color: var(--background-color); padding: 0 !important; border-radius: 0 !important; box-shadow: none !important; max-width: 100 !important; margin: 0 !important; }
        .stTabs [role="tab"] { font-size: 16px; padding: 10px 20px; margin-right: 5px; border: 1px solid var(--border-color); background-color: var(--tab-bg); border-radius: 6px 6px 0 0; color: var(--text-color); }
        .stTabs [aria-selected="true"] { background-color: var(--tab-selected-bg); border-bottom: none; font-weight: bold; }
        .tab-content { animation: fadein 0.6s ease-in; background-color: var(--card-bg); padding: 20px; border-radius: 0 0 10px 10px; box-shadow: 0 2px 4px var(--shadow-color); margin-top: -1px; border: 1px solid var(--border-color); border-top: none; }
        .stTextArea textarea { border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; font-size: 16px; background-color: var(--card-bg); color: var(--text-color); }
        .stButton button { border-radius: 8px; font-weight: 500; transition: all 0.3s ease; height: 40px !important; margin: 0 !important; padding: 0 15px !important; }
        .stButton button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px var(--shadow-color); }
        .stButton > button[kind="primary"] { background-color: #ED9121 !important; color: white !important; border: none !important; }
        .stButton > button[kind="primary"]:hover { background-color: #e07b00 !important; color: white !important; }
        h1 { color: var(--heading-color); font-weight: 600; font-size: 20px; }
        h2, h3 { color: var(--heading-color); font-weight: 600; }
        .stTabs [data-baseweb="tab-panel"] { background-color: var(--card-bg); padding: 15px; border-radius: 0 0 10px 10px; border: 1px solid var(--border-color); border-top: none; color: var(--text-color); }
        .stProgress > div > div { background-color: #4CAF50; }
        @keyframes fadein { from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: translateY(0);} }
        .feedback-header { font-size: 14px !important; }
        .button-row { display: flex; justify-content: flex-start; gap: 10px; margin-top: 0 !important; align-items: center; }
        .button-row .stButton { flex: 0 0 auto; margin-top: 0 !important; }
        .text-area-container { display: flex; flex-direction: column; align-items: flex-start; }
    </style>
""", unsafe_allow_html=True)


logger = logging.getLogger(__name__)

def main():
    # Initialize session state variable if it doesn't exist
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = False  # Default value

    logger.info(f"Starting application - sidebar_collapsed = {st.session_state.sidebar_collapsed}")

    try:
        # Display heading unconditionally on the main page
        with st.container():
            st.markdown("<h1 style='font-size: 20px;'>Autonomous Campaign Builder</h1>", unsafe_allow_html=True)
        logger.debug("Main heading displayed")

        # Sidebar configuration
        with st.sidebar:
            logger.debug(f"Sidebar initialized - sidebar_collapsed = {st.session_state.sidebar_collapsed}")
            col1, col2 = st.columns([1, 3], gap="small")
            with col1:
                image_path = os.path.join(script_dir, "assets", "info.png")
                if os.path.exists(image_path):
                    st.image(image_path, width=50, use_container_width=False)
                else:
                    st.write("🤖")
            with col2:
                st.markdown(
                    """
                    <h1 style='font-size: 24px; color: #4B8BBE; margin: 0; padding: 0; line-height: 1; vertical-align: middle;'>Autonomous Campaign Builder</h1>
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
            for industry in ["Automotives", "Healthcare", "Powerenergy"]:
                domain_value = industry.lower()
                if st.button(industry, type="primary" if st.session_state.selected_domain == domain_value else "secondary", use_container_width=True):
                    st.session_state.selected_domain = domain_value
                    st.rerun()
            st.markdown(f"**Industry: {st.session_state.selected_domain.upper()}**")

        col1, col2 = st.columns([2, 3])

        with col1:
           goal = st.text_area("Campaign Goal", placeholder="Enter your campaign request", height=120, label_visibility="hidden")
        with col2:
            st.markdown('<div class="horizontal-buttons">', unsafe_allow_html=True)

            if st.button("➡️", key="go_button"):
                if goal.strip():
                    st.session_state.generated = True
                    st.session_state.stop_requested = False
                    st.session_state.tab_contents = {}
                    # Removed sidebar_collapsed = False to preserve heading
                    logger.debug("Generate clicked")
                    time.sleep(0.2)
                    st.rerun()
                else:
                    st.error("Please enter a request before generating.")

            if st.button("🚫", key="stop_button"):
                st.session_state.stop_requested = True
                st.rerun()

            if st.button("↩️", key="back_button"):
                st.session_state.generated = False
                st.session_state.stop_requested = False
                st.session_state.tab_contents = {}
                st.session_state.active_tab = 0
                st.session_state.feedback = {}
                st.session_state.sidebar_collapsed = True
                logger.debug("Back button clicked, setting sidebar_collapsed to True")
                time.sleep(0.2)
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
        # ✅ Render from existing session state if available
        if st.session_state.generated:
            tab_labels = ["📊 Market Analysis", "🎯 Target Audience", "📈 Campaign Strategy", "✍️ Content", "🔬 Simulation", "📄 Final Report", "📬 Email Distribution"]
            tabs = st.tabs(tab_labels)

            # 🧠 Just re-render tabs if already generated (e.g., feedback clicked)
            if st.session_state.tab_contents:
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
                return  # ✅ prevent rerun of workflow

        if st.session_state.generated and goal.strip():
            with st.spinner("Generating campaign..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    if not st.session_state.tab_contents:
                        status_text.text("Initializing campaign builder...")
                        progress_bar.progress(0.1)
                        campaign_workflow, initial_state = run_autonomous_campaign_builder(
                            goal=goal,
                            domain=st.session_state.selected_domain,
                            model_provider=st.session_state.selected_llm
                        )
                        progress_bar.progress(0.2)
                        if st.session_state.stop_requested:
                            logger.info("Stop requested before workflow start")
                            status_text.text("⛔ Process stopped by user.")
                            st.session_state.generated = False
                            st.stop()

                        node_to_tab = {
                            "research_market_trends": (0, "market_analysis"),
                            "segment_audience": (1, "audience_segments"),
                            "create_campaign_strategy": (2, "campaign_strategy"),
                            "generate_content": (3, "campaign_content"),
                            "simulate_campaign": (4, "simulation_results"),
                            "generate_final_report": (5, "final_report"),
                            "send_campaign_emails": (6, "email_status")
                        }
                        total_steps = len(node_to_tab)
                        step_counter = 0

                        for i, output in enumerate(campaign_workflow.stream(initial_state)):
                            logger.info(f"Checking stop at step {step_counter + 1}/{total_steps}")
                            if st.session_state.stop_requested:
                                logger.info("Stop requested during workflow")
                                status_text.text("⛔ Process stopped by user.")
                                st.session_state.generated = False
                                st.stop()
                            node = list(output.keys())[0] if output else None
                            if node:
                                step_counter += 1
                                progress = min(step_counter / total_steps, 0.9)
                                progress_bar.progress(progress)
                                status_text.text(f"Step {step_counter}/{total_steps}: Processing {node}...")
                                time.sleep(STEP_DELAY / 2)
                                state = output[node]
                                for key, value in state.items():
                                    setattr(initial_state, key, value)
                                if node in node_to_tab:
                                    tab_index, content_key = node_to_tab[node]
                                    if content_key in state:
                                        st.session_state.tab_contents[tab_index] = {"node": node, "content": state[content_key]}
                                        st.session_state.active_tab = tab_index

                    progress_bar.progress(1.0)
                    st.session_state.state = initial_state
                    st.rerun()

                except Exception as e:
                    logger.error(f"Workflow error: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"Generation error: {str(e)}")
                    st.session_state.generated = False

        elif st.session_state.generated and not goal.strip():
            st.error("Please enter a valid campaign request to proceed.")
            st.session_state.generated = False

        else:
            # Custom CSS
            st.markdown("""
                <style>
                    :root {
                        --background-color: rgba(255, 255, 255, 1.0);
                        --text-color: #1c1c1c;
                        --card-bg: white;
                        --border-color: #e6e6e6;
                        --heading-color: #2c3e50;
                        --shadow-color: rgba(0, 0, 0, 0.1);
                        --tab-bg: #f1f5f9;
                        --tab-selected-bg: white;
                        --usecase-text-color: #333;
                    }
                    @media (prefers-color-scheme: dark) {
                        :root {
                            --background-color: rgba(30, 30, 30, 0.95);
                            --text-color: #f1f1f1;
                            --card-bg: #2d2d2d;
                            --border-color: #444444;
                            --heading-color: #8ab4f8;
                            --shadow-color: rgba(0, 0, 0, 0.3);
                            --tab-bg: #383838;
                            --tab-selected-bg: #2d2d2d;
                            --usecase-text-color: #f1f1f1;
                        }
                    }
                    body {
                        background: linear-gradient(var(--background-color), var(--background-color)), url('https://images.unsplash.com/photo-1551836022-4c4c79ecde16?auto=format&fit=crop&w=1400&q=80') no-repeat center center fixed;
                        background-size: cover;
                        color: var(--text-color);
                        font-family: 'Segoe UI', sans-serif;
                    }
                    #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
                    .stApp {
                        background-color: var(--background-color);
                        padding: 0 !important;
                        border-radius: 0 !important;
                        box-shadow: none !important;
                        max-width: 100% !important;
                        margin: 0 !important;
                    }
                    .stTabs [role="tab"] {
                        font-size: 16px;
                        padding: 10px 20px;
                        margin-right: 5px;
                        border: 1px solid var(--border-color);
                        background-color: var(--tab-bg);
                        border-radius: 6px 6px 0 0;
                        color: var(--text-color);
                    }
                    .stTabs [aria-selected="true"] {
                        background-color: var(--tab-selected-bg);
                        border-bottom: none;
                        font-weight: bold;
                    }
                    .tab-content {
                        animation: fadein 0.6s ease-in;
                        background-color: var(--card-bg);
                        padding: 20px;
                        border-radius: 0 0 10px 10px;
                        box-shadow: 0 2px 4px var(--shadow-color);
                        margin-top: -1px;
                        border: 1px solid var(--border-color);
                        border-top: none;
                    }
                    .stTextArea textarea {
                        border: 1px solid var(--border-color);
                        border-radius: 8px;
                        padding: 10px;
                        font-size: 16px;
                        background-color: var(--card-bg);
                        color: var(--text-color);
                    }
                    .stButton button {
                        border-radius: 8px;
                        font-weight: 500;
                        transition: all 0.3s ease;
                    }
                    .stButton button:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 8px var(--shadow-color);
                    }
                    h1, h2, h3 {
                        color: var(--heading-color);
                        font-weight: 600;
                    }
                    .stTabs [data-baseweb="tab-panel"] {
                        background-color: var(--card-bg);
                        padding: 15px;
                        border-radius: 0 0 10px 10px;
                        border: 1px solid var(--border-color);
                        border-top: none;
                        color: var(--text-color);
                    }
                    .stProgress > div > div {
                        background-color: #4CAF50;
                    }
                    @keyframes fadein {
                        from {opacity: 0; transform: translateY(10px);}
                        to {opacity: 1; transform: translateY(0);}
                    }
                    /* Custom style for feedback section */
                    .feedback-header {font-size: 14px !important;}
                    /* Custom style for usecase section */
                    .usecase-container {
                        font-size: 18px;
                        line-height: 1;
                        color: var(--usecase-text-color);
                    }
                    .usecase-container h2 {
                        font-size: 26px;
                        margin-bottom: 10px;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Insert Usecase HTML into the main content
            st.markdown("""
                <div class="usecase-container">
                    <h2>Usecase</h2>
                    <p>The campaign builder will generate:</p>
                    <ol>
                        <li><strong>Market Analysis</strong> – Trends, opportunities, and competitive landscape</li>
                        <li><strong>Target Audience</strong> – Primary and secondary segments with messaging points</li>
                        <li><strong>Campaign Strategy</strong> – Timeline, channels, budget, and KPIs</li>
                        <li><strong>Content Examples</strong> – Email, social media, and landing page content</li>
                        <li><strong>Performance Simulation</strong> – Projected results and optimization recommendations</li>
                        <li><strong>Final Report</strong> – Complete campaign plan</li>
                        <li><strong>Email Distribution</strong> – Send campaign emails to target customers</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)


    except Exception as e:
        logger.error(f"Top-level error: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Top-level error: {str(e)}. Please check the logs.")
if __name__ == "__main__":
    main()
