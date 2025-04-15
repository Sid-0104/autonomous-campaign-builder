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
from datetime import datetime

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
def sanitize_text(text):
    if text is None:
        return ""
    text = str(text)
    text = ''.join(char if 32 <= ord(char) <= 126 else ' ' for char in text)
    text = text.replace("**", "").replace("*", "").replace("-", " ").replace("•", " ").replace("#", "").replace("`", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_pdf(title, content, pdf_path="market_insights.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    margin_left = 10
    margin_right = 10
    margin_top = 15
    pdf.set_left_margin(margin_left)
    pdf.set_right_margin(margin_right)
    page_width = pdf.w - margin_left - margin_right
    line_height = 7

    def add_heading():
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(30, 30, 30)
        clean_title = sanitize_text(title)
        pdf.cell(0, 10, f"Market Insights: {clean_title}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)

    add_heading()

    bullet = "-"
    clean_content = sanitize_text(content)
    paragraphs = re.split(r'\n\s*\n', clean_content) if clean_content else []

    for para in paragraphs:
        lines = para.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            is_bullet = line.startswith("-") or line.startswith(".")
            content_line = line.lstrip("-.").strip()
            if content_line[:40].lower().startswith('top') or ':' in content_line[:40]:
                pdf.set_font("Helvetica", "B", 12)
            else:
                pdf.set_font("Helvetica", "", 12)
            formatted = f"{bullet} {content_line}" if is_bullet else f"  {content_line}"
            words = formatted.split()
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                if pdf.get_string_width(test_line) > page_width:
                    pdf.set_x(margin_left)
                    pdf.multi_cell(page_width, line_height, text=current_line.strip(), align='L')
                    current_line = word + " "
                    if pdf.get_y() + line_height > 277:
                        pdf.add_page()
                        add_heading()
                        pdf.set_y(margin_top)
                else:
                    current_line = test_line
            if current_line:
                pdf.set_x(margin_left)
                pdf.multi_cell(page_width, line_height, text=current_line.strip(), align='L')
                if pdf.get_y() + line_height > 277:
                    pdf.add_page()
                    add_heading()
                    pdf.set_y(margin_top)
            pdf.ln(line_height)

    try:
        pdf.output(pdf_path)
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                raise ValueError("PDF is empty")
        else:
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
    except Exception as e:
        raise

    with open(pdf_path, "rb") as f:
        return f.read()

# Function for vector db initialization with caching
@st.cache_resource
def cached_initialize_vector_db(domain, model_provider):
    return initialize_vector_db(domain=domain, model_provider=model_provider)

def run_autonomous_campaign_builder(goal: str, domain: str = "automotives", model_provider: str = "gemini"):
    logger.info(f"Initializing campaign builder for {domain} domain using {model_provider}")
    try:
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
                    time.sleep(5)
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
if 'sidebar_collapsed' not in st.session_state:
    st.session_state.sidebar_collapsed = True

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

def main():
    logger.info(f"Starting application - sidebar_collapsed = {st.session_state.sidebar_collapsed}")

    try:
        # Display heading only when sidebar_collapsed is True
        if st.session_state.sidebar_collapsed:
            with st.container():
                st.markdown("<h1 style='font-size: 20px;'>Autonomous Campaign Builder</h1>", unsafe_allow_html=True)
            logger.debug("Main heading displayed due to sidebar_collapsed = True")

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
                if st.button(llm, key=f"llm_{llm.lower()}", use_container_width=True, type="primary" if st.session_state.selected_llm == llm.lower() else "secondary"):
                    st.session_state.selected_llm = llm.lower()
                    st.rerun()
            st.markdown(f"**Using: {st.session_state.selected_llm.upper()}**")

            st.subheader("Select Industry")
            for industry in ["Automotives", "Healthcare", "Powerenergy"]:
                domain_value = industry.lower()
                if st.button(industry, key=f"industry_{domain_value}", use_container_width=True, type="primary" if st.session_state.selected_domain == domain_value else "secondary"):
                    st.session_state.selected_domain = domain_value
                    st.rerun()

            st.markdown(f"**Industry: {st.session_state.selected_domain.upper()}**")

        col1, col2 = st.columns([2, 3])

        with col1:
            goal = st.text_area("", placeholder="Enter your campaign request", height=68)

        with col2:
            with st.container():
                st.markdown('<div class="button-row">', unsafe_allow_html=True)
                col_a, col_b, col_c = st.columns([1, 1, 1], gap="small")
                with col_a:
                    if st.button("Generate", key="go_button"):
                        if goal.strip():
                            st.session_state.generated = True
                            st.session_state.stop_requested = False
                            st.session_state.tab_contents = {}
                            st.session_state.sidebar_collapsed = False
                            logger.debug("Generate clicked, setting sidebar_collapsed to False")
                            time.sleep(0.2)
                            st.rerun()
                        else:
                            st.error("Please enter a request before generating.")

                with col_b:
                    if st.button("Stop", key="stop_button"):
                        st.session_state.stop_requested = True
                        st.rerun()

                with col_c:
                    if st.button("Back", key="back_button"):
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

        if st.session_state.generated and goal.strip():
            with st.spinner("Generating campaign..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    if not st.session_state.tab_contents:
                        status_text.text("Initializing campaign builder...")
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
                                time.sleep(STEP_DELAY / 2)  # Reduced delay to 2.5 seconds for responsiveness
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
            st.markdown("""
            <style>
                .usecase-container {
                    font-size: 18px;
                    line-height: 1.6;
                    color: #333;
                }
                .usecase-container h2 {
                    font-size: 24px;
                    margin-bottom: 10px;
                }
            </style>
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

def render_section(title, content, filename, node_name, key_prefix, state_obj):
    st.markdown(f"### {title}")
    st.write(content)
    try:
        pdf_data = generate_pdf(title, content)
        st.download_button(
            label="📥 Download PDF",
            data=pdf_data,
            file_name=filename,
            key=f"download_{key_prefix}_{node_name}_{datetime.now().timestamp()}"
        )
    except Exception as e:
        logger.error(f"Failed to generate PDF for {title}: {str(e)}")
        st.warning("PDF download is unavailable due to an error. Check logs for details.")
    st.markdown("#### 🗳️ How would you rate this section?")
    feedback_key = f"{node_name}_recorded"
    col1, col2 = st.columns(2)
    if feedback_key not in st.session_state:
        with col1:
            if st.button("👍", key=f"{key_prefix}_positive", help="Like"):
                st.session_state[feedback_key] = "positive"
                st.toast("👍 Positive feedback recorded!", icon="✅")
                time.sleep(2)
                st.rerun()

        with col2:
            if st.button("👎", key=f"{key_prefix}_negative", help="Dislike"):
                st.session_state[feedback_key] = "negative"
                st.toast("👎 Negative feedback recorded!", icon="⚠️")
                time.sleep(2)
                st.rerun()

    else:
        feedback = st.session_state[feedback_key]
        if feedback == "positive":
            st.success("feedback already recorded.")
        else:
            st.info("feedback already recorded.")
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
                    st.session_state.state = new_state
                    st.success("✅ Section regenerated!")
                    st.rerun()

if __name__ == "__main__":
    main()