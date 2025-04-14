import streamlit as st
import time
import os
import re
from dotenv import load_dotenv
from agents import AGENT_REGISTRY
from core.vector_db import load_mock_data, initialize_vector_db
from core.state import CampaignState
from workflows.campaign_workflow import build_campaign_workflow
from fpdf import FPDF
from langchain_community.vectorstores import Chroma
from core.vector_db import initialize_vector_db

# 🔁 Import feedback functions
from utils.feedback import init_feedback_csv, save_node_feedback

# Initialize feedback storage
init_feedback_csv()

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

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = None
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = "gemini"
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = "automobiles"
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

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Cache data loading and vector DB initialization
@st.cache_resource
def cached_load_mock_data():
    return load_mock_data()

@st.cache_resource
def cached_initialize_vector_db(campaign_data, customer_segments):
    return initialize_vector_db(campaign_data, customer_segments)

# ===== Section Renderer + Feedback =====
def render_section(title, content, filename, node_name, key_prefix, state_obj):
    st.markdown(f"### {title}")
    st.write(content)
    st.download_button(
        label="📥 Download PDF",
        data=generate_pdf(title, content),
        file_name=filename,
        key=f"download_{key_prefix}_{node_name}"
        )

    st.markdown("#### 🗳️ How would you rate this section?")
    feedback_key = f"{node_name}_recorded"

    col1, col2, spacer = st.columns([1, 1, 6])
    if feedback_key not in st.session_state:
        with col1:
            if st.button("👍", key=f"{key_prefix}_positive"):
                msg = save_node_feedback(node_name, "Positive Feedback")
                st.success(msg)
                st.session_state[feedback_key] = True
        with col2:
            if st.button("👎", key=f"{key_prefix}_negative"):
                msg = save_node_feedback(node_name, "Negative Feedback")
                st.success(msg)
                st.session_state[feedback_key] = True
    else:
        st.info("✅ Feedback already recorded for this section.")

    # ✅ Regenerate logic
    if state_obj:
        st.markdown("#### 🔄 Want to improve this section?")
        if st.button("♻️ Regenerate", key=f"{key_prefix}_regen"):
            with st.spinner("Regenerating content..."):
                new_state = AGENT_REGISTRY[node_name](state_obj)
                new_content = None

                node_to_content = {
                    "research_market_trends": "market_analysis",
                    "segment_audience": "audience_segments",
                    "create_campaign_strategy": "campaign_strategy",
                    "generate_content": "campaign_content",
                    "simulate_campaign": "simulation_results",
                    "generate_final_report": "final_report",
                    "send_campaign_emails": "email_status"
                }

                if node_name in node_to_content and node_to_content[node_name] in new_state:
                    new_content = new_state[node_to_content[node_name]]
                    st.session_state.tab_contents[st.session_state.active_tab]["content"] = new_content
                    st.success("Section regenerated successfully!")
                    st.rerun()


    



def main():
    # Sidebar configuration
    with st.sidebar:
        col1, col2 = st.columns([1, 3])  # Adjust ratios as needed
        with col1:
           st.image("assests/info.png", width=60)  # Smaller width to better align with text
        with col2:
            st.markdown(
            """
            <div style="display: flex; align-items: center; height: 100%;">
                <h1 style='font-size: 20px; color:#4B8BBE; margin: 0;'>Autonomous<br>Campaign Builder</h1>
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
        for industry in ["Automobiles", "Healthcare", "Power-Energy"]:
            if st.button(industry, type="primary" if st.session_state.selected_domain == industry.lower().replace("-", "") else "secondary", use_container_width=True):
                st.session_state.selected_domain = industry.lower().replace("-", "")
                st.rerun()
        st.markdown(f"**Industry: {st.session_state.selected_domain.upper()}**")

    # Main content
    st.markdown("<h5 style='margin-bottom: 0.5rem;'>💬 Enter your campaign request</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        goal = st.text_area("Campaign goal", value="Boost Q2 SUV sales in the Western region by 15%", height=68, key="goal_input")
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
                    status_text.text("Loading data...")
                    sales_data, campaign_data, customer_segments = cached_load_mock_data()
                    progress_bar.progress(0.1)
                    if st.session_state.stop_requested:
                        status_text.text("⛔ Process stopped by user.")
                        st.session_state.generated = False
                        st.stop()

                    status_text.text("Initializing knowledge base...")
                    
                    vector_db = initialize_vector_db(campaign_data, customer_segments)
                    progress_bar.progress(0.2)
                    if st.session_state.stop_requested:
                        status_text.text("⛔ Process stopped by user.")
                        st.session_state.generated = False
                        st.stop()

                    campaign_workflow = build_campaign_workflow()
                    initial_state = CampaignState(goal=goal, vector_db=vector_db, sales_data=sales_data, mode="ai", selected_llm=st.session_state.selected_llm)
                    steps = list(AGENT_REGISTRY.keys())
                    total_steps = len(steps)

                    for i, output in enumerate(campaign_workflow.stream(initial_state)):
                        if st.session_state.stop_requested:
                            status_text.text("⛔ Process stopped by user.")
                            st.session_state.generated = False
                            st.stop()
                        node = list(output.keys())[0] if output else None
                        if node:
                            progress = min((i + 1) / total_steps, 0.9)
                            progress_bar.progress(progress)
                            status_text.text(f"Step {i+1}/{total_steps}: Processing {node}...")
                            state = output[node]
                            for key, value in state.items():
                                setattr(initial_state, key, value)
                            tab_index = -1
                            content_key = ""
                            node_to_tab = {
                                "research_market_trends": (0, "market_analysis"),
                                "segment_audience": (1, "audience_segments"),
                                "create_campaign_strategy": (2, "campaign_strategy"),
                                "generate_content": (3, "campaign_content"),
                                "simulate_campaign": (4, "simulation_results"),
                                "generate_final_report": (5, "final_report"),
                                "send_campaign_emails": (6, "email_status")
                            }
                            if node in node_to_tab:
                                tab_index, content_key = node_to_tab[node]
                            if tab_index >= 0 and content_key and content_key in state:
                                st.session_state.tab_contents[tab_index] = {"node": node, "content": state[content_key]}
                                st.session_state.active_tab = tab_index

                    progress_bar.progress(1.0)
                    status_text.text("Campaign generation complete!")
                    st.session_state.state = initial_state

                # Display tabs
                tab_labels = ["📊 Market Analysis", "🎯 Target Audience", "📈 Campaign Strategy", "✍️ Content", "🔬 Simulation", "📄 Final Report", "📬 Email Distribution"]
                tabs = st.tabs(tab_labels)

                for tab_index, tab in enumerate(tabs):
                    with tab:
                        if tab_index in st.session_state.tab_contents:
                            node = st.session_state.tab_contents[tab_index]["node"]
                            content = st.session_state.tab_contents[tab_index]["content"]
                            st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                            # Removed redundant title output
                            section_title = node.replace('_', ' ').title()
                            pdf_data = generate_pdf(section_title, content)
                            st.download_button(
                                label=f"📥 Download {section_title} PDF",
                                data=pdf_data,
                                file_name=f"{node}.pdf",
                                mime="application/pdf",
                                key=f"download_{node}_{tab_index}"
                            )
                            st.markdown("---")
                            render_section(section_title, content, f"{node}.pdf", node, f"tab_{tab_index}",st.session_state.state)
                            st.markdown("</div>", unsafe_allow_html=True)

                with tabs[5]:
                    if hasattr(st.session_state.state, 'final_report') and "final_report" not in st.session_state.tab_contents:
                        st.session_state.tab_contents[5] = {"node": "final_report", "content": st.session_state.state.final_report}
                    if 5 in st.session_state.tab_contents:
                        final_report_content = st.session_state.tab_contents[5]["content"]
                        pdf_data = generate_pdf("Final Report", final_report_content)
                        st.download_button(
                            label="📥 Download Complete Report",
                            data=pdf_data,
                            file_name="campaign_report.pdf",
                            mime="application/pdf",
                            key="download_final_report"
                        )
                        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                        render_section("Final Report", final_report_content, "final_report.pdf", "final_report", "report",st.session_state.state)
                        st.markdown("</div>", unsafe_allow_html=True)

                with tabs[6]:
                    st.subheader("Campaign Distribution")
                    if st.button("📧 Send Campaign Emails"):
                        with st.spinner("Sending emails to customers..."):
                            email_progress = st.progress(0)
                            email_state = AGENT_REGISTRY["send_campaign_emails"](st.session_state.state)
                            st.session_state.state = email_state
                            if hasattr(email_state, 'email_status'):
                                if "failed" in email_state.email_status.lower():
                                    st.error(email_state.email_status)
                                else:
                                    st.success(email_state.email_status)
                                    email_progress.progress(1.0)

            except Exception as e:
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

if __name__ == "__main__":
    main()