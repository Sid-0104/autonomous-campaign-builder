import os
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Optional
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Update the configuration section at the top
from dotenv import load_dotenv
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/text-embedding-004")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 120))
STEP_DELAY = int(os.environ.get("STEP_DELAY", 10))

# Mock data for the PoC
def load_mock_data():
    # Load sales data
    sales_data = pd.read_json('c:\\Users\\user\\Desktop\\campaign builder\\sales_data.json')
    
    # Load campaign data
    with open('c:\\Users\\user\\Desktop\\campaign builder\\campaign_data.json', 'r') as f:
        campaign_data = json.load(f)
    
    # Load customer segments
    with open('c:\\Users\\user\\Desktop\\campaign builder\\customer_segments.json', 'r') as f:
        customer_segments = json.load(f)
    
    return sales_data, campaign_data, customer_segments

# Initialize vector database for campaigns and customer segments
def initialize_vector_db(campaign_data, customer_segments):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
    # Create documents for campaigns
    campaign_docs = []
    for campaign in campaign_data:
        content = f"Campaign: {campaign['name']}\n"
        content += f"Region: {campaign['region']}\n"
        content += f"Period: {campaign['start_date']} to {campaign['end_date']}\n"
        content += f"Target models: {', '.join(campaign['target_models'])}\n"
        content += f"Channels: {', '.join(campaign['channels'])}\n"
        content += f"Messaging: {campaign['messaging']}\n"
        content += f"Results: ROI {campaign['results']['roi']}, Sales {campaign['results']['sales']}"
        
        metadata = {
            "type": "campaign",
            "name": campaign["name"],
            "region": campaign["region"],
            "target_models": campaign["target_models"]
        }
        
        campaign_docs.append(Document(page_content=content, metadata=metadata))
    
    # Create documents for customer segments
    segment_docs = []
    for segment in customer_segments:
        content = f"Segment: {segment['name']}\n"
        content += f"Demographics: {json.dumps(segment['demographics'])}\n"
        content += f"Preferred models: {', '.join(segment['preferences']['models'])}\n"
        content += f"Values features: {', '.join(segment['preferences']['features'])}\n"
        content += f"Preferred channels: {', '.join(segment['preferences']['channels'])}"
        
        metadata = {
            "type": "segment",
            "name": segment["name"],
            "preferred_models": segment["preferences"]["models"]
        }
        
        segment_docs.append(Document(page_content=content, metadata=metadata))
    
    # Combine all documents
    all_docs = campaign_docs + segment_docs
    
    # Create the vector store
    vector_store = FAISS.from_documents(all_docs, embeddings)
    
    return vector_store

def get_llm(temperature=0.5):
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
        max_retries=MAX_RETRIES,
        request_timeout=REQUEST_TIMEOUT
    )
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=5, max=60))
def safe_llm_invoke(llm, prompt):
    try:
        return llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            wait_time = 30  # Base wait time
            if "retry_delay" in str(e):
                import re
                match = re.search(r'seconds: (\d+)', str(e))
                if match:
                    wait_time = int(match.group(1)) * 2 
            print(f"API rate limit reached. Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            raise
        print(f"Error invoking LLM: {str(e)}")
        raise

# Define the state class
class CampaignState(BaseModel):
    goal: str
    vector_db: Any
    sales_data: Any
    market_analysis: Optional[str] = None
    audience_segments: Optional[str] = None
    campaign_strategy: Optional[str] = None
    campaign_content: Optional[str] = None
    simulation_results: Optional[str] = None
    final_report: Optional[str] = None

# Then keep all the workflow functions below
def research_market_trends(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.7)
    prompt = f"""
    Conduct a comprehensive market analysis for automotive marketing campaigns with these specifications:
    
    Campaign Goal: {state.goal}
    
    Research Focus Areas:
    1. Current market trends in the automotive industry
    2. Consumer buying patterns for SUVs (if SUV is mentioned in goal)
    3. Competitive landscape analysis
    4. Emerging technologies impacting automotive marketing
    5. Regional market dynamics (if region is specified)
    
    Required Output Format:
    - Executive Summary (2-3 paragraphs)
    - Key Trends (bulleted list)
    - Opportunities (ranked by potential impact)
    - Threats/Risks (with mitigation suggestions)
    - Recommended Data Sources (industry reports, studies, etc.)
    
    Additional Considerations:
    - Focus on actionable insights
    - Include relevant statistics and data points
    - Highlight any seasonal factors
    - Note any regulatory changes affecting marketing
    """
    
    response = safe_llm_invoke(llm, prompt)
    state.market_analysis = response.content if response else "No insights available"
    return state


def segment_audience(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.5)
    
    model_focus = "SUV" if "SUV" in state.goal else "general"
    
    docs = state.vector_db.similarity_search(f"customer segments interested in {model_focus}", k=2)
    segment_info = "\n\n".join([doc.page_content for doc in docs if doc.metadata["type"] == "segment"])
    
    prompt = f"""
    Based on the goal: "{state.goal}" and market analysis: "{state.market_analysis}", 
    I need to identify the optimal customer segments to target.
    
    Here's information on potential segments:
    {segment_info}
    
    Please provide:
    1. Primary segment to target and why
    2. Secondary segment (if applicable)
    3. Key messaging points that would resonate with these segments
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    state.audience_segments = response.content
    return state

def create_campaign_strategy(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.7)
    
    docs = state.vector_db.similarity_search(f"successful campaigns for {state.goal}", k=2)
    past_campaigns = "\n\n".join([doc.page_content for doc in docs if doc.metadata["type"] == "campaign"])
    
    prompt = f"""
    Create a comprehensive campaign strategy based on:
    - Goal: {state.goal}
    - Market Analysis: {state.market_analysis}
    - Target Audience: {state.audience_segments}
    
    Reference these past campaigns for insights:
    {past_campaigns}
    
    Your strategy should include:
    1. Campaign name and theme
    2. Proposed timeline and key dates
    3. Channel mix and budget allocation
    4. Key performance indicators (KPIs)
    5. High-level messaging approach
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    state.campaign_strategy = response.content
    return state

def generate_content(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.8)
    
    prompt = f"""
    Based on the campaign strategy:
    {state.campaign_strategy}
    
    And target audience information:
    {state.audience_segments}
    
    Create the following campaign content:
    1. Email subject line (3 options)
    2. Email body copy (main message, 150-200 words)
    3. Social media posts (3 variants, each under 280 characters)
    4. Landing page headline and key selling points
    
    Ensure all content is cohesive, on-brand, and speaks directly to the identified audience segments.
    """
    
    try:
        response = safe_llm_invoke(llm, prompt)
        state.campaign_content = response.content
    except Exception as e:
        state.campaign_content = f"Content generation failed: {str(e)}"
    
    return state

def simulate_campaign(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.4)
    if not llm:
        state.simulation_results = "Could not generate simulation due to API limitations"
        return state
    
    prompt = f"""
    Simulate the expected results of this campaign:
    
    Goal: {state.goal}
    Strategy: {state.campaign_strategy}
    Content Examples: {state.campaign_content}
    
    Provide a detailed simulation including:
    1. Estimated reach and engagement metrics by channel
    2. Expected lead generation and conversion rates
    3. Projected sales lift and ROI
    4. Potential challenges or risks
    5. Recommendations for optimization
    
    Base your projections on realistic marketing metrics and outcomes.
    """
    
    try:
        response = safe_llm_invoke(llm, prompt)
        state.simulation_results = response.content
    except Exception as e:
        state.simulation_results = f"Simulation incomplete due to error: {str(e)}"
    
    return state

def generate_final_report(state: CampaignState) -> CampaignState:
    report = f"""
    # Autonomous Campaign Builder: Campaign Plan
    
    ## Campaign Overview
    **Goal:** {state.goal}
    **Date Generated:** {datetime.now().strftime('%Y-%m-%d')}
    
    ## Market Analysis
    {state.market_analysis}
    
    ## Target Audience
    {state.audience_segments}
    
    ## Campaign Strategy
    {state.campaign_strategy}
    
    ## Campaign Content Examples
    {state.campaign_content}
    
    ## Performance Simulation
    {state.simulation_results}
    """
    
    state.final_report = report
    return state

def build_campaign_workflow():
    workflow = StateGraph(CampaignState)

    workflow.add_node("research_market_trends", research_market_trends)
    workflow.add_node("segment_audience", segment_audience)
    workflow.add_node("create_campaign_strategy", create_campaign_strategy)
    workflow.add_node("generate_content", generate_content)
    workflow.add_node("simulate_campaign", simulate_campaign)
    workflow.add_node("generate_final_report", generate_final_report)

    workflow.add_edge("research_market_trends", "segment_audience")
    workflow.add_edge("segment_audience", "create_campaign_strategy")
    workflow.add_edge("create_campaign_strategy", "generate_content")
    workflow.add_edge("generate_content", "simulate_campaign")
    workflow.add_edge("simulate_campaign", "generate_final_report")

    workflow.set_entry_point("research_market_trends")
    workflow.set_finish_point("generate_final_report")

    return workflow.compile()



def run_autonomous_campaign_builder(goal: str):
    
    sales_data, campaign_data, customer_segments = load_mock_data()
    
    vector_db = initialize_vector_db(campaign_data, customer_segments)
    
    campaign_workflow = build_campaign_workflow()
    
    initial_state = CampaignState(
        goal=goal,
        vector_db=vector_db,
        sales_data=sales_data
    )
    
   
    STEP_DELAY = 10  
    
    for output in campaign_workflow.stream(initial_state):
        node = list(output.keys())[0] if output else None
        if node:
            print(f"Completed: {node}")
            time.sleep(STEP_DELAY)
    
    
    final_state = campaign_workflow.invoke(initial_state)
    
    return final_state["final_report"]


if __name__ == "__main__":
    campaign_goal = "Boost Q2 SUV sales in the Western region by 15%"
    final_report = run_autonomous_campaign_builder(campaign_goal)
    print(final_report)


 


