from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from langchain_core.messages import HumanMessage
import random

def segment_audience(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.6)

    # Extract keywords from goal
    goal_keywords = state.goal.lower().split()
    search_terms = " ".join([term for term in goal_keywords if len(term) > 3])

    # Get relevant segments
    docs = state.vector_db.similarity_search(f"customer segments interested in {search_terms}", k=2)
    segment_info = "\n\n".join([doc.page_content for doc in docs if doc.metadata.get("type") == "segment"])

    # Prompt variants for regenerate uniqueness
    prompt_variants = [
        f"""
        You're a customer insights expert. Based on the inputs below, provide two ideal target segments.

        ### CAMPAIGN GOAL:
        {state.goal}

        ### MARKET CONTEXT:
        {state.market_analysis[:300]}... [truncated]

        ### CUSTOMER DATA:
        {segment_info}

        ### DELIVERABLE:
        - 1 primary segment (with justification)
        - 1 secondary segment
        - Messaging points tailored to both

        Use bullet points and clear formatting.
        """,

        f"""
        As a marketing strategist, your task is to identify the best audience for this campaign:

        GOAL: {state.goal}
        MARKET: {state.market_analysis[:300]}... [truncated]
        CUSTOMER SEGMENTS: {segment_info}

        Return:
        • Primary Segment: description, traits, why it's a fit  
        • Secondary Segment: brief reason  
        • Messaging: 3-5 points to hook both segments  
        Format in markdown.
        """,

        f"""
        You are tasked with mapping campaign audiences.

        - CAMPAIGN OBJECTIVE: {state.goal}
        - MARKET INSIGHTS: {state.market_analysis[:300]}... [truncated]
        - HISTORIC SEGMENTS: {segment_info}

        DELIVER:
        1. Primary target persona: traits, appeal
        2. Backup segment: why it's considered
        3. Messaging bullets

        Be crisp, strategic, and practical.
        """
    ]

    selected_prompt = random.choice(prompt_variants)

    response = safe_llm_invoke(llm, selected_prompt)
    state.audience_segments = response.content if response else "Audience segmentation failed"

    return state
