from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from langchain_core.messages import HumanMessage

def create_campaign_strategy(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.7)
    
    # Get relevant past campaigns
    docs = state.vector_db.similarity_search(f"successful campaigns for {state.goal}", k=2)
    past_campaigns = "\n\n".join([doc.page_content for doc in docs if doc.metadata["type"] == "campaign"])
    
    prompt = f"""
    You're a marketing strategist creating a campaign plan.

    ### INPUTS:
    - GOAL: {state.goal}
    - MARKET ANALYSIS: {state.market_analysis[:500]}... [truncated]
    - TARGET AUDIENCE: {state.audience_segments[:500]}... [truncated]
    - REFERENCE CAMPAIGNS: {past_campaigns[:500]}... [truncated]

    ### DELIVERABLE:
    Create a structured campaign strategy with:
    1. Campaign name & theme (catchy, memorable)
    2. Timeline (key dates, duration)
    3. Channel strategy (prioritized channels + budget %)
    4. KPIs (specific metrics to track success)
    5. Messaging framework (key value propositions)

    Format with clear headings and bullet points.
    """
    
    response = safe_llm_invoke(llm, prompt)
    state.campaign_strategy = response.content if response else "Strategy generation failed"
    return state