import random
from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from langchain_core.messages import HumanMessage

def create_campaign_strategy(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.8)  # 🔥 more creativity

    # Get relevant past campaigns
    docs = state.vector_db.similarity_search(f"successful campaigns for {state.goal}", k=2)
    past_campaigns = "\n\n".join([doc.page_content for doc in docs if doc.metadata["type"] == "campaign"])

    # 🔁 Prompt templates to inject variation
    prompt_templates = [
        "Craft a campaign strategy to meet the following marketing goal.",
        "Develop a fresh campaign plan aligned to the business objectives below.",
        "Design a marketing strategy using the inputs provided.",
        "You're a strategist, write a creative yet data-backed campaign blueprint.",
        "Prepare a well-structured campaign strategy for the goal outlined."
    ]

    prompt_intro = random.choice(prompt_templates)

    prompt = f"""
    {prompt_intro}

    ### INPUTS:
    - GOAL: {state.goal}
    - MARKET ANALYSIS: {state.market_analysis[:200]}... [truncated]
    - TARGET AUDIENCE: {state.audience_segments[:200]}... [truncated]
    - REFERENCE CAMPAIGNS: {past_campaigns[:200]}... [truncated]

    ### DELIVERABLE:
    Create a structured campaign strategy with:
    1. Campaign name & theme (aligned with brand and goal)
    2. Timeline (key dates, duration)
    3. Channel strategy (prioritized channels + budget %)
    4. KPIs (specific metrics to track success)
    5. Messaging framework (key value propositions)

    Format with clear headings and bullet points in clear and concise manner.
    """

    response = safe_llm_invoke(llm, prompt)
    state.campaign_strategy = response.content if response else "Strategy generation failed"
    return state
