from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from langchain_core.messages import HumanMessage
from typing import Dict

AUTOMOBILE_PROMPT = """You're a marketing strategist in the automotive industry.
Focus on:
- Vehicle launch and promotional campaigns
- Seasonal sales events and financing offers
- Brand positioning against competitors
- Digital and traditional channel mix for {goal}
- Test drive and dealership experience strategies
"""

HEALTHCARE_PROMPT = """You're a marketing strategist in the healthcare industry.
Focus on:
- Patient education and awareness campaigns
- Preventive care and wellness programs
- Provider reputation and trust building
- Regulatory compliant marketing for {goal}
- Patient experience and satisfaction strategies
"""

POWER_ENERGY_PROMPT = """You're a marketing strategist in the power and energy sector.
Focus on:
- Energy efficiency and cost-saving campaigns
- Renewable energy adoption strategies
- Sustainability and environmental messaging
- Regulatory considerations for {goal}
- Smart technology and innovation positioning
"""

DOMAIN_PROMPTS = {
    "automobiles": AUTOMOBILE_PROMPT,
    "healthcare": HEALTHCARE_PROMPT,
    "powerenergy": POWER_ENERGY_PROMPT
}

def create_campaign_strategy(state: CampaignState) -> CampaignState:
    # Update this line to use the selected model from state
    llm = get_llm(temperature=0.7, model_provider=state.selected_llm)
    
    # Get domain-specific prompt
    domain = state.selected_domain.lower().replace("-", "")
    domain_intro = DOMAIN_PROMPTS.get(domain, AUTOMOBILE_PROMPT).format(goal=state.goal)
    
    # Handle case where vector_db is None
    past_campaigns = ""
    if state.vector_db is not None:
        try:
            # Get relevant past campaigns
            docs = state.vector_db.similarity_search(f"successful {domain} campaigns for {state.goal}", k=2)
            past_campaigns = "\n\n".join([doc.page_content for doc in docs if doc.metadata.get("category", "") == "campaign"])
            
            # If no campaign documents found, use all documents
            if not past_campaigns:
                past_campaigns = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error searching vector database: {e}")
            past_campaigns = "Vector database search failed. Using default campaigns."
    else:
        print("Vector database not available. Using default campaigns.")
        past_campaigns = "Vector database not available. Using default campaigns."
    
    # Domain-specific default campaigns when vector DB is not available
    default_campaigns = {
        "automobiles": """
        Past Campaign 1: Spring SUV Sales Event
        - 20% increase in SUV test drives
        - Focus on family safety features
        - Digital and dealership integration
        
        Past Campaign 2: Year-End Clearance
        - 15% sales increase
        - Competitive financing offers
        - Multi-channel marketing approach
        """,
        "healthcare": """
        Past Campaign 1: Preventive Care Awareness
        - 30% increase in checkup appointments
        - Focus on early detection benefits
        - Digital health integration
        
        Past Campaign 2: Family Health Initiative
        - 25% new patient registration
        - Multi-generational care approach
        - Community outreach success
        """,
        "powerenergy": """
        Past Campaign 1: Green Energy Switch
        - 40% increase in solar adoption
        - Focus on cost savings
        - Smart home integration
        
        Past Campaign 2: Energy Efficiency Program
        - 20% reduction in consumption
        - Rebate program success
        - Community engagement
        """
    }
    
    # If no campaign info from vector DB, use default campaigns
    if past_campaigns in ["Vector database search failed. Using default campaigns.", 
                         "Vector database not available. Using default campaigns."]:
        past_campaigns = default_campaigns.get(domain, default_campaigns["automobiles"])
    
    # Domain-specific channel recommendations
    channel_recommendations = {
        "automobiles": "Consider dealership events, automotive publications, test drive promotions, and targeted digital ads",
        "healthcare": "Consider patient portals, health fairs, provider networks, and HIPAA-compliant digital marketing",
        "powerenergy": "Consider utility partnerships, sustainability events, home improvement channels, and green certification programs"
    }
    
    channels = channel_recommendations.get(domain, channel_recommendations["automobiles"])
    
    prompt = f"""
    {domain_intro}

    ### INPUTS:
    - GOAL: {state.goal}
    - MARKET ANALYSIS: {state.market_analysis[:200] if state.market_analysis else "No market analysis available"}... [truncated]
    - TARGET AUDIENCE: {state.audience_segments[:200] if state.audience_segments else "No audience segments defined"}... [truncated]
    - REFERENCE CAMPAIGNS: {past_campaigns}

    ### CHANNEL GUIDANCE:
    {channels}

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