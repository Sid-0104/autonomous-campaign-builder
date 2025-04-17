from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from langchain_core.messages import HumanMessage
from typing import Dict

AUTOMOBILE_PROMPT = """You're a content creator specializing in automotive marketing.
Focus on:
- Vehicle features, performance, and specifications
- Driving experience and lifestyle benefits
- Financing and promotional offers
- Test drive invitations and dealership events for {goal}
- Safety, technology, and innovation messaging
"""

HEALTHCARE_PROMPT = """You're a content creator specializing in healthcare marketing.
Focus on:
- Patient education and wellness information
- Provider expertise and credentials
- Treatment benefits and outcomes
- Preventive care and health management for {goal}
- Patient testimonials and success stories (anonymized)
"""

POWER_ENERGY_PROMPT = """You're a content creator specializing in power and energy marketing.
Focus on:
- Energy efficiency and cost savings
- Renewable energy benefits and sustainability
- Smart technology and innovation
- Environmental impact and green initiatives for {goal}
- Regulatory compliance and safety messaging
"""

DOMAIN_PROMPTS = {
    "automobiles": AUTOMOBILE_PROMPT,
    "healthcare": HEALTHCARE_PROMPT,
    "powerenergy": POWER_ENERGY_PROMPT
}

def generate_content(state: CampaignState) -> CampaignState:
    # Update this line to use the selected model from state
    llm = get_llm(temperature=0.7, model_provider=state.selected_llm)
    
    # Get domain-specific prompt
    domain = state.selected_domain.lower().replace("-", "")
    domain_intro = DOMAIN_PROMPTS.get(domain, AUTOMOBILE_PROMPT).format(goal=state.goal)
    
    # Domain-specific content types
    content_types = {
        "automobiles": ["Email campaign", "Social media posts", "Landing page copy", "Test drive invitation"],
        "healthcare": ["Patient newsletter", "Health tips social posts", "Service information page", "Appointment reminder"],
        "powerenergy": ["Energy savings email", "Sustainability social posts", "Service comparison page", "Smart technology guide"]
    }
    
    selected_content = content_types.get(domain, content_types["automobiles"])
    
    prompt = f"""
    {domain_intro}

    ### INPUTS:
    - CAMPAIGN GOAL: {state.goal}
    - TARGET AUDIENCE: {state.audience_segments[:300]}... [truncated]
    - CAMPAIGN STRATEGY: {state.campaign_strategy[:300]}... [truncated]

    ### DELIVERABLE:
    Create the following content pieces for this campaign:

    1. {selected_content[0]}:
       - Subject line
       - Body copy (150-200 words)
       - Call to action

    2. {selected_content[1]} (3 posts):
       - Post 1 (with hashtags)
       - Post 2 (with hashtags)
       - Post 3 (with hashtags)

    3. {selected_content[2]}:
       - Headline
       - Subheadline
       - Main copy (200-250 words)
       - Features/benefits section
       - Call to action

    4. {selected_content[3]}:
       - Headline
       - Short copy (100 words)
       - Call to action

    Ensure all content aligns with the campaign strategy and speaks directly to the target audience.
    Use persuasive language appropriate for the {domain} industry.
    """
    
    response = safe_llm_invoke(llm, prompt)
    state.campaign_content = response.content if response else "Content generation failed"
    return state