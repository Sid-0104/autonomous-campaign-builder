from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState

def generate_content(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.8)
    
    prompt = f"""
    You're a creative content developer creating campaign materials.

    ### INPUTS:
    - CAMPAIGN GOAL: {state.goal}
    - TARGET AUDIENCE: {state.audience_segments[:300]}... [truncated]
    - CAMPAIGN STRATEGY: {state.campaign_strategy[:300]}... [truncated]

    ### DELIVERABLE:
    Create 3 content examples (one per channel) that align with the strategy:
    1. EMAIL: Subject line + first paragraph (compelling opener)
    2. SOCIAL MEDIA: 1-2 posts with hashtags (platform-appropriate)
    3. LANDING PAGE: Headline + key benefits section

    Each example should include the key message, call-to-action, and visual direction.
    """
    
    response = safe_llm_invoke(llm, prompt)
    state.campaign_content = response.content if response else "Content generation failed"
    return state