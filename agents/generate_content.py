import random
from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState

def generate_content(state: CampaignState):
    llm = get_llm(temperature=0.8)

    # 🔁 Random variation in prompt intro
    intros = [
        "You're a creative content developer building campaign assets.",
        "As a content strategist, generate compelling multi-channel material.",
        "You are designing attention-grabbing content for a marketing campaign.",
        "Write persuasive content to support this marketing initiative.",
        "Develop audience-targeted content across multiple platforms."
    ]

    selected_intro = random.choice(intros)

    prompt = f"""
    {selected_intro}

    ### INPUTS:
    - CAMPAIGN GOAL: {state.goal}
    - TARGET AUDIENCE: {state.audience_segments[:100]}... [truncated]
    - CAMPAIGN STRATEGY: {state.campaign_strategy[:100]}... [truncated]

    ### DELIVERABLE:
    Create 3 content examples (one per channel) that align with the strategy:
    1. EMAIL: Subject line + first paragraph (compelling opener)
    2. SOCIAL MEDIA: 1-2 posts with hashtags (platform-appropriate)
    3. LANDING PAGE: Headline + key benefits section

    Each example should:
    - Include a clear value proposition
    - Have a compelling call-to-action
    - Provide visual direction suggestions
    - Match the brand voice and campaign goals
    """

    response = safe_llm_invoke(llm, prompt)
    state.campaign_content = response.content if response else "Content generation failed"

    return {"campaign_content": state.campaign_content}
