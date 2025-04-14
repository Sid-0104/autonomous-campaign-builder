from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
import random

def simulate_campaign(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.4)

    # === Prompt Variations ===
    prompts = [
        f"""
        You're a marketing analyst simulating campaign performance.

        ### INPUTS:
        - CAMPAIGN GOAL: {state.goal}
        - STRATEGY: {state.campaign_strategy[:300]}... [truncated]
        - CONTENT: {state.campaign_content[:300]}... [truncated]

        ### DELIVERABLE:
        Provide a realistic performance simulation with:
        1. CHANNEL METRICS: Reach, engagement, CTR by channel
        2. CONVERSION FUNNEL: Leads, conversions, cost-per-acquisition
        3. ROI PROJECTION: Expected sales lift and return on investment
        4. RISK ASSESSMENT: Top 3 risks with mitigation tactics
        5. OPTIMIZATION: 3 specific recommendations to improve performance

        Use industry benchmarks and format in clear bullet points.
        """,

        f"""
        You're an expert in digital marketing analytics.

        Based on:
        • Goal: {state.goal}
        • Strategy: {state.campaign_strategy[:300]}... [truncated]
        • Content Plan: {state.campaign_content[:300]}... [truncated]

        Forecast the campaign outcomes by answering:
        - Which channel will perform best and why?
        - Estimated CTR, conversion rate, and CPA per channel
        - ROI and sales uplift
        - Risks and how to mitigate them
        - Suggestions to improve the campaign

        Keep it concise and data-driven.
        """,

        f"""
        You're running a simulation for a new marketing campaign.

        Details:
        - Goal: {state.goal}
        - Strategy Summary: {state.campaign_strategy[:300]}
        - Content Snapshot: {state.campaign_content[:300]}

        Predict:
        • Expected reach & conversions per channel
        • ROI estimates
        • Conversion funnel visualization (brief)
        • 3 risks + fixes
        • Optimization tips for better results

        Present findings in an analytical report format.
        """
    ]

    prompt = random.choice(prompts)

    response = safe_llm_invoke(llm, prompt)
    state.simulation_results = response.content if response else "Simulation failed"
    return state
