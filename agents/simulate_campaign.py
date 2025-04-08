from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState

def simulate_campaign(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.4)
    
    prompt = f"""
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

    Use industry benchmarks and be realistic. Format with clear sections and data points.
    """
    
    response = safe_llm_invoke(llm, prompt)
    state.simulation_results = response.content if response else "Simulation failed"
    return state