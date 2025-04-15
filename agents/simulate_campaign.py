from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from typing import Dict

AUTOMOBILE_PROMPT = """You're a marketing analyst simulating automotive campaign performance.
Focus on:
- Vehicle sales lift and lead generation metrics
- Dealership traffic and test drive conversions
- Digital engagement with vehicle features and specifications
- Competitive conquest rates for {goal}
- Brand perception and consideration metrics
"""

HEALTHCARE_PROMPT = """You're a marketing analyst simulating healthcare campaign performance.
Focus on:
- Patient appointment bookings and service utilization
- Provider selection and referral metrics
- Health education content engagement
- Patient acquisition cost for {goal}
- Satisfaction and retention metrics
"""

POWER_ENERGY_PROMPT = """You're a marketing analyst simulating power and energy campaign performance.
Focus on:
- Service adoption and conversion rates
- Energy savings and efficiency metrics
- Sustainability impact measurements
- Customer acquisition cost for {goal}
- Brand trust and loyalty metrics
"""

DOMAIN_PROMPTS = {
    "automobiles": AUTOMOBILE_PROMPT,
    "healthcare": HEALTHCARE_PROMPT,
    "powerenergy": POWER_ENERGY_PROMPT
}

def simulate_campaign(state: CampaignState) -> CampaignState:
    # Update this line to use the selected model from state
    llm = get_llm(temperature=0.4, model_provider=state.selected_llm)
    
    # Get domain-specific prompt
    domain = state.selected_domain.lower().replace("-", "")
    domain_intro = DOMAIN_PROMPTS.get(domain, AUTOMOBILE_PROMPT).format(goal=state.goal)
    
    # Domain-specific benchmarks
    benchmarks = {
        "automobiles": {
            "email_open_rate": "22-28%",
            "social_engagement": "1.5-3.2%",
            "landing_page_conversion": "2.5-5%",
            "cost_per_lead": "$25-45",
            "test_drive_conversion": "15-25%"
        },
        "healthcare": {
            "email_open_rate": "19-24%",
            "social_engagement": "1.2-2.8%",
            "landing_page_conversion": "3-6%",
            "cost_per_appointment": "$35-75",
            "patient_retention": "65-80%"
        },
        "powerenergy": {
            "email_open_rate": "20-26%",
            "social_engagement": "1.0-2.5%",
            "landing_page_conversion": "2-4.5%",
            "cost_per_acquisition": "$50-120",
            "service_adoption": "10-20%"
        }
    }
    
    domain_benchmarks = benchmarks.get(domain, benchmarks["automobiles"])
    benchmark_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in domain_benchmarks.items()])
    
    prompt = f"""
    {domain_intro}

    ### INPUTS:
    - CAMPAIGN GOAL: {state.goal}
    - STRATEGY: {state.campaign_strategy[:300]}... [truncated]
    - CONTENT: {state.campaign_content[:300]}... [truncated]

    ### INDUSTRY BENCHMARKS:
    {benchmark_text}

    ### DELIVERABLE:
    Provide a realistic performance simulation with:
    1. CHANNEL METRICS: Reach, engagement, CTR by channel
    2. CONVERSION FUNNEL: Leads, conversions, cost-per-acquisition
    3. ROI PROJECTION: Expected sales lift and return on investment
    4. RISK ASSESSMENT: Top 3 risks with mitigation tactics
    5. OPTIMIZATION: 3 specific recommendations to improve performance

    Use the industry benchmarks provided and be realistic. Format with clear sections and data points.
    Include specific metrics relevant to the {domain} industry.
    """
    
    response = safe_llm_invoke(llm, prompt)
    state.simulation_results = response.content if response else "Simulation failed"
    return state