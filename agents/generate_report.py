from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from typing import Dict

AUTOMOBILE_PROMPT = """You're a marketing director creating a final report for an automotive campaign.
Focus on:
- Vehicle sales projections and market share impact
- Dealership engagement and test drive metrics
- Digital and traditional media performance for automotive
- Competitive positioning for {goal}
- Brand lift and consideration metrics
"""

HEALTHCARE_PROMPT = """You're a marketing director creating a final report for a healthcare campaign.
Focus on:
- Patient acquisition and retention metrics
- Provider reputation and referral growth
- Health education impact and engagement
- Regulatory compliance for {goal}
- Patient satisfaction and outcomes
"""

POWER_ENERGY_PROMPT = """You're a marketing director creating a final report for a power and energy campaign.
Focus on:
- Customer acquisition and service adoption
- Energy efficiency and sustainability metrics
- Technology adoption and smart solutions
- Regulatory considerations for {goal}
- Customer satisfaction and loyalty metrics
"""

DOMAIN_PROMPTS = {
    "automobiles": AUTOMOBILE_PROMPT,
    "healthcare": HEALTHCARE_PROMPT,
    "powerenergy": POWER_ENERGY_PROMPT
}

def generate_final_report(state: CampaignState) -> CampaignState:
    # Update this line to use the selected model from state
    llm = get_llm(temperature=0.4, model_provider=state.selected_llm)
    
    # Get domain-specific prompt
    domain = state.selected_domain.lower().replace("-", "")
    domain_intro = DOMAIN_PROMPTS.get(domain, AUTOMOBILE_PROMPT).format(goal=state.goal)
    
    # Domain-specific report sections
    report_sections = {
        "automobiles": ["Vehicle Sales Impact", "Dealership Performance", "Media Effectiveness", "Competitive Analysis"],
        "healthcare": ["Patient Acquisition", "Provider Reputation", "Service Utilization", "Compliance Summary"],
        "powerenergy": ["Customer Adoption", "Energy Efficiency Impact", "Technology Integration", "Sustainability Metrics"]
    }
    
    sections = report_sections.get(domain, report_sections["automobiles"])
    sections_text = "\n".join([f"- {section}" for section in sections])
    
    prompt = f"""
    {domain_intro}

    ### INPUTS:
    - CAMPAIGN GOAL: {state.goal}
    - MARKET ANALYSIS: {state.market_analysis[:200]}... [truncated]
    - TARGET AUDIENCE: {state.audience_segments[:200]}... [truncated]
    - CAMPAIGN STRATEGY: {state.campaign_strategy[:200]}... [truncated]
    - CONTENT EXAMPLES: {state.campaign_content[:200]}... [truncated]
    - PERFORMANCE SIMULATION: {state.simulation_results[:200]}... [truncated]

    ### REPORT SECTIONS:
    Include these domain-specific sections:
    {sections_text}

    ### DELIVERABLE:
    Create a comprehensive final campaign report with:
    1. EXECUTIVE SUMMARY: Key findings and recommendations (1 paragraph)
    2. CAMPAIGN OVERVIEW: Goal, strategy, audience (2-3 paragraphs)
    3. PERFORMANCE METRICS: Expected results across channels (use data from simulation)
    4. DOMAIN-SPECIFIC SECTIONS: Address the sections listed above
    5. IMPLEMENTATION PLAN: Timeline, resources, next steps
    6. CONCLUSION: Final recommendations and success factors

    Format as a professional report with clear headings and concise content.
    Include specific metrics and insights relevant to the {domain} industry.
    """
    
    response = safe_llm_invoke(llm, prompt)
    state.final_report = response.content if response else "Report generation failed"
    return state