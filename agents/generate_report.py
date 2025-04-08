from datetime import datetime
from core.state import CampaignState

def generate_final_report(state: CampaignState) -> CampaignState:
    # Extract campaign name from strategy if available
    campaign_name = "Campaign Plan"
    if state.campaign_strategy and "# " in state.campaign_strategy:
        first_line = state.campaign_strategy.split('\n')[0]
        if first_line.startswith('# '):
            campaign_name = first_line.replace('# ', '')
    
    report = f"""
    # {campaign_name}
    
    ## Campaign Overview
    **Goal:** {state.goal}
    **Date Generated:** {datetime.now().strftime('%Y-%m-%d')}
    
    ## Market Analysis
    {state.market_analysis}
    
    ## Target Audience
    {state.audience_segments}
    
    ## Campaign Strategy
    {state.campaign_strategy}
    
    ## Campaign Content Examples
    {state.campaign_content}
    
    ## Performance Simulation
    {state.simulation_results}
    
    ## Next Steps
    1. Review and refine this AI-generated campaign plan
    2. Develop detailed creative assets based on content examples
    3. Set up tracking for the identified KPIs
    4. Implement the campaign according to the proposed timeline
    5. Monitor performance and apply optimization recommendations
    """
    
    state.final_report = report
    return state