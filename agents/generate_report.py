from datetime import datetime
from core.state import CampaignState
import random

def generate_final_report(state: CampaignState):
    # Extract campaign name from strategy if possible
    campaign_name = "Campaign Plan"
    if state.campaign_strategy:
        first_line = state.campaign_strategy.strip().split("\n")[0]
        if first_line.startswith("# "):
            campaign_name = first_line.replace("# ", "")
        elif len(first_line.split()) <= 6:
            campaign_name = first_line

    # Random variation in next steps
    variations = [
        [
            "1. Review this draft with stakeholders",
            "2. Align final creative assets with strategy",
            "3. Assign owners for each campaign channel",
            "4. Launch according to the proposed timeline",
            "5. Track KPIs and iterate based on performance"
        ],
        [
            "1. Validate messaging with brand team",
            "2. Design visuals based on content direction",
            "3. Prepare landing pages and automation flows",
            "4. Launch in phases across regions",
            "5. Measure success and document learnings"
        ],
        [
            "1. Get stakeholder approval on final plan",
            "2. Start asset production using content examples",
            "3. Finalize media budget allocations",
            "4. Launch and track performance in real-time",
            "5. Review campaign against KPIs for optimization"
        ]
    ]
    next_steps = random.choice(variations)

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
{chr(10).join(next_steps)}

---

This plan was auto-generated using intelligent workflows. Use it as a blueprint to refine your go-to-market execution.
"""

    state.final_report = report
    return {"final_report": state.final_report}
