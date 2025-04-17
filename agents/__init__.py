from .research_market import research_market_trends
from .segment_audience import segment_audience
from .create_strategy import create_campaign_strategy
from .generate_content import generate_content
from .simulate_campaign import simulate_campaign
from .generate_report import generate_final_report
from .send_emails import send_campaign_emails

AGENT_REGISTRY = {
    "research_market_trends": research_market_trends,
    "segment_audience": segment_audience,
    "create_campaign_strategy": create_campaign_strategy,
    "generate_content": generate_content,
    "simulate_campaign": simulate_campaign,
    "generate_final_report": generate_final_report,
    "send_campaign_emails": send_campaign_emails
}