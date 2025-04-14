from typing import Any, Optional
from pydantic import BaseModel

class CampaignState(BaseModel):
    goal: str
    vector_db: Any
    sales_data: Any
    market_analysis: Optional[str] = None
    audience_segments: Optional[str] = None
    campaign_strategy: Optional[str] = None
    campaign_content: Optional[str] = None
    simulation_results: Optional[str] = None
    final_report: Optional[str] = None
    email_status: Optional[str] = None

    # Add this to your CampaignState class initialization
    class CampaignState:
        def __init__(self, goal="", vector_db=None, sales_data=None, mode="ai", selected_llm=None):
            self.goal = goal
            self.vector_db = vector_db
            self.sales_data = sales_data
            self.mode = mode
            self.selected_llm = selected_llm  