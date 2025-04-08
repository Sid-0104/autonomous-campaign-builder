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