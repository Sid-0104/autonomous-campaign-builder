from typing import Any, Optional, List, Dict
from pydantic import BaseModel

class CampaignState(BaseModel):
    goal: str
    vector_db: Any
    sales_data: Any
    industry_data: Optional[Any] = None
    selected_domain: str = "automotives"
    selected_llm: str = "gemini"
    market_analysis: Optional[str] = None
    audience_segments: Optional[str] = None
    campaign_strategy: Optional[str] = None
    campaign_content: Optional[str] = None
    simulation_results: Optional[str] = None
    final_report: Optional[str] = None
    email_status: Optional[str] = None
    sent_emails: List[Dict] = []  # Track sent emails
    email_templates: List[Dict] = []  # Store email templates
    
    # Workflow control fields
    workflow_status: str = "running"  # Options: running, paused, completed
    awaiting_ui_action: bool = False  # Flag to indicate if waiting for UI action