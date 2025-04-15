from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from tavily import TavilyClient
import os
import pandas as pd
import numpy as np
from typing import Dict

AUTOMOBILE_PROMPT = """You are a senior data analyst at an automotive marketing agency. Analyze the market focusing on:
- Current vehicle preferences and buying patterns
- Regional market dynamics for {goal}
- Competitor analysis in the automotive sector
- Economic factors affecting car purchases
- Technological trends in the automotive industry
"""

HEALTHCARE_PROMPT = """You are a senior data analyst at a healthcare marketing agency. Analyze the market focusing on:
- Current healthcare service demands
- Regional healthcare needs for {goal}
- Competitor analysis in the healthcare sector
- Regulatory compliance requirements
- Medical technology advancements
"""

POWER_ENERGY_PROMPT = """You are a senior data analyst at an energy sector marketing agency. Analyze the market focusing on:
- Current energy consumption patterns
- Regional energy demands for {goal}
- Competitor analysis in the energy sector
- Regulatory environment
- Renewable energy trends
"""

DOMAIN_PROMPTS = {
    "automobiles": AUTOMOBILE_PROMPT,
    "healthcare": HEALTHCARE_PROMPT,
    "powerenergy": POWER_ENERGY_PROMPT
}

def research_market_trends(state: CampaignState) -> CampaignState:
    # Initialize Tavily search
    api_key = os.environ.get("TAVILY_API_KEY")
    search = TavilyClient(api_key=api_key)
    
    # Get domain-specific prompt
    domain = state.selected_domain.lower().replace("-", "")
    domain_intro = DOMAIN_PROMPTS.get(domain, AUTOMOBILE_PROMPT)
    
    # Extract data based on domain
    sales_df = state.sales_data
    
    # Define domain-specific data columns and file prefixes
    domain_config = {
        "automobiles": {
            "columns": ["cartype", "region", "fuel_variant"],
            "file_prefix": "automotives_",
            "search_terms": ["SUV", "sedan", "electric", "hybrid", "automotive"]
        },
        "healthcare": {
            "columns": ["treatment_type", "doctor_name", "campaign_id"],
            "file_prefix": "hospital_",
            "search_terms": ["healthcare", "medical", "treatment", "patient care"]
        },
        "powerenergy": {
            "columns": ["energy_type", "region", "consumption_type"],
            "file_prefix": "energy_",
            "search_terms": ["renewable energy", "power grid", "energy consumption"]
        }
    }
    
    config = domain_config.get(domain, domain_config["automobiles"])
    columns_to_check = config["columns"]
    
    # Extract relevant information from sales data
    data_points = {}
    for col in columns_to_check:
        if col in sales_df.columns:
            data_points[col] = sales_df[col].unique()
        else:
            data_points[col] = []
    
    # Build search term based on domain and goal
    domain_search_terms = config["search_terms"]
    search_term = f"{domain} industry market trends {' '.join(domain_search_terms)} {state.goal}"
    search_results = search.search(search_term)
    
    # Get LLM - Update this line to use the selected model from state
    llm = get_llm(temperature=0.7, model_provider=state.selected_llm)
    
    # Filter and create data summary
    filtered_dict = {k: v for k, v in data_points.items() if isinstance(v, np.ndarray) and len(v) > 0}
    data_summary = ', '.join([
        f"{col}: {', '.join(map(str, values[:5]))}" + (", ..." if len(values) > 5 else "")
        for col, values in filtered_dict.items()
    ])
    
    # Combine domain-specific intro with detailed analysis requirements
    prompt = f"""
        {domain_intro.format(goal=state.goal)}

        ### 📊 Data Sources:
        1. **Internal Datasets**
        - **{domain.capitalize()} Data**: Historical data with key data points:
          {data_summary}

        2. **External Market Research**
        {search_results}

        ---

        ### 🧠 Your Analysis Should Cover:

        1. **Market Trends**
        - Identify current trends from internal + external data
        - Include any seasonal patterns

        2. **Consumer Behavior Patterns**
        - Focus on patterns relevant to the campaign goal
        - Align findings with sales data and external behavior insights

        3. **Campaign Performance Analysis**
        - Evaluate effectiveness of previous campaigns by channel, timing, and conversion
        - Recommend best strategies based on historical performance and external benchmarks

        4. **Competitive Landscape**
        - Use external research to highlight emerging trends, strategies, or competitor moves
        - Focus on factors impacting marketing outcomes

        5. **Regional Analysis**
        - Only include if the goal targets specific regions
        - Compare regional performance using both internal and external insights

        ---

        ### 📦 Output Format:

        #### 1. Executive Summary
        - 1 brief paragraph summarizing top findings, actionable insights, and suggested strategic direction.

        #### 2. Key Market Insights
        - Condense insights into clear bullet points.
        - Each point should include a **stat**, **insight**, and **why it matters**.
        - Filter automatically based on what's relevant to the campaign goal.
        - Provide only 3 bullet points.

        #### 3. Top Opportunities (Ranked)
        - List top 3 opportunities to improve performance or target growth areas.
        - Rank by potential impact (High > Medium > Low).

        #### 4. Key Performance Indicators (KPIs)
        - List 3 key performance indicators (KPIs) to track campaign success.
        - Each point should include a **metric**, **target**, and **why it matters**.

        #### 5. Recommended Actions
        - List two specific steps to optimize the campaign.
        - Cite both internal metrics and external sources for support.

        ---

        ### 🔍 Output Guidelines:
        - **DO NOT include irrelevant trends or data.** Filter everything based on what matters to the goal.
        - Use clear formatting (bullets, numbers, short paragraphs).
        - Pull actual statistics and insights directly from the external market research where applicable.
        - Insights must be **practical**, **data-backed**, and **easy to understand** for marketing stakeholders.
        """
    
    response = safe_llm_invoke(llm, prompt)
    state.market_analysis = response.content if response else "No insights available"
    return state