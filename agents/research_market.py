from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from tavily import TavilyClient
import os
import pandas as pd

def research_market_trends(state: CampaignState) -> CampaignState:
    # Initialize Tavily search
    api_key = os.environ.get("TAVILY_API_KEY")
    search = TavilyClient(api_key=api_key)
    
    # Extract key terms from goal and sales data for more targeted search
    sales_df = state.sales_data
    
    # Extract relevant information from sales data
    car_types = sales_df['cartype'].unique() if 'cartype' in sales_df.columns else []
    regions = sales_df['region'].unique() if 'region' in sales_df.columns else []
    fuel_types = sales_df['fuel_variant'].unique() if 'fuel_variant' in sales_df.columns else []
    
    # Build search term based on goal and available data
    search_terms = []
    
    # Add car types from data if mentioned in goal
    for car_type in car_types:
        if car_type.lower() in state.goal.lower():
            search_terms.append(car_type)
    
    # Add regions from data if mentioned in goal
    for region in regions:
        if region.lower() in state.goal.lower():
            search_terms.append(region)
    
    # Add fuel types if mentioned in goal (especially for EV/hybrid trends)
    for fuel in fuel_types:
        if fuel.lower() in state.goal.lower():
            search_terms.append(fuel)
    
    # Combine search terms
    search_term_str = " ".join(search_terms)
    search_term = f"automotive industry market trends {search_term_str} {state.goal}"
    
    # Perform search
    search_results = search.search(search_term)
    
    # Get LLM
    llm = get_llm(temperature=0.7)
    
    prompt = f"""
        You are a senior data analyst at an automotive marketing agency. You are tasked with generating a comprehensive and strategic market analysis for an upcoming campaign.

        ---

        ### 🎯 Campaign Objective:
        {state.goal}

        ---

        ### 📊 Data Sources:
        You must use and reference insights from the following:

        1. **Internal Datasets**
        - **Sales Data**: Historical vehicle sales by model, date, region, etc. Key data points:
          - Car types: {', '.join(car_types)}
          - Regions: {', '.join(regions)}
          - Fuel variants: {', '.join(fuel_types)}

        2. **External Market Research**  
        Present these insights in **concise bullet points**, filtering only **relevant data based on the campaign goal**.  
        Use the following source:
        {search_results}

        ---

        ### 🧠 Your Analysis Should Cover:

        1. **Market Trends**
        - Identify current vehicle sales trends from internal + external data.
        - Include any seasonal patterns or technology-driven shifts (e.g., EVs, hybrid adoption).

        2. **Consumer Buying Patterns**
        - Focus only if relevant to the goal (e.g., car type preferences, regions, fuel variants).
        - Align findings with sales data and external behavior insights.

        3. **Campaign Performance Analysis**
        - Evaluate effectiveness of previous campaigns by channel, timing, and conversion.
        - Recommend best strategies based on what's worked and external benchmarks.

        4. **Competitive and Technology Landscape**
        - Use external research to highlight emerging technologies, digital strategies, or competitor moves impacting marketing outcomes.

        5. **Regional Market Analysis**
        - Only include if the goal targets specific regions.
        - Compare regional performance using both internal and external insights.

        ---

        ### 📦 Output Format:

        #### 1. Executive Summary
        - 2–3 brief paragraphs summarizing top findings, actionable insights, and suggested strategic direction.

        #### 2. Key Market Insights (Bullet Format)
        - Condense insights into clear bullet points.
        - Each point should include a **stat**, **insight**, and **why it matters**.
        - Filter automatically based on what's relevant to the campaign goal.

        #### 3. Top Opportunities (Ranked)
        - List top opportunities to improve performance or target growth areas.
        - Rank by potential impact (High > Medium > Low).

        #### 4. Threats & Risks
        - Identify challenges based on internal weaknesses or external market dynamics.
        - Add mitigation strategies for each.

        #### 5. Recommended Actions
        - List specific steps to optimize the campaign.
        - Cite both internal metrics and external sources for support.

        ---

        ### 🔍 Output Guidelines:
        - **DO NOT include irrelevant trends or data.** Filter everything based on what matters to the goal.
        - Use clear formatting (bullets, numbers, short paragraphs).
        - Pull actual statistics and insights directly from `{search_results}` where applicable.
        - Insights must be **practical**, **data-backed**, and **easy to understand** for marketing stakeholders.

        """

    
    response = safe_llm_invoke(llm, prompt)
    state.market_analysis = response.content if response else "No insights available"
    return state