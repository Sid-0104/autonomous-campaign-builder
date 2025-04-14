import os
import random
from tavily import TavilyClient

def research_market_trends(state):
    goal = state.goal

    try:
        # Prompt variations to improve regenerate diversity
        query_templates = [
            f"latest marketing trends and top competitors for {goal}",
            f"emerging industry trends for {goal} and similar campaigns",
            f"market challenges and opportunities for {goal} campaign",
            f"recent innovations and market moves related to {goal}"
        ]
        selected_query = random.choice(query_templates)

        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.search(query=selected_query)
        results = response.get("results", [])

        if not results:
            raise ValueError("No search results")

        # Shuffle results for slight regeneration variation
        random.shuffle(results)

        report = "\n".join([
            f"- **{r['title']}**: {r.get('content') or r.get('url', '')}"
            for r in results[:5]  # Limit to top 5
        ])

        return {
            "market_analysis": f"🔍 **Market Insights for:** `{goal}`\n\n{report}"
        }

    except Exception as e:
        print(f"[⚠️ Tavily Failed] {e}")
        return {
            "market_analysis": f"⚠️ Market research could not be fetched.\n\n**Reason:** {str(e)}"
        }
