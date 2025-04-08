from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from langchain_core.messages import HumanMessage

def segment_audience(state: CampaignState) -> CampaignState:
    llm = get_llm(temperature=0.5)
    
    # Extract key terms from goal for better segment matching
    goal_keywords = state.goal.lower().split()
    search_terms = " ".join([term for term in goal_keywords if len(term) > 3])
    
    # Get relevant customer segments
    docs = state.vector_db.similarity_search(f"customer segments interested in {search_terms}", k=2)
    segment_info = "\n\n".join([doc.page_content for doc in docs if doc.metadata["type"] == "segment"])
    
    prompt = f"""
    You're a customer insights specialist identifying target audiences.

    ### INPUTS:
    - CAMPAIGN GOAL: {state.goal}
    - MARKET ANALYSIS: {state.market_analysis[:300]}... [truncated]
    - AVAILABLE SEGMENTS: {segment_info}

    ### DELIVERABLE:
    Provide a concise audience targeting plan with:
    1. PRIMARY SEGMENT: Name, demographics, and why they're ideal (3 reasons max)
    2. SECONDARY SEGMENT: Name, demographics, and why they're secondary
    3. MESSAGING POINTS: 3-5 key points that will resonate with these segments

    Format with clear headings and bullet points.
    """
    
    response = safe_llm_invoke(llm, prompt)
    state.audience_segments = response.content if response else "Audience segmentation failed"
    return state