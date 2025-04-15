from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from langchain_core.messages import HumanMessage
from typing import Dict

AUTOMOBILE_PROMPT = """You're a customer insights specialist in the automotive industry.
Focus on:
- Vehicle preferences by demographic
- Purchase behaviors (new vs. used, financing patterns)
- Lifestyle factors influencing automotive choices
- Regional preferences for {goal}
- Fuel/powertrain preferences (electric, hybrid, gas)
"""

HEALTHCARE_PROMPT = """You're a customer insights specialist in the healthcare industry.
Focus on:
- Patient demographics and service utilization
- Healthcare needs by age group and condition
- Insurance coverage and payment preferences
- Regional healthcare access for {goal}
- Preventive vs. treatment service preferences
"""

POWER_ENERGY_PROMPT = """You're a customer insights specialist in the power and energy sector.
Focus on:
- Energy consumption patterns by demographic
- Renewable vs. traditional energy preferences
- Price sensitivity for energy services
- Regional energy needs for {goal}
- Sustainability consciousness and green energy adoption
"""

DOMAIN_PROMPTS = {
    "automobiles": AUTOMOBILE_PROMPT,
    "healthcare": HEALTHCARE_PROMPT,
    "powerenergy": POWER_ENERGY_PROMPT
}

def segment_audience(state: CampaignState) -> CampaignState:
    # Update this line to use the selected model from state
    llm = get_llm(temperature=0.5, model_provider=state.selected_llm)
    
    # Get domain-specific prompt
    domain = state.selected_domain.lower().replace("-", "")
    domain_intro = DOMAIN_PROMPTS.get(domain, AUTOMOBILE_PROMPT).format(goal=state.goal)
    
    # Extract key terms from goal for better segment matching
    goal_keywords = state.goal.lower().split()
    search_terms = " ".join([term for term in goal_keywords if len(term) > 3])
    
    # Handle case where vector_db is None
    segment_info = ""
    if state.vector_db is not None:
        try:
            # Get relevant customer segments
            docs = state.vector_db.similarity_search(f"{domain} customer segments interested in {search_terms}", k=2)
            
            # Fix: Check for 'type' key in metadata or use all documents
            segment_info = "\n\n".join([doc.page_content for doc in docs if doc.metadata.get("category", "") == "customer"])
            
            # If no customer segments found, use all documents
            if not segment_info:
                segment_info = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error searching vector database: {e}")
            segment_info = "Vector database search failed. Using default segmentation."
    else:
        print("Vector database not available. Using default segmentation.")
        segment_info = "Vector database not available. Using default segmentation."
    
    # Domain-specific segment fields
    segment_fields = {
        "automobiles": "vehicle preferences, purchase type, fuel variant",
        "healthcare": "treatment preferences, insurance type, visit frequency",
        "powerenergy": "energy consumption, sustainability preferences, price sensitivity"
    }
    
    fields = segment_fields.get(domain, segment_fields["automobiles"])
    
    # Domain-specific default segments when vector DB is not available
    default_segments = {
        "automobiles": """
        - Segment 1: Urban professionals (25-45), prefer SUVs, interested in hybrid/electric options
        - Segment 2: Suburban families (30-50), prefer larger vehicles, safety-conscious
        - Segment 3: Outdoor enthusiasts (20-60), prefer rugged vehicles with off-road capabilities
        """,
        "healthcare": """
        - Segment 1: Young families (25-40), preventive care focus, digital-first approach
        - Segment 2: Seniors (65+), chronic condition management, prefer in-person care
        - Segment 3: Middle-aged adults (40-65), wellness-oriented, mix of virtual and in-person
        """,
        "powerenergy": """
        - Segment 1: Eco-conscious homeowners, interested in renewable energy, willing to pay premium
        - Segment 2: Budget-conscious consumers, focused on cost savings and efficiency
        - Segment 3: Tech-savvy early adopters, interested in smart home integration
        """
    }
    
    # If no segment info from vector DB, use default segments
    if segment_info in ["Vector database search failed. Using default segmentation.", 
                        "Vector database not available. Using default segmentation."]:
        segment_info = default_segments.get(domain, default_segments["automobiles"])
    
    prompt = f"""
    {domain_intro}

    ### INPUTS:
    - CAMPAIGN GOAL: {state.goal}
    - MARKET ANALYSIS: {state.market_analysis[:300] if state.market_analysis else "No market analysis available"}... [truncated]
    - AVAILABLE SEGMENTS: {segment_info}

    ### DELIVERABLE:
    Provide a concise audience targeting plan with:
    1. PRIMARY SEGMENT: 
       - Profile description (including {fields})
       - Key characteristics
       - Why they're ideal (3 reasons max)
    
    2. SECONDARY SEGMENT:
       - Profile description (including {fields})
       - Key characteristics
       - Why they're secondary
    
    3. MESSAGING POINTS:
       - 3-5 key points that will resonate with these segments
       - Value propositions specific to each segment

    Format with clear headings and bullet points.
    """
    
    response = safe_llm_invoke(llm, prompt)
    state.audience_segments = response.content if response else "Audience segmentation failed"
    return state