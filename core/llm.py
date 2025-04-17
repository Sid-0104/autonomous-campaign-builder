import os
import re
import time
import random
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

# Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 120))


def get_llm(temperature=0.5, model_provider=None):
    """
    Get the appropriate LLM based on the selected provider.
    
    Args:
        temperature: Float value controlling randomness in generation
        model_provider: String indicating which model to use ('openai' or 'gemini')
                       If None, will check session state or default to Gemini
    
    Returns:
        A configured LLM instance
    """
    # Try to get model selection from streamlit session state if not explicitly provided
    if model_provider is None:
        try:
            import streamlit as st
            if 'selected_llm' in st.session_state:
                model_provider = st.session_state.selected_llm
            else:
                model_provider = "gemini"  # Default
        except ImportError:
            model_provider = "gemini"  # Default if streamlit not available
    
    # Normalize model provider name to lowercase for comparison
    model_provider = model_provider.lower() if model_provider else "gemini"
    
    print(f"Model provider requested: {model_provider}")  # Debug log
    
    # Use OpenAI if selected and API key is available
    if model_provider == "openai" and OPENAI_API_KEY:
        print(f"Using OpenAI model: {OPENAI_MODEL}")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
            max_retries=MAX_RETRIES,
            request_timeout=REQUEST_TIMEOUT
        )
    # Otherwise use Gemini (default)
    else:
        if model_provider == "openai" and not OPENAI_API_KEY:
            print("Warning: OpenAI selected but API key not found. Falling back to Gemini.")
        print(f"Using Gemini model: {GEMINI_MODEL}")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
            max_retries=MAX_RETRIES,
            request_timeout=REQUEST_TIMEOUT
        )

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=5, max=60))
def safe_llm_invoke(llm, prompt):
    try:
        return llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        error_message = str(e).lower()
        
        # Handle rate limiting for both OpenAI and Gemini
        if any(term in error_message for term in ["quota", "429", "rate limit", "too many requests"]):
            wait_time = 30  # Base wait time
            if "retry_delay" in error_message:
                match = re.search(r'seconds: (\d+)', error_message)
                if match:
                    wait_time = int(match.group(1)) * 2 
            print(f"API rate limit reached. Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            raise
        
        # Handle authentication errors
        elif any(term in error_message for term in ["authentication", "auth", "key", "unauthorized", "401"]):
            print(f"Authentication error: {e}")
            # Don't retry auth errors
            raise ValueError(f"API authentication failed. Please check your API key: {e}")
        
        # Handle other errors
        else:
            print(f"Error invoking LLM: {e}")
            raise