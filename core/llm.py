import os
import re
import time
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 120))

def get_llm(temperature=0.5):
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
        if "quota" in str(e).lower() or "429" in str(e):
            wait_time = 30  # Base wait time
            if "retry_delay" in str(e):
                match = re.search(r'seconds: (\d+)', str(e))
                if match:
                    wait_time = int(match.group(1)) * 2 
            print(f"API rate limit reached. Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            raise
        print(f"Error invoking LLM: {str(e)}")
        raise