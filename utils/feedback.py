import os
import uuid
import datetime
import pandas as pd

FEEDBACK_DIR = "feedback"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "node_feedback.csv")

def init_feedback_csv():
    """Create feedback folder and CSV file if it doesn't exist."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame(columns=["id", "timestamp", "node_name", "user_rating"])
        df.to_csv(FEEDBACK_FILE, index=False)

def save_node_feedback(node_name, user_rating):
    """Append feedback to CSV file inside feedback directory."""
    feedback = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "node_name": node_name,
        "user_rating": user_rating
    }
    df = pd.DataFrame([feedback])
    df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    return "✅ Feedback submitted!"