import os
import uuid
import datetime
import pandas as pd

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEEDBACK_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "feedback"))
FEEDBACK_FILE = os.path.normpath(os.path.join(FEEDBACK_DIR, "node_feedback.csv"))

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


def init_feedback_system():
    """Initialize the feedback system for testing purposes"""
    print("Initializing feedback system...")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Feedback Directory: {FEEDBACK_DIR}")
    print(f"Feedback File: {FEEDBACK_FILE}")
    
    # Initialize CSV file
    init_feedback_csv()
    
    # Test saving feedback
    test_result = save_node_feedback("test_node", 5)
    print(f"Test feedback saved: {test_result}")
    
    # Verify file exists and has content
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        print(f"Feedback file contains {len(df)} records")
        print("Sample record:")
        print(df.head(1))
    else:
        print("Error: Feedback file not created")
    
    return "Feedback system initialized successfully"