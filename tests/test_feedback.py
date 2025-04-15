import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch
import uuid
import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the functions to test
from utils.feedback import init_feedback_csv, save_node_feedback, FEEDBACK_DIR, FEEDBACK_FILE

@pytest.fixture
def mock_feedback_dir(monkeypatch, tmp_path):
    """Create a temporary feedback directory for testing"""
    test_feedback_dir = tmp_path / "feedback"
    test_feedback_file = test_feedback_dir / "node_feedback.csv"
    
    # Patch the constants in the feedback module
    monkeypatch.setattr("utils.feedback.FEEDBACK_DIR", str(test_feedback_dir))
    monkeypatch.setattr("utils.feedback.FEEDBACK_FILE", str(test_feedback_file))
    
    return test_feedback_dir, test_feedback_file

def test_init_feedback_csv(mock_feedback_dir):
    """Test that init_feedback_csv creates the directory and file correctly"""
    test_dir, test_file = mock_feedback_dir
    
    # Run the function
    init_feedback_csv()
    
    # Check that directory was created
    assert test_dir.exists()
    
    # Check that file was created with correct columns
    assert test_file.exists()
    df = pd.read_csv(test_file)
    assert list(df.columns) == ["id", "timestamp", "node_name", "user_rating"]
    assert len(df) == 0  # Should be empty

def test_init_feedback_csv_existing_file(mock_feedback_dir):
    """Test that init_feedback_csv doesn't overwrite existing file"""
    test_dir, test_file = mock_feedback_dir
    
    # Create directory and file with some data
    test_dir.mkdir(exist_ok=True)
    initial_df = pd.DataFrame({
        "id": ["test-id"],
        "timestamp": ["2023-01-01T12:00:00"],
        "node_name": ["test_node"],
        "user_rating": [5]
    })
    initial_df.to_csv(test_file, index=False)
    
    # Run the function
    init_feedback_csv()
    
    # Check that file still has the original data
    df = pd.read_csv(test_file)
    assert len(df) == 1
    assert df.iloc[0]["id"] == "test-id"

@patch("uuid.uuid4")
@patch("datetime.datetime")
def test_save_node_feedback(mock_datetime, mock_uuid, mock_feedback_dir):
    """Test that save_node_feedback correctly saves feedback"""
    test_dir, test_file = mock_feedback_dir
    
    # Mock uuid and datetime
    mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
    mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, 12, 0, 0)
    
    # Initialize the CSV file
    init_feedback_csv()
    
    # Save feedback
    result = save_node_feedback("test_node", 4)
    
    # Check return message
    assert result == "✅ Feedback submitted!"
    
    # Check that feedback was saved correctly
    df = pd.read_csv(test_file)
    assert len(df) == 1
    assert df.iloc[0]["id"] == "12345678-1234-5678-1234-567812345678"
    assert df.iloc[0]["timestamp"] == "2023-01-01T12:00:00"
    assert df.iloc[0]["node_name"] == "test_node"
    assert df.iloc[0]["user_rating"] == 4

def test_save_multiple_feedback_entries(mock_feedback_dir):
    """Test that multiple feedback entries can be saved"""
    test_dir, test_file = mock_feedback_dir
    
    # Initialize the CSV file
    init_feedback_csv()
    
    # Save multiple feedback entries
    save_node_feedback("node1", 5)
    save_node_feedback("node2", 3)
    save_node_feedback("node3", 4)
    
    # Check that all entries were saved
    df = pd.read_csv(test_file)
    assert len(df) == 3
    assert df.iloc[0]["node_name"] == "node1"
    assert df.iloc[1]["node_name"] == "node2"
    assert df.iloc[2]["node_name"] == "node3"
