# Autonomous-campaign-builder

The Autonomous Campaign Builder is an AI-powered tool that generates complete marketing campaigns based on your business goals. Using advanced AI models and vector databases, it analyzes market trends, segments audiences, creates campaign strategies, generates content, and simulates campaign performance.

## Features

### Market Analysis: 
Automatically analyzes market trends, opportunities, and competitive landscape
### Target Audience Segmentation: 
Identifies primary and secondary customer segments with tailored messaging points
### Campaign Strategy Development: 
Creates comprehensive timelines, channel strategies, budget allocations, and KPIs
### Content Generation:
Produces email templates, social media posts, and landing page content
### Performance Simulation:
Projects campaign results and provides optimization recommendations
### Final Report:
Compiles a complete campaign plan for implementation
### Email Distribution:
Sends campaign emails to target customers

## Getting Started
### Prerequisites
- Python 3.8+
- Google API key for Gemini AI models
### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/campaign-builder-genai-demo.git
 ```
```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
 ```

3. Install dependencies:
```bash
pip install -r requirements.txt
    
##if need be for chromadb
    #(1. install the Microsoft C++ Build Tools:)
```bash
# Open this URL in your browser and download the installer
https://visualstudio.microsoft.com/visual-cpp-build-tools/
 ```


4. Create a .env file in the project root with your API key:
```plaintext
# Google API key
GOOGLE_API_KEY=your_api_key_here
EMBEDDING_MODEL=models/text-embedding-004

# OPEN AI API KEY
OPENAI_API_KEY=your_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# tavily key
TAVILY_API_KEY=your_tavily_api_key_here

# smtp email credentials
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
SENDER_EMAIL=your-email@example.com
EMAIL_APP_PASSWORD=your_app_password_here
 ```

### Running the Application
Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
 ```

## Usage
1. Enter your campaign goal in the sidebar (e.g., "Boost Q2 SUV sales in the Western region by 15%")
2. Click "Generate Campaign" to start the process
3. View the results in the respective tabs as they are generated
4. Download the final report when complete
5. Use the Email Distribution tab to send campaign emails to your target audience

## Project Structure
- streamlit_app.py : Main application interface
- core/ : Core functionality modules
  - vector_db.py : Vector database for knowledge retrieval
  - state.py : Campaign state management
- agents/ : AI agent implementations for each campaign step
- workflows/ : Campaign workflow definitions
- data/ : Sample data for campaigns, sales, and customer segments

## Technologies Used
- Streamlit: Web interface
- LangChain: AI orchestration
- Google Gemini & OpenAI: AI models for content generation
- ChromaDB: Vector database for semantic search
- Pandas: Data manipulation
