# Autonomous Campaign Builder 🚀

The **Autonomous Campaign Builder** is an AI-driven application built using Streamlit that helps marketing professionals and businesses automatically generate marketing campaigns from a single goal. Using advanced AI models and vector databases, it analyzes market trends, segments audiences, creates campaign strategies, generates content, and simulates campaign performance across various domains such as **automotive**, **healthcare**, and **energy**.

## ✨ Features

### 📊 Market Analysis
Automatically analyzes market trends, opportunities, and competitive landscape

### 👥 Target Audience Segmentation
Identifies primary and secondary customer segments with tailored messaging points

### 📝 Campaign Strategy Development
Creates comprehensive timelines, channel strategies, budget allocations, and KPIs

### ✍️ Content Generation
Produces email templates, social media posts, and landing page content

### 📈 Performance Simulation
Projects campaign results and provides optimization recommendations

### 📑 Final Report
Compiles a complete campaign plan for implementation

### 📧 Email Distribution
Sends campaign emails to target customers

## 🖥️ How to Use the UI

This application is designed for simplicity, even for non-technical users.

### 1. **Select LLM (AI Brain)**
Choose between:
- **Gemini** (Google's model)
- **OpenAI** (like ChatGPT)

📌 The app will use your selected model to generate responses.

### 2. **Select Domain (Industry)**
Choose the industry relevant to your campaign:
- Automotives
- Healthcare
- Power & Energy

🔍 This ensures that strategies are tailored to your market.

### 3. **Enter Your Campaign Goal**
Example: `Increase SUV sales in Q2` or `Boost solar panel adoption in South India`.

🧠 Based on this goal, the AI will:
- Analyze the market
- Identify your audience
- Build strategy
- Generate content
- Simulate performance
- Create a final report
- Prepare email drafts

### 4. **Generate Campaign**
Click `Generate` and wait. The system will:
- Show a progress bar
- Walk through 7 stages of campaign creation

✅ Once complete, you'll see 7 **Tabs** — each with their section's output.

### 5. **Download PDF**
Inside each tab, click `📥 Download PDF` to download that section as a styled, branded PDF (with InfoObjects footer and report name).

### 6. **Submit Feedback**
Each section offers:
- 👍 Like / 👎 Dislike buttons
- Your feedback helps improve AI output quality

### 7. **Regenerate Campaign**
Didn't like the output? Click `♻️ Regenerate` to re-run the full campaign process using a fresh AI prompt.

### 8. **Email Distribution**
- Review auto-generated email templates
- Click `📧 Send Emails` to email them to selected users (from `filtered_customers.csv`)

## 📋 Output Tabs
The generated campaign is split into 7 tabs:
1. **📊 Market Analysis**
2. **👥 Target Audience**
3. **📝 Campaign Strategy**
4. **✍️ Content**
5. **📈 Simulation**
6. **📑 Final Report**
7. **📧 Email Distribution**

## 🔧 Project Structure
- Python + Streamlit
- FPDF for PDF generation
- LangChain for AI orchestration
- ChromaDB for vector search
- Google Gemini or OpenAI LLMs

## Getting Started

### Prerequisites
- Python 3.8+
- Google API key for Gemini AI models
- (Optional) OpenAI API key for alternative model support

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autonomous-campaign-builder.git
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. If needed for ChromaDB, install the Microsoft C++ Build Tools:
   - Visit: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Download and run the installer

5. Create a .env file in the project root with your API keys:
```plaintext
# Google API key
GOOGLE_API_KEY=your_api_key_here
EMBEDDING_MODEL=models/text-embedding-004

# OPEN AI API KEY
OPENAI_API_KEY=your_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Tavily key
TAVILY_API_KEY=your_tavily_api_key_here

# SMTP email credentials
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
SENDER_EMAIL=your-email@example.com
EMAIL_APP_PASSWORD=your_app_password_here
```

### Running the Application

Launch the Streamlit app:

```bash
streamlit run streamlit_app.py