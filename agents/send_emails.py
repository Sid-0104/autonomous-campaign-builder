import os
import smtplib
from email.message import EmailMessage
from core.state import CampaignState
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

def generate_personalized_email(state: CampaignState, customer_data: dict) -> str:
    """Generate personalized email content using Gemini"""
    try:
        # Extract customer details with better handling for customer_with_emails format
        customer_name = customer_data.get('name', customer_data.get('customer_name', 'Valued Customer'))
        
        # Handle preferences which might be in different formats
        preferences = customer_data.get('preferred_models', customer_data.get('preferences', ''))
        if isinstance(preferences, str) and preferences.startswith('['):
            try:
                preferences = eval(preferences)
                preferences = ', '.join(preferences)
            except:
                pass
                
        # Get region with fallbacks
        region = customer_data.get('region', customer_data.get('location', ''))
        
        # Get email for personalization
        email = customer_data.get('email', '')
        
        prompt = f"""
        Create a personalized marketing email based on:
        
        Campaign Goal: {state.goal}
        Campaign Strategy: {state.campaign_strategy[:500] if hasattr(state, 'campaign_strategy') else ''}
        
        Customer Details:
        - Name: {customer_name}
        - Email: {email}
        - Preferences: {preferences}
        - Region: {region}
        
        The email should:
        1. Include a personalized greeting using the customer's name
        2. Reference their specific preferences if available
        3. Highlight campaign benefits relevant to their region
        4. Include a clear call-to-action
        5. Be professional yet engaging
        
        Format:
        SUBJECT: [Campaign Name] - Personalized Offer for {customer_name}
        
        [Email Body with personalization]
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating email content: {str(e)}")
        return ""

def send_campaign_emails(state: CampaignState) -> CampaignState:
    """Send personalized emails to customers using campaign state and customer data"""
    # Email configuration
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("EMAIL_APP_PASSWORD")
    
    if not all([sender_email, sender_password]):
        state.email_status = "Email sending failed: Missing email credentials"
        return state
    
    try:
        # Load customer data
        base_path = 'c:\\Users\\user\\Desktop\\campaign builder'
        try:
            customer_data = pd.read_csv(f'{base_path}\\data\\customer_with_emails.csv')
        except FileNotFoundError:
            customer_data = pd.read_csv(f'{base_path}\\data\\customer.csv')
        
        # Filter valid emails
        customer_data = customer_data[customer_data['email'].notna()]
        
        if customer_data.empty:
            state.email_status = "No customer emails found"
            return state
        
        # Connect to SMTP server
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            
            sent_count = 0
            failed_count = 0
            
            for _, customer in customer_data.iterrows():
                try:
                    # Generate personalized email
                    email_content = generate_personalized_email(state, customer.to_dict())
                    
                    if not email_content:
                        failed_count += 1
                        continue
                    
                    # Parse subject and body
                    if "SUBJECT:" in email_content:
                        subject, body = email_content.split("SUBJECT:", 1)[1].split("\n\n", 1)
                        subject = subject.strip()
                        body = body.strip()
                    else:
                        subject = f"Special Offer from Our Campaign"
                        body = email_content
                    
                    # Create and send email
                    msg = EmailMessage()
                    msg['From'] = sender_email
                    msg['To'] = customer['email']
                    msg['Subject'] = subject
                    msg.set_content(body)
                    
                    server.send_message(msg)
                    sent_count += 1
                    
                except Exception as e:
                    print(f"Failed to send to {customer['email']}: {str(e)}")
                    failed_count += 1
            
            state.email_status = f"Emails sent: {sent_count}, Failed: {failed_count}"
            
    except Exception as e:
        state.email_status = f"Email sending failed: {str(e)}"
    
    return state