import os
import smtplib
import time
import random
from email.message import EmailMessage
from core.state import CampaignState
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Define a custom exception for rate limiting
class RateLimitException(Exception):
    def __init__(self, retry_after=None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds.")

# Add retry decorator for the generate_content function
@retry(
    retry=retry_if_exception_type(RateLimitException),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    reraise=True
)
def generate_content_with_retry(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        error_str = str(e)
        if "429" in error_str and "quota" in error_str.lower():
            # Extract retry delay if available
            retry_seconds = 30  # Default retry delay
            if "retry_delay" in error_str:
                import re
                match = re.search(r'seconds: (\d+)', error_str)
                if match:
                    retry_seconds = int(match.group(1)) + random.randint(1, 5)  # Add jitter
            
            print(f"Rate limit hit. Waiting for {retry_seconds} seconds before retrying...")
            time.sleep(retry_seconds)
            raise RateLimitException(retry_after=retry_seconds)
        raise

def generate_personalized_email(state: CampaignState, customer_data: dict) -> str:
    """Generate personalized email content using Gemini"""
    try:
        # Extract customer details with better handling for customers_with_emails format
        first_name = customer_data.get('first_name', '')
        last_name = customer_data.get('last_name', '')
        customer_name = f"{first_name} {last_name}".strip() if first_name or last_name else 'Valued Customer'
        
        # Handle preferences which might be in different formats
        preferences = customer_data.get('preferred_model', customer_data.get('preferred_models', ''))
        car_type = customer_data.get('preferred_cartype', '')
        fuel_type = customer_data.get('preferred_fuel_variant', '')
        
        # Get email for personalization
        email = customer_data.get('email', '')
        
        prompt = f"""
            Create a personalized marketing email based on:

            Campaign Goal: {state.goal}
            Campaign Strategy: {state.campaign_strategy[:500] if hasattr(state, 'campaign_strategy') else ''}

            Customer Details:
            - Name: {customer_name}
            - Email: {email}
            - Preferred Car Model: {preferences}
            - Preferred Car Type: {car_type}
            - Preferred Fuel Type: {fuel_type}

            The email should:
            1. Include a personalized greeting using the customer's name
            2. Reference their specific car preferences
            3. Highlight campaign benefits relevant to their preferences
            4. Include a clear call-to-action
            5. Be professional yet engaging

            EXAMPLE TEMPLATE:

            Format:
            SUBJECT: {state.goal} – Personalized Offer for {customer_name}

            Hi {customer_name},

            Looking for the perfect {car_type} that runs on {fuel_type}? We’ve got just the thing for you!

            At [Your Company Name], we noticed your interest in the {preferences}, and we’re thrilled to offer you an exclusive deal as part of our {state.goal} campaign.

            Here’s what you can look forward to:
            - 🚗 Top deals on {car_type}s like the {preferences}
            - ⛽ Options tailored to your {fuel_type} preference
            - 🎁 Limited-time bonuses when you book a test drive this week!

            Don’t miss out – click below to explore your personalized options now.

            [CTA Button – e.g., Explore Now →]

            Best,  
            [Your Name]  
            [Your Company Name]
            """

        
        # Use the retry-enabled function
        return generate_content_with_retry(prompt)
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
        base_path = 'D:\\autonomous-campaign-builder\\autonomous-campaign-builder'
        try:
            # Load the customer data
            customer_data = pd.read_csv(f'{base_path}\\data\\customers_with_emails.csv')
            
            # IMPORTANT: Limit to first 6 customers only for testing
            customer_data = customer_data.head(6)
            
            # If you want to target specific models (e.g., Honda City)
            if hasattr(state, 'target_model') and state.target_model:
                customer_data = customer_data[customer_data['preferred_model'] == state.target_model]
                
        except FileNotFoundError:
            customer_data = pd.read_csv(f'{base_path}\\data\\customer.csv')
            # Limit to first 6 customers here too
            customer_data = customer_data.head(6)
        
        # Filter valid emails
        customer_data = customer_data[customer_data['email'].notna()]
        
        if customer_data.empty:
            state.email_status = "No customer emails found"
            return state
        
        # Print info about who we're emailing
        print(f"Sending emails to {len(customer_data)} customers:")
        for _, customer in customer_data.iterrows():
            print(f"  - {customer['first_name']} {customer['last_name']} ({customer['email']})")
        
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
                    print(f"Successfully sent email to {customer['email']}")
                    
                    # Add a delay between emails to avoid rate limiting
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"Failed to send to {customer['email']}: {str(e)}")
                    failed_count += 1
            
            state.email_status = f"Emails sent: {sent_count}, Failed: {failed_count}"
            
    except Exception as e:
        state.email_status = f"Email sending failed: {str(e)}"
    
    return state