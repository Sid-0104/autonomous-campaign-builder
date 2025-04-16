import os
import smtplib
import time
import pandas as pd
from dotenv import load_dotenv
from email.message import EmailMessage
from core.llm import get_llm, safe_llm_invoke
from core.state import CampaignState
from typing import Dict
import datetime

# Load environment variables
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

AUTOMOBILE_PROMPT = """You're an email marketing specialist in the automotive industry.
Focus on:
- Vehicle promotion and special offers
- Test drive invitations and dealership events
- Owner loyalty and service reminders
- New model announcements for {goal}
- Financing and leasing options
"""

HEALTHCARE_PROMPT = """You're an email marketing specialist in the healthcare industry.
Focus on:
- Patient appointment reminders and follow-ups
- Health education and preventive care
- Provider introductions and specialties
- Service announcements for {goal}
- Insurance and payment options
"""

POWER_ENERGY_PROMPT = """You're an email marketing specialist in the power and energy sector.
Focus on:
- Energy saving tips and efficiency programs
- Service upgrades and new offerings
- Smart technology integration
- Sustainability initiatives for {goal}
- Billing options and payment plans
"""

DOMAIN_PROMPTS = {
    "automobiles": AUTOMOBILE_PROMPT,
    "healthcare": HEALTHCARE_PROMPT,
    "powerenergy": POWER_ENERGY_PROMPT
}

def generate_content_with_retry(prompt, state: CampaignState, max_retries=3):
    """Retry content generation with exponential backoff"""
    llm = get_llm(temperature=0.4, model_provider=state.selected_llm)
    retries = 0
    
    while retries < max_retries:
        try:
            response = safe_llm_invoke(llm, prompt)
            if response:
                return response.content
            else:
                retries += 1
                time.sleep(2 ** retries)  # Exponential backoff
        except Exception as e:
            print(f"Error generating content (attempt {retries+1}): {str(e)}")
            retries += 1
            time.sleep(2 ** retries)
    
    return ""

def generate_personalized_email(state: CampaignState, customer_data: dict) -> str:
    """Generate personalized email content using Gemini"""
    try:
        # Extract customer details with better handling for customers_with_emails format
        full_name = customer_data.get('full_name', '')
        name_parts = full_name.split() if full_name else []
        first_name = name_parts[0] if name_parts else ''
        last_name = name_parts[-1] if len(name_parts) > 1 else ''
        customer_name = full_name if full_name else 'Valued Customer'
        
        # Get email for personalization
        email = customer_data.get('email', '')
        
        # Get domain-specific prompt
        domain = state.selected_domain.lower().replace("-", "")
        domain_intro = DOMAIN_PROMPTS.get(domain, AUTOMOBILE_PROMPT).format(goal=state.goal)
        
        prompt = f"""
            {domain_intro}

            Create a personalized marketing email based on:

            Campaign Goal: {state.goal}
            Campaign Strategy: {state.campaign_strategy[:500] if hasattr(state, 'campaign_strategy') else ''}

            Customer Details:
            - Name: {customer_name}
            - Email: {email}
            - Age: {customer_data.get('age', 'Unknown')}
            - Gender: {customer_data.get('gender', 'Unknown')}

            The email should:
            1. Include a personalized greeting using the customer's name
            2. Reference their demographic information appropriately
            3. Highlight campaign benefits relevant to their demographic
            4. Include a clear call-to-action
            5. Be professional yet engaging

            Format:
            SUBJECT: Personalized Offer for {customer_name} - {state.goal}

            Hi {customer_name},

            [Body: Create compelling content that aligns with campaign goals and customer demographics]

            [CTA Button]

            Best regards,
            [Company Name]
        """
        
        # Use the retry-enabled function with state parameter
        return generate_content_with_retry(prompt, state)
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
        # Load customer data from filtered_customers.csv
        customer_data_path = os.path.join(script_dir, 'data', 'filtered_customers.csv')
        
        try:
            customer_data = pd.read_csv(customer_data_path)
        except FileNotFoundError:
            state.email_status = "Email sending failed: Customer data file not found"
            return state
        
        # Filter valid emails
        customer_data = customer_data[customer_data['email'].notna()]
        
        if customer_data.empty:
            state.email_status = "No customer emails found"
            return state
        
        # For testing/development, limit to first few customers
        # Remove this line in production
        customer_data = customer_data.head(4)
        
        # Connect to SMTP server
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            
            sent_count = 0
            failed_count = 0
            sent_emails = []
            
            for _, customer in customer_data.iterrows():
                try:
                    # Generate personalized email
                    email_content = generate_personalized_email(state, customer.to_dict())
                    
                    if not email_content:
                        failed_count += 1
                        continue
                    
                    # Parse subject and body
                    if "SUBJECT:" in email_content:
                        parts = email_content.split("SUBJECT:", 1)
                        subject_body = parts[1].strip()
                        subject, body = subject_body.split("\n", 1)
                        subject = subject.strip()
                        body = body.strip()
                    else:
                        subject = f"Special Offer: {state.goal}"
                        body = email_content
                    
                    # Create email message
                    msg = EmailMessage()
                    msg.set_content(body)
                    msg['Subject'] = subject
                    msg['From'] = sender_email
                    msg['To'] = customer['email']
                    
                    # Send email
                    server.send_message(msg)
                    sent_count += 1
                    
                    if not hasattr(state, 'email_templates') or not isinstance(state.email_templates, list):
                        state.email_templates = []


                    # Store template if first email
                    if not state.email_templates:
                        state.email_templates.append({
                            'subject': subject,
                            'content': body
                        })
                    
                    # Track sent emails
                    sent_emails.append({
                        'customer_id': customer.get('customer_id', ''),
                        'name': customer.get('full_name', ''),
                        'email': customer.get('email', ''),
                        'subject': subject,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    print(f"Error sending email to {customer.get('email', '')}: {str(e)}")
                    failed_count += 1
            
            state.sent_emails = sent_emails
            state.email_status = f"""Email Campaign Summary:
            - Total recipients: {len(customer_data)}
            - Successfully sent: {sent_count}
            - Failed to send: {failed_count}
            """
            
    except Exception as e:
        state.email_status = f"Email sending failed: {str(e)}"
    
    return state