import os
import smtplib
import time
import re
import pandas as pd
from dotenv import load_dotenv
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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
            1. Include a personalized greeting using the customer's name in the body ONLY
            2. Reference their demographic information appropriately
            3. Highlight campaign benefits relevant to their demographic
            4. Include a clear call-to-action
            5. Be professional yet engaging
            6. Use this format exactly:

            SUBJECT: <Write a compelling subject line here>

            BODY:
            <Start with a greeting using the customer's first name. Then provide engaging, benefit-driven content tailored to the demographics. Include a clear CTA. Do NOT wrap anything in asterisks. Do NOT repeat the subject line in the body.>
            """
        
        # Use the retry-enabled function with state parameter
        return generate_content_with_retry(prompt, state)
    except Exception as e:
        print(f"Error generating email content: {str(e)}")
        return ""

def process_email_formatting(content: str, customer_data: dict) -> str:
    """
    Process email formatting:
    1. Replace asterisk-wrapped text with appropriate formatting
    2. Replace custom variables with customer data
    """
    # First handle any customer data variables that might need substitution
    variables = {
        'CUSTOMER_NAME': customer_data.get('full_name', 'Valued Customer'),
        'FIRST_NAME': customer_data.get('full_name', '').split()[0] if customer_data.get('full_name') else 'Valued Customer',
        'EMAIL': customer_data.get('email', ''),
        'AGE': str(customer_data.get('age', '')),
        'GENDER': customer_data.get('gender', '')
    }
    
    # Replace variables in content
    for var, value in variables.items():
        content = content.replace(f"%{var}%", value)
    
    # Process text between double asterisks for HTML email
    # In HTML email, we'll convert **text** to <strong>text</strong>
    if '**' in content:
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    
    return content

def parse_email_content(email_content: str) -> tuple[str, str]:
    """Parse LLM output into a subject and body, and remove extra markers from the body."""
    subject = "Special Offer"
    body = email_content.strip()

    # Match subject line only
    subject_match = re.search(r"^SUBJECT:\s*(.+)$", body, re.MULTILINE)
    if subject_match:
        subject = subject_match.group(1).strip()
        # Remove the subject line from the body completely
        body = re.sub(r"^SUBJECT:.*$", "", body, flags=re.MULTILINE).strip()

    # Match only what's after BODY:
    body_match = re.search(r"^BODY:\s*(.*)", body, re.DOTALL | re.IGNORECASE)
    if body_match:
        body = body_match.group(1).strip()
    
    # Remove any accidental embedded SUBJECT or BODY tags in the body content
    body = re.sub(r"^SUBJECT:.*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"^BODY:\s*", "", body, flags=re.MULTILINE)
    
    # Additional check to ensure we don't duplicate the subject in the body
    if body.startswith(subject):
        body = body[len(subject):].strip()
    
    # Remove any "Subject:" text from the beginning of the body
    body = re.sub(r'^Subject:.*?\n', '', body, flags=re.IGNORECASE | re.MULTILINE)
        
    return subject, body.strip()

def convert_bullet_points_to_html(content):
    """Convert bullet points to proper HTML lists"""
    # Check if we have bullet points (lines starting with - or • or *)
    bullet_pattern = r'(^|\n)[-•*]\s+(.*?)(?=(\n[-•*]|\n\n|\n$|$))'
    if re.search(bullet_pattern, content, re.DOTALL):
        # Start a list
        content = re.sub(r'(?:^|\n)(Exclusive Offer Just for You:)(\s*?)(?=\n[-•*])', r'\1\2\n<ul class="offer-list">', content)
        
        # Convert each bullet point to a list item
        content = re.sub(bullet_pattern, r'\1<li class="offer-item">\2</li>', content, flags=re.DOTALL)
        
        # Close the list before "Key Features" section
        content = re.sub(r'(</li>)(\s*?)(?=\nKey Features)', r'\1\n</ul>\2', content)
        
        # Handle Key Features section if it has bullet points too
        if re.search(r'\nKey Features.*?\n[-•*]', content, re.DOTALL):
            content = re.sub(r'(\nKey Features.*?)(\s*?)(?=\n[-•*])', r'\1\2\n<ul class="features-list">', content)
            # Convert feature bullet points and close list
            feature_bullets = re.findall(r'(\n[-•*]\s+.*?)(?=(\n\n|\n$|$))', content[content.find('Key Features'):], re.DOTALL)
            for bullet, _ in feature_bullets:
                content = content.replace(bullet, f'<li class="feature-item">{bullet.strip()[2:].strip()}</li>')
            
            # Close features list if not already closed
            if '<ul class="features-list">' in content and '</ul>' not in content[content.find('<ul class="features-list">'):]:
                content += '\n</ul>'
    
    return content

def send_campaign_emails(state: CampaignState) -> tuple[CampaignState, int]:
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
        customer_data = customer_data.head(1)
        
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
                    
                    # Parse subject and body using the dedicated function
                    subject, body = parse_email_content(email_content)
                    
                    # Process formatting in the email body
                    processed_body = process_email_formatting(body, customer.to_dict())
                    
                    # Always use HTML for better formatting
                    msg = MIMEMultipart('alternative')
                    
                    # Add plain text version - better removal of HTML tags
                    plain_text = re.sub(r'<[^>]+>', '', processed_body)
                    part1 = MIMEText(plain_text, 'plain')
                    msg.attach(part1)
                    
                    # Convert bullet points to proper HTML
                    html_body = convert_bullet_points_to_html(processed_body)
                    
                    # Improved HTML version with proper styling
                    html_content = f"""
                    <html>
                    <head>
                        <style>
                            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
                            .offer-list, .features-list {{ list-style-type: disc; margin-left: 20px; padding-left: 0; }}
                            .offer-item, .feature-item {{ margin-bottom: 10px; }}
                            strong {{ font-weight: bold; }}
                            h3, h4 {{ margin-top: 20px; margin-bottom: 10px; }}
                            .cta {{ margin-top: 20px; font-weight: bold; }}
                        </style>
                    </head>
                    <body>
                        <div>
                            {html_body.replace('\n', '<br>')}
                        </div>
                    </body>
                    </html>
                    """
                    
                    part2 = MIMEText(html_content, 'html')
                    msg.attach(part2)
                    
                    # Set message headers
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
                            'content': processed_body
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
            state.email_status = f"""  Summary:
            - Total recipients: {len(customer_data)}
            - Successfully sent: {sent_count}
            - Failed to send: {failed_count}
            """
            
    except Exception as e:
        state.email_status = f"Email sending failed: {str(e)}"
    
    return state