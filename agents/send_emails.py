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
            retry_seconds = 30
            if "retry_delay" in error_str:
                import re
                match = re.search(r'seconds: (\d+)', error_str)
                if match:
                    retry_seconds = int(match.group(1)) + random.randint(1, 5)
            print(f"Rate limit hit. Waiting for {retry_seconds} seconds before retrying...")
            time.sleep(retry_seconds)
            raise RateLimitException(retry_after=retry_seconds)
        raise

def generate_personalized_email(state: CampaignState, customer_data: dict) -> str:
    try:
        first_name = customer_data.get('first_name', '')
        last_name = customer_data.get('last_name', '')
        customer_name = f"{first_name} {last_name}".strip() or 'Valued Customer'
        email = customer_data.get('email', '')

        # Collect preferences
        preference_keys = [k for k in customer_data.keys() if 'prefer' in k.lower() or 'interest' in k.lower()]
        preferences = []
        for key in preference_keys:
            value = customer_data.get(key, '')
            if isinstance(value, list):
                preferences.extend([str(item) for item in value if item])
            elif value:
                preferences.append(str(value))
        preferences_str = ', '.join(preferences) or 'No specific preferences'

        # === Prompt Variations ===
        prompt_variants = [
            f"""
            Create a personalized marketing email.

            Campaign Goal: {state.goal}
            Strategy: {state.campaign_strategy[:500]}

            Customer:
            - Name: {customer_name}
            - Email: {email}
            - Preferences: {preferences_str}

            Email must include:
            - Personalized greeting
            - Benefits tied to preferences
            - Strong call to action
            """,

            f"""
            Write a campaign email tailored to this customer.

            GOAL: {state.goal}
            STRATEGY: {state.campaign_strategy[:500]}
            NAME: {customer_name}
            INTERESTS: {preferences_str}

            Make it:
            - Friendly and persuasive
            - Focused on customer benefits
            - CTA-driven
            """,

            f"""
            You're writing a 1:1 marketing email.

            GOAL: {state.goal}
            CUSTOMER NAME: {customer_name}
            PREFERENCES: {preferences_str}

            Include:
            • Engaging subject line
            • Value proposition based on interests
            • Clear CTA

            Be brief and engaging.
            """
        ]

        prompt = random.choice(prompt_variants)
        return generate_content_with_retry(prompt)

    except Exception as e:
        print(f"Error generating email content: {str(e)}")
        return ""

def send_campaign_emails(state: CampaignState) -> CampaignState:
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("EMAIL_APP_PASSWORD")

    if not all([sender_email, sender_password]):
        state.email_status = "Email sending failed: Missing email credentials"
        return state

    try:
        # Load customer data
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            customer_data_path = os.path.join(script_dir, 'data', 'customers_with_emails.csv')
            customer_data = pd.read_csv(customer_data_path).head(6)
        except FileNotFoundError:
            try:
                customer_data_path = os.path.join(script_dir, 'data', 'customer.csv')
                customer_data = pd.read_csv(customer_data_path).head(6)
            except FileNotFoundError:
                customer_data_path = os.path.join(script_dir, 'data', 'customers.csv')
                customer_data = pd.read_csv(customer_data_path).head(6)

        # Filter out rows with missing emails
        customer_data = customer_data[customer_data['email'].notna()]
        if customer_data.empty:
            state.email_status = "No customer emails found"
            return state

        # Connect to SMTP and send emails
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            sent_count = 0
            failed_count = 0

            for _, customer in customer_data.iterrows():
                try:
                    email_content = generate_personalized_email(state, customer.to_dict())
                    if not email_content:
                        failed_count += 1
                        continue

                    if "SUBJECT:" in email_content:
                        subject, body = email_content.split("SUBJECT:", 1)[1].split("\n\n", 1)
                        subject = subject.strip()
                        body = body.strip()
                    else:
                        subject = f"Special Offer Just for You!"
                        body = email_content

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
