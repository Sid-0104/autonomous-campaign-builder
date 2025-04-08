import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Get API key from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/text-embedding-004")

def load_mock_data():
    base_path = 'c:\\Users\\user\\Desktop\\campaign builder'
    
    try:
        # Load sales data from CSV
        sales_data = pd.read_csv(f'{base_path}\\data\\sales.csv')
        
        # Load campaign data from CSV
        campaign_data = pd.read_csv(f'{base_path}\\data\\marketing_campaign.csv')
        
        # Try to load customer segments with different possible filenames
        try:
            customer_segments = pd.read_csv(f'{base_path}\\data\\customer_with_emails.csv')
        except FileNotFoundError:
            try:
                customer_segments = pd.read_csv(f'{base_path}\\data\\customer.csv')
            except FileNotFoundError:
                # Final fallback
                customer_segments = pd.read_csv(f'{base_path}\\l\\customer_segments.csv')
        
        # Convert DataFrames to list of dictionaries for compatibility with existing code
        campaign_data_list = campaign_data.to_dict('records')
        customer_segments_list = customer_segments.to_dict('records')
        
        return sales_data, campaign_data_list, customer_segments_list
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Attempting to load from alternative location...")
        
        try:
            # Fallback to the 'l' directory if 'data' doesn't exist
            sales_data = pd.read_csv(f'{base_path}\\l\\sales_data.csv')
            campaign_data = pd.read_csv(f'{base_path}\\l\\campaign_data.csv')
            customer_segments = pd.read_csv(f'{base_path}\\l\\customer_segments.csv')
            
            # Convert DataFrames to list of dictionaries
            campaign_data_list = campaign_data.to_dict('records')
            customer_segments_list = customer_segments.to_dict('records')
            
            return sales_data, campaign_data_list, customer_segments_list
        except Exception as e:
            print(f"Failed to load data from alternative location: {e}")
            # Return empty data as a last resort to prevent complete failure
            return pd.DataFrame(), [], []

def initialize_vector_db(campaign_data, customer_segments):
    # Ensure API key is available
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Create embeddings with explicit API key
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY,
        model=EMBEDDING_MODEL,
        task_type="retrieval_query"  # Specify task type explicitly
    )
    
    # Create documents for campaigns
    campaign_docs = []
    for campaign in campaign_data:
        # Safely extract fields with proper error handling
        try:
            # Handle target models - could be in different formats
            target_models = campaign.get('target_models', '')
            if isinstance(target_models, str):
                try:
                    # Try to safely evaluate string representations of lists
                    if target_models.startswith('['):
                        target_models = eval(target_models)
                    else:
                        target_models = [target_models]
                except:
                    target_models = [target_models]
            elif not isinstance(target_models, list):
                target_models = [str(target_models)]
                
            # Handle channels similarly
            channels = campaign.get('channels', '')
            if isinstance(channels, str):
                try:
                    if channels.startswith('['):
                        channels = eval(channels)
                    else:
                        channels = [channels]
                except:
                    channels = [channels]
            elif not isinstance(channels, list):
                channels = [str(channels)]
            
            # Handle campaign metrics - adapt to your actual schema
            # Check for both dot notation and direct column access
            roi = campaign.get('results.roi', campaign.get('roi', ''))
            sales = campaign.get('results.sales', campaign.get('sales', ''))
            impressions = campaign.get('results.impressions', campaign.get('impressions', ''))
            
            # Build content with available fields
            content = f"Campaign: {campaign.get('name', campaign.get('campaign_name', ''))}\n"
            content += f"Region: {campaign.get('region', '')}\n"
            
            # Handle date fields which might be in different formats
            start_date = campaign.get('start_date', '')
            end_date = campaign.get('end_date', '')
            content += f"Period: {start_date} to {end_date}\n"
            
            # Add budget if available
            if 'budget' in campaign:
                content += f"Budget: {campaign.get('budget', '')}\n"
                
            content += f"Target models: {', '.join(target_models)}\n"
            content += f"Channels: {', '.join(channels)}\n"
            content += f"Messaging: {campaign.get('messaging', campaign.get('message', ''))}\n"
            
            # Add all available metrics
            content += f"Results: "
            if roi: content += f"ROI {roi}, "
            if sales: content += f"Sales {sales}, "
            if impressions: content += f"Impressions {impressions}"
            
            # Create metadata with key fields for retrieval
            metadata = {
                "type": "campaign",
                "name": campaign.get("name", campaign.get("campaign_name", "")),
                "region": campaign.get("region", ""),
                "target_models": target_models
            }
            
            campaign_docs.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"Error processing campaign: {e}")
            continue
    
    # Create documents for customer segments
    segment_docs = []
    for segment in customer_segments:
        try:
            # Check for both flattened and nested structures
            # For demographics
            age_range = segment.get('demographics.age_range', segment.get('age_range', ''))
            income = segment.get('demographics.income', segment.get('income', ''))
            education = segment.get('demographics.education', segment.get('education', ''))
            
            # For preferences
            preferred_models = segment.get('preferences.models', segment.get('preferred_models', ''))
            if isinstance(preferred_models, str):
                try:
                    if preferred_models.startswith('['):
                        preferred_models = eval(preferred_models)
                    else:
                        preferred_models = [preferred_models]
                except:
                    preferred_models = [preferred_models]
            elif not isinstance(preferred_models, list):
                preferred_models = [str(preferred_models)]
                
            # Features
            features = segment.get('preferences.features', segment.get('features', ''))
            if isinstance(features, str):
                try:
                    if features.startswith('['):
                        features = eval(features)
                    else:
                        features = [features]
                except:
                    features = [features]
            elif not isinstance(features, list):
                features = [str(features)]
                
            # Channels
            channels = segment.get('preferences.channels', segment.get('preferred_channels', ''))
            if isinstance(channels, str):
                try:
                    if channels.startswith('['):
                        channels = eval(channels)
                    else:
                        channels = [channels]
                except:
                    channels = [channels]
            elif not isinstance(channels, list):
                channels = [str(channels)]
            
            # Check for email field which might be in your customer data
            email = segment.get('email', '')
            
            # Build content string
            content = f"Segment: {segment.get('name', segment.get('segment_name', ''))}\n"
            content += f"Demographics: "
            if age_range: content += f"Age {age_range}, "
            if income: content += f"Income {income}, "
            if education: content += f"Education {education}\n"
            else: content += "\n"
            
            if preferred_models:
                content += f"Preferred models: {', '.join(preferred_models)}\n"
            if features:
                content += f"Values features: {', '.join(features)}\n"
            if channels:
                content += f"Preferred channels: {', '.join(channels)}"
            if email:
                content += f"\nContact: {email}"
            
            metadata = {
                "type": "segment",
                "name": segment.get("name", segment.get("segment_name", "")),
                "preferred_models": preferred_models if preferred_models else []
            }
            
            segment_docs.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"Error processing segment: {e}")
            continue
    
    # Combine all documents
    all_docs = campaign_docs + segment_docs
    
    # Create the vector store
    vector_store = FAISS.from_documents(all_docs, embeddings)
    
    return vector_store