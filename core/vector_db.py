import os
import pandas as pd
# Use Chroma instead of FAISS
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv
import time

# Load environment variables from .env file, searching parent directories if needed
load_dotenv(find_dotenv(usecwd=True))

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def load_documents_by_domain(domain="automotives"):
    """
    Load all CSV files for a specific domain and convert each row to a document.
    Returns both the raw dataframes and the chunked documents.
    """
    # Get the base directory dynamically
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, 'data')
    
    # Define file patterns for each domain
    domain_prefix = domain.lower()
    
    # Initialize empty dataframes and document list
    sales_data = pd.DataFrame()
    campaign_data = pd.DataFrame()
    customer_data = pd.DataFrame()
    all_docs = []
    
    # Process sales data
    sales_filename = f"{domain_prefix}_sales.csv"
    sales_path = os.path.join(data_dir, sales_filename)
    if not os.path.exists(sales_path):
        sales_path = os.path.join(data_dir, "sales.csv")
    
    if os.path.exists(sales_path):
        try:
            sales_data = pd.read_csv(sales_path)
            print(f"Loaded sales data: {len(sales_data)} rows from {sales_path}")
            
            # Create documents from sales data
            for i, row in sales_data.iterrows():
                text = ", ".join(f"{col}: {row[col]}" for col in sales_data.columns)
                metadata = {
                    "source": os.path.basename(sales_path),
                    "row": i,
                    "category": "sales",
                    "domain": domain_prefix
                }
                all_docs.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            print(f"Error loading sales data: {e}")
    
    # Process campaign data
    campaign_filename = f"{domain_prefix}_marketing_campaign.csv"
    campaign_path = os.path.join(data_dir, campaign_filename)
    if not os.path.exists(campaign_path):
        campaign_path = os.path.join(data_dir, "marketing_campaign.csv")
    
    if os.path.exists(campaign_path):
        try:
            campaign_data = pd.read_csv(campaign_path)
            print(f"Loaded campaign data: {len(campaign_data)} rows from {campaign_path}")
            
            # Create documents from campaign data
            for i, row in campaign_data.iterrows():
                text = ", ".join(f"{col}: {row[col]}" for col in campaign_data.columns)
                metadata = {
                    "source": os.path.basename(campaign_path),
                    "row": i,
                    "category": "campaign",
                    "domain": domain_prefix
                }
                all_docs.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            print(f"Error loading campaign data: {e}")
    
    # Process customer data
    customer_filename = f"{domain_prefix}_customer.csv"
    customer_path = os.path.join(data_dir, customer_filename)
    if not os.path.exists(customer_path):
        customer_path = os.path.join(data_dir, "customer.csv")
    
    if os.path.exists(customer_path):
        try:
            customer_data = pd.read_csv(customer_path)
            print(f"Loaded customer data: {len(customer_data)} rows from {customer_path}")
            
            # Create documents from customer data
            for i, row in customer_data.iterrows():
                text = ", ".join(f"{col}: {row[col]}" for col in customer_data.columns)
                metadata = {
                    "source": os.path.basename(customer_path),
                    "row": i,
                    "category": "customer",
                    "domain": domain_prefix
                }
                all_docs.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            print(f"Error loading customer data: {e}")
    
    print(f"Created {len(all_docs)} documents from {domain} domain data")
    
    # Convert campaign and customer data to list of dictionaries
    campaign_data_list = campaign_data.to_dict('records') if not campaign_data.empty else []
    customer_data_list = customer_data.to_dict('records') if not customer_data.empty else []
    
    return sales_data, campaign_data_list, customer_data_list, all_docs

def get_embeddings_model(model_provider="gemini"):
    """Get the appropriate embeddings model based on the selected LLM provider"""
    try:
        if model_provider.lower() == "openai":
            if not OPENAI_API_KEY:
                print("Warning: OPENAI_API_KEY not set. Falling back to Gemini.")
                return get_embeddings_model("gemini")
                
            embeddings = OpenAIEmbeddings(
                api_key=OPENAI_API_KEY,
                model=OPENAI_EMBEDDING_MODEL
            )
            print(f"Using OpenAI embeddings: {OPENAI_EMBEDDING_MODEL}")
        else:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
                
            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY,
                model=EMBEDDING_MODEL,
                task_type="retrieval_query"
            )
            print(f"Using Gemini embeddings: {EMBEDDING_MODEL}")
            
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        print("Falling back to HuggingFace embeddings")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings()

def load_rowwise_documents_from_folder(folder_path: str, category_tag: str, domain_filter=None):
    all_docs = []
    
    # Handle domain filter variations
    domain_filters = [domain_filter] if domain_filter else []
    if domain_filter == "automotives":
        domain_filters.append("automotives")
    elif domain_filter == "automotives":
        domain_filters.append("automotives")
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Apply domain filter if specified
            if domain_filters:
                # Check if any of the domain filters match
                if not any(filename.startswith(filter_name.lower()) for filter_name in domain_filters) and not filename == "sales.csv":
                    continue
                    
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                print(f"Processing {filename} with {len(df)} rows")

                # Determine category based on filename
                if "sales" in filename.lower():
                    category = "sales"
                elif "campaign" in filename.lower() or "marketing" in filename.lower():
                    category = "campaign"
                elif "customer" in filename.lower():
                    category = "customer"
                else:
                    category = category_tag
                
                # Extract domain from filename if possible
                domain = filename.split("_")[0] if "_" in filename else "general"

                for i, row in df.iterrows():
                    text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
                    metadata = {
                        "source": filename,
                        "row": i,
                        "category": category,
                        "domain": domain
                    }
                    all_docs.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Loaded {len(all_docs)} document chunks from {folder_path}")
    return all_docs

def initialize_vector_db(campaign_data=None, customer_segments=None, domain="automotives", model_provider="gemini"):
    
    # Get embeddings model (OpenAI or Gemini) through your existing function
    embeddings = get_embeddings_model(model_provider)
    
    # Add validation to ensure embeddings are properly initialized
    if embeddings is None:
        raise ValueError("Failed to initialize embeddings. Please check your API keys and settings.")

    # Ensure API key is available
    if model_provider.lower() == "openai" and not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY environment variable is not set. Falling back to Gemini.")
        model_provider = "gemini"
        
    if model_provider.lower() == "gemini" and not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Get the base directory dynamically
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Debug information about input data
    print(f"Processing data for domain: {domain} using {model_provider} embeddings")
    
    # Load documents as you're currently doing
    _, campaign_data_loaded, customer_segments_loaded, all_docs = load_documents_by_domain(domain)
    
    # Use provided data or loaded data
    campaign_data = campaign_data if campaign_data is not None else campaign_data_loaded
    customer_segments = customer_segments if customer_segments is not None else customer_segments_loaded
    
    # If no documents were found, create dummy documents
    if len(all_docs) == 0:
        print("No documents found. Creating dummy documents for initialization.")
        dummy_docs = [
            Document(
                page_content=f"Dummy {category} document for {domain}",
                metadata={"source": "dummy", "category": category, "domain": domain}
            )
            for category in ["sales", "campaign", "customer"]
        ]
        all_docs.extend(dummy_docs)
    
    print(f"Using {len(all_docs)} documents for vector database")
    
    # Use Chroma instead of FAISS
    try:
        print("Creating Chroma vector store...")
        
        # Define persistent directory for Chroma
        chroma_path = os.path.join(script_dir, f"chroma_db_{domain}_{model_provider}")
        os.makedirs(chroma_path, exist_ok=True)
        
        # Process documents in smaller batches to avoid memory issues
        batch_size = 50
        
        # Start with first batch
        first_batch = all_docs[:min(batch_size, len(all_docs))]
        print(f"Initializing with first batch of {len(first_batch)} documents...")
        
        # Create Chroma vector store with first batch - add collection_metadata to specify embedding dimension
        embedding_dimension = 1536 if model_provider.lower() == "openai" else 768
        print(f"Using embedding dimension: {embedding_dimension} for {model_provider}")
        
        vector_store = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            persist_directory=chroma_path,
            collection_metadata={"hnsw:space": "cosine", "dimension": embedding_dimension}
        )
        
        # Make sure to persist after first batch
        vector_store.persist()
        
        # Add remaining documents in batches
        if len(all_docs) > batch_size:
            remaining_docs = all_docs[batch_size:]
            total_batches = (len(remaining_docs) + batch_size - 1) // batch_size
            
            for i in range(0, len(remaining_docs), batch_size):
                batch = remaining_docs[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
                
                try:
                    # Add current batch to existing store
                    vector_store.add_documents(documents=batch)
                    vector_store.persist()  # Save after each batch
                    print(f"Batch {batch_num}/{total_batches} completed and persisted")
                except Exception as batch_error:
                    print(f"Error processing batch {batch_num}: {batch_error}")
                    # Try one by one if batch fails
                    for j, doc in enumerate(batch):
                        try:
                            vector_store.add_documents(documents=[doc])
                            vector_store.persist()
                        except Exception as doc_error:
                            print(f"Error with document {j} in batch {batch_num}: {str(doc_error)[:100]}...")
                            continue
        
        print(f"Successfully created Chroma vector store and saved to {chroma_path}")
        
        return vector_store
        
    except Exception as chroma_error:
        print(f"Error creating Chroma store: {chroma_error}")
        
        # Create a simple in-memory store as last resort
        print("Creating simple in-memory store as fallback")
        from langchain_community.vectorstores import DocArrayInMemorySearch
        
        try:
            simple_store = DocArrayInMemorySearch.from_documents(
                all_docs[:100],  # Use first 100 docs
                embeddings
            )
            return simple_store
        except Exception as e:
            print(f"Failed to create any vector store: {e}")
            return None