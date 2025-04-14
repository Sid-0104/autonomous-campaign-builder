import hashlib
import time
import pandas as pd
import chromadb
from chromadb.config import Settings

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma


def load_mock_data():
    sales_data = pd.read_csv("data/sales.csv")
    campaign_data = pd.read_csv("data/marketing_campaign.csv")
    customer_segments = pd.read_csv("data/customer.csv")
    return sales_data, campaign_data, customer_segments


def get_text_column(df, preferred=["description", "objective", "summary", "text", "content", "message"]):
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    raise ValueError("No suitable text column found in the dataframe")


def batch_embed(embedding, docs, batch_size=10, sleep_sec=0.3):
    """
    Embed documents in batches to avoid API rate limits.
    """
    embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        try:
            batch_embeddings = embedding.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"❌ Batch {i // batch_size + 1} failed: {e}")
            embeddings.extend([[0.0] * 768 for _ in batch])
        time.sleep(sleep_sec)
    return embeddings


def initialize_vector_db(campaign_data, customer_segments):
    # Initialize embedding model
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create Chroma client (in-memory)
    chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))

    chroma_client.reset()

    # Create or reset the collection
    collection_name = "campaign_knowledge_base"
    if collection_name in [col.name for col in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=collection_name)
    collection = chroma_client.create_collection(name=collection_name)

    # Pick text columns
    campaign_text_col = get_text_column(campaign_data)
    customer_text_col = get_text_column(customer_segments)

    print(f"✅ Using campaign column: {campaign_text_col}")
    print(f"✅ Using customer column: {customer_text_col}")

    documents = []
    metadatas = []

    for _, row in campaign_data.iterrows():
        text = str(row[campaign_text_col])
        documents.append(text)
        metadatas.append({"type": "campaign", "id": str(row.get("id", _))})

    for _, row in customer_segments.iterrows():
        text = str(row[customer_text_col])
        documents.append(text)
        metadatas.append({"type": "customer_segment", "id": str(row.get("id", _))})

    # Embed the documents
    embeddings = batch_embed(
        embedding,
        documents,
        batch_size=10,
        sleep_sec=0.3
    )

    ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in documents]

    # Add documents and embeddings to Chroma
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    # ✅ Wrap with LangChain's Chroma to enable .similarity_search()
    langchain_chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        client=chroma_client
    )

    return langchain_chroma
