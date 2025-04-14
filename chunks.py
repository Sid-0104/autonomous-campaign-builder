import os
import pandas as pd
from langchain.schema import Document
import google.generativeai as genai
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import shutil


load_dotenv()

# === Google Gemini Wrapper ===

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key=None, model_name="models/embedding-001"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        genai.configure(api_key=self.api_key)

    def embed_documents(self, texts):
        return [
            genai.embed_content(model=self.model_name, content=text, task_type="retrieval_document")["embedding"]
            for text in texts
        ]

    def embed_query(self, text):
        return genai.embed_content(model=self.model_name, content=text, task_type="retrieval_query")["embedding"]


# === Config ===
USE_OPENAI = True # Set this to True if you want to switch to OpenAI
CHROMA_DB_ROOT_DIR = "chroma_openai_store_embeddings" if USE_OPENAI else "chroma_gemini_store_embeddings"

# Embedding model settings
OPENAI_EMBED_MODEL = "text-embedding-3-small"

from langchain_community.embeddings import OpenAIEmbeddings

def get_embedding_model():
    if USE_OPENAI:
        print(" Using OpenAI Embeddings")
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    else:
        print(" Using Google Gemini Embeddings")
        return GeminiEmbeddings()

# === Load CSVs in a given folder and treat each row as one chunk ===
def load_rowwise_documents_from_folder(folder_path: str, category_tag: str):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            for i, row in df.iterrows():
                text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
                metadata = {
                    "source": filename,
                    "row": i,
                    "category": category_tag
                }
                all_docs.append(Document(page_content=text, metadata=metadata))
    return all_docs

# === Embedding + ChromaDB storing ===
def embed_and_store(docs, persist_dir: str, collection_name: str, embedding_model, reset=True):
    if reset:
        shutil.rmtree(persist_dir, ignore_errors=True)

    # Check if the DB already exists (if not, create with from_documents)
    db_exists = os.path.exists(persist_dir)

    if not db_exists:
        # First time creating the collection
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        print(f"✅ Initialized new collection: {collection_name} with {len(docs)} documents")
        return

    # Reconnect to existing collection to check for duplicates
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name
    )

    # Load existing metadata
    existing = vectordb.get(include=["metadatas"])
    existing_keys = set()
    if existing and "metadatas" in existing:
        for meta in existing["metadatas"]:
            key = f"{meta.get('source')}_{meta.get('row')}"
            existing_keys.add(key)

    # Filter only new docs
    new_docs = []
    for doc in docs:
        key = f"{doc.metadata.get('source')}_{doc.metadata.get('row')}"
        if key not in existing_keys:
            new_docs.append(doc)

    if new_docs:
        vectordb.add_documents(new_docs)
        print(f"✅ Added {len(new_docs)} new docs to {collection_name}")
    else:
        print(f"⚠️ No new documents found for {collection_name}")


# === Main pipeline for all subfolders ===
def process_all_subfolders(root_data_folder: str):
    embedding_model = get_embedding_model()

    for subfolder in os.listdir(root_data_folder):
        subfolder_path = os.path.join(root_data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            collection_name = subfolder.lower().replace("&", "_and_").replace(" ", "_")
            print(f" Processing subfolder: {subfolder} → Collection: {collection_name}")

            docs = load_rowwise_documents_from_folder(subfolder_path, category_tag=subfolder)
            if docs:
                persist_path = os.path.join(CHROMA_DB_ROOT_DIR, collection_name)
                embed_and_store(docs, persist_path, collection_name, embedding_model)
            else:
                print(f" No CSVs found in {subfolder_path}")

# === Search Chroma Collection ===
def query_chroma_collection(query: str, collection_name: str, top_k: int = 5):
    persist_path = os.path.join(CHROMA_DB_ROOT_DIR, collection_name)
    embedding_model = get_embedding_model()

    vectordb = Chroma(
        persist_directory=persist_path,
        embedding_function=embedding_model,
        collection_name=collection_name
    )

    results = vectordb.similarity_search(query, k=top_k)
    print(f"\n🔍 Top {top_k} results for: '{query}' [Collection: {collection_name}]\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}\n")

# === Run Everything ===
if __name__ == "__main__":
    process_all_subfolders(r"./datas")
    query_chroma_collection("SUV sales of hyundai where city = Chennai only ", "automobiles", top_k=6)
