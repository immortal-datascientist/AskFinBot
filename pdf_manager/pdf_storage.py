# pdf_storage.py

import os
import pickle
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


PDF_DATA_DIR = "stored_pdfs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Keep consistent for upload + retrieval

def list_saved_pdfs():
    os.makedirs(PDF_DATA_DIR, exist_ok=True)
    return [name for name in os.listdir(PDF_DATA_DIR) if os.path.isdir(os.path.join(PDF_DATA_DIR, name))]

def get_pdf_data_path(pdf_name):
    return os.path.join(PDF_DATA_DIR, pdf_name)

def load_vectors_qdrant(collection_name):
    """
    Load Qdrant vector store.
    If collection does not exist, raise an error instead of creating it.
    Warn if stored collection vector size != embedding model vector size.
    """

    collection_name = collection_name.lower()
    print(f"[INFO] Using Qdrant collection: {collection_name}")
    
    client = QdrantClient(url="http://localhost:6333")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check embedding size from model
    embed_dim = len(embeddings.embed_query("test"))

    # Get all existing collections
    collections = [c.name for c in client.get_collections().collections]

    if collection_name not in collections:
        raise ValueError(
            f"❌ Collection '{collection_name}' not found in Qdrant.\n"
            f"Upload the PDF first to create and populate this collection."
        )

    # Get existing collection info
    info = client.get_collection(collection_name)
    stored_dim = None

    if hasattr(info, "config") and hasattr(info.config, "params"):
        try:
            stored_dim = info.config.params.vectors.size
        except Exception:
            stored_dim = None

    print(f"[INFO] Collection '{collection_name}' loaded with dim={stored_dim}, model dim={embed_dim}")

    if stored_dim is not None and stored_dim != embed_dim:
        raise ValueError(
            f"⚠ Dimension mismatch! Qdrant has dim={stored_dim} but embedding model produces dim={embed_dim}.\n"
            "Retrieval will fail unless you re-ingest with the same embedding model."
        )

    # Check how many vectors are in the collection
    count_info = client.count(collection_name=collection_name, exact=True)
    print(f"[INFO] Number of vectors in collection: {count_info.count}")
    if count_info.count == 0:
        raise ValueError(
            f"⚠ Collection '{collection_name}' exists but contains 0 vectors.\n"
            f"Upload and process the PDF before querying."
        )

    # Return vector store
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    return qdrant



def load_image_map(pdf_name):
    path = os.path.join(get_pdf_data_path(pdf_name), "images.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def load_table_map(pdf_name):
    path = os.path.join(get_pdf_data_path(pdf_name), "tables.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}
