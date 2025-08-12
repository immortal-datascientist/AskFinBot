import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# FIXED: Match with pdf_upload.py
PDF_DATA_DIR = "stored_pdfs"

def list_saved_pdfs():
    if not os.path.exists(PDF_DATA_DIR):
        os.makedirs(PDF_DATA_DIR)
    return [name for name in os.listdir(PDF_DATA_DIR) if os.path.isdir(os.path.join(PDF_DATA_DIR, name))]

def get_pdf_data_path(pdf_name):
    return os.path.join(PDF_DATA_DIR, pdf_name)

def load_vectors(pdf_name):
    path = os.path.join(get_pdf_data_path(pdf_name), "vectors")
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)  # include this

def load_image_map(pdf_name):
    path = os.path.join(get_pdf_data_path(pdf_name), "images.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_table_map(pdf_name):
    path = os.path.join(get_pdf_data_path(pdf_name), "tables.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
