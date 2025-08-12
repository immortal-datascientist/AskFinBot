
import streamlit as st
import os
import re
import pickle
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from user_interface.chat_ui import extract_table_images_from_pdf

# === Extract images ===
def extract_images_from_pdf(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_map = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = os.path.join(output_folder, f"page{page_index+1}_img{img_index+1}.{image_ext}")
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_map.setdefault(page_index + 1, []).append(image_path)

    return image_map

# === Handle PDF upload or reuse cached ===
def handle_pdf_upload(uploaded_file):
    pdf_name = uploaded_file.name
    base_name = re.sub(r'[^\w\-]', '_', os.path.splitext(pdf_name)[0])

    storage_dir = os.path.join("stored_pdfs", base_name)
    vectors_path = os.path.join(storage_dir, "vectors")
    images_path = os.path.join(storage_dir, "images.pkl")
    tables_path = os.path.join(storage_dir, "tables.pkl")
    pdf_path = os.path.join(storage_dir, "original.pdf")
    extracted_images_dir = os.path.join(storage_dir, "images")
    extracted_tables_dir = os.path.join(storage_dir, "tables")

    os.makedirs(storage_dir, exist_ok=True)

    if os.path.exists(vectors_path):
        # ✅ Load cached data
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.vectors = FAISS.load_local(
            vectors_path,
            st.session_state.embeddings,
            allow_dangerous_deserialization=True
        )

        with open(images_path, "rb") as f:
            st.session_state.image_map = pickle.load(f)

        with open(tables_path, "rb") as f:
            st.session_state.table_image_map = pickle.load(f)

        st.success(f"✅ Loaded cached data for: {pdf_name}")
        return

    # 🆕 New upload: process
    st.session_state.embeddings = HuggingFaceEmbeddings()
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["page"] = doc.metadata.get("page", 0)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    for i, doc in enumerate(documents):
        doc.metadata["chunk_index"] = i

    # Create FAISS vector index
    st.session_state.documents = documents
    st.session_state.vectors = FAISS.from_documents(documents, st.session_state.embeddings)
    st.session_state.vectors.save_local(vectors_path)

    # Extract and save images
    st.session_state.image_map = extract_images_from_pdf(pdf_path, extracted_images_dir)
    with open(images_path, "wb") as f:
        pickle.dump(st.session_state.image_map, f)

    # Extract and save tables
    st.session_state.table_image_map = extract_table_images_from_pdf(pdf_path, extracted_tables_dir)
    with open(tables_path, "wb") as f:
        pickle.dump(st.session_state.table_image_map, f)

    st.success(f"✅ Processed and cached new PDF: {pdf_name}")



