##################################  Original code ######################################
import streamlit as st
import os
import re
import pickle
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader

################## Changes ########################
from langchain_community.vectorstores import FAISS

from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
###################################################

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from user_interface.chat_ui import extract_table_images_from_pdf

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

# === Function to handle PDF upload ===
def handle_pdf_upload(uploaded_file):
    pdf_name = uploaded_file.name
    base_name = re.sub(r'[^\w\-]', '_', os.path.splitext(pdf_name)[0])

    storage_dir = os.path.join("stored_pdfs", base_name)
    images_path = os.path.join(storage_dir, "images.pkl")
    tables_path = os.path.join(storage_dir, "tables.pkl")
    pdf_path = os.path.join(storage_dir, "original.pdf")
    extracted_images_dir = os.path.join(storage_dir, "images")
    extracted_tables_dir = os.path.join(storage_dir, "tables")
    collection_name = base_name.lower()

    os.makedirs(storage_dir, exist_ok=True)

    # Use same embedding model in all scripts
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(url="http://localhost:6333")  # Docker Qdrant

    # Check if collection already exists
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in existing_collections:
        # ✅ Load cached data from Qdrant
        st.session_state.vectors = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=st.session_state.embeddings,
        )
        with open(images_path, "rb") as f:
            st.session_state.image_map = pickle.load(f)
        with open(tables_path, "rb") as f:
            st.session_state.table_image_map = pickle.load(f)
        st.success(f"✅ Loaded cached data for: {pdf_name}")
        return

    # 🆕 Process and upload new PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["page"] = doc.metadata.get("page", 0)

    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
    documents = splitter.split_documents(docs)
    for i, doc in enumerate(documents):
        doc.metadata["chunk_index"] = i


    vector_size = len(st.session_state.embeddings.embed_query("test"))

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Upload vectors
    st.session_state.vectors = Qdrant.from_documents(
        documents,
        st.session_state.embeddings,
        url="http://localhost:6333",
        collection_name=collection_name
    )

    # Extract and save images
    st.session_state.image_map = extract_images_from_pdf(pdf_path, extracted_images_dir)
    with open(images_path, "wb") as f:
        pickle.dump(st.session_state.image_map, f)

    # Extract and save tables
    st.session_state.table_image_map = extract_table_images_from_pdf(pdf_path, extracted_tables_dir)
    with open(tables_path, "wb") as f:
        pickle.dump(st.session_state.table_image_map, f)

    st.success(f"✅ Processed and cached new PDF: {pdf_name}")



























    


######################################## This is Gdrive code #########################################
# #pdf_upload.py
# import streamlit as st
# import os, re, pickle, fitz
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from user_interface.chat_ui import extract_table_images_from_pdf
# from gdrive.gdrive_uploader import upload_pdf_data_to_gdrive

# def extract_images_from_pdf(pdf_path, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     doc = fitz.open(pdf_path)
#     image_map = {}
#     for page_index in range(len(doc)):
#         page = doc[page_index]
#         for img_index, img in enumerate(page.get_images(full=True)):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_bytes = base_image["image"]
#             ext = base_image["ext"]
#             image_path = os.path.join(output_folder, f"page{page_index+1}_img{img_index+1}.{ext}")
#             with open(image_path, "wb") as f:
#                 f.write(image_bytes)
#             image_map.setdefault(page_index + 1, []).append(image_path)
#     return image_map

# def handle_pdf_upload(uploaded_file):
#     pdf_name = uploaded_file.name
#     base_name = re.sub(r'[^\w\-]', '_', os.path.splitext(pdf_name)[0])
#     st.session_state.embeddings = HuggingFaceEmbeddings()

#     temp_dir = os.path.join(os.getcwd(), "temp")
#     os.makedirs(temp_dir, exist_ok=True)

#     temp_pdf_path = os.path.join(temp_dir, f"{base_name}.pdf")
#     temp_vector_path = os.path.join(temp_dir, f"{base_name}_vectors")
#     extracted_images_dir = os.path.join(temp_dir, f"{base_name}_images")
#     extracted_tables_dir = os.path.join(temp_dir, f"{base_name}_tables")

#     with open(temp_pdf_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     loader = PyPDFLoader(temp_pdf_path)
#     docs = loader.load()
#     for doc in docs:
#         doc.metadata["page"] = doc.metadata.get("page", 0)

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = splitter.split_documents(docs)
#     for i, doc in enumerate(documents):
#         doc.metadata["chunk_index"] = i

#     # st.session_state.documents = documents
#     # st.session_state.vectors = FAISS.from_documents(documents, st.session_state.embeddings)
#     # st.session_state.vectors.save_local(temp_vector_path)

#     st.session_state.documents = documents
#     vectorstore = FAISS.from_documents(documents, st.session_state.embeddings)
#     st.session_state.vectors = vectorstore

#     # ✅ Save vector store as FAISS index locally
#     vectorstore.save_local(temp_vector_path)

#     # ✅ Also save as .pkl (this is what will be uploaded & reloaded)
#     faiss_pickle_path = os.path.join(temp_vector_path, "faiss_index.pkl")
#     with open(faiss_pickle_path, "wb") as f:
#         pickle.dump(vectorstore, f)

#     image_map = extract_images_from_pdf(temp_pdf_path, extracted_images_dir)
#     table_map = extract_table_images_from_pdf(temp_pdf_path, extracted_tables_dir)

#     uploaded_files = upload_pdf_data_to_gdrive(
#         base_name,
#         temp_pdf_path,
#         temp_vector_path,
#         image_map,
#         table_map,
#         extracted_images_dir,
#         extracted_tables_dir
#     )

#     st.session_state.image_map = image_map
#     st.session_state.table_image_map = table_map
#     st.success(f"✅ Uploaded and processed {pdf_name} to Google Drive.")

#     st.markdown("### Files uploaded to Google Drive:")
#     for name in uploaded_files:
#         st.markdown(f"- {name}")

##################################################################################################


