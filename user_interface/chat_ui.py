
# chat_ui.py
import streamlit as st 
from langchain.chains import create_retrieval_chain
from models.llm_models import document_chain
from ocr_extractor.ocr_image_matcher import find_relevant_images_with_keywords
from langchain_community.tools import DuckDuckGoSearchRun
import os
import hashlib
from PIL import Image
import pdfplumber

search_tool = DuckDuckGoSearchRun()

def is_irrelevant(answer_text):
    """Detect irrelevant answers based on common phrases or low length."""
    if not answer_text or len(answer_text.strip().split()) < 10:
        return True
    
    answer_lower = answer_text.lower()
    irrelevant_keywords = [
        "not related to the provided context",
        "i cannot provide",
        "does not mention",
        "no relevant information",
        "unable to answer",
        "context is about",
        "outside the scope",
        "not covered",
        "information is missing"
    ]
    return any(kw in answer_lower for kw in irrelevant_keywords)

def get_chat_response():
    input_text = st.session_state.input_text
    if not input_text:
        return

    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": input_text})

    matched_chunks = response.get("context", [])
    matched_pages = [doc.metadata["page"] + 1 for doc in matched_chunks]
    answer_text = response.get("answer", "").strip()

    # 🕵️ Check if answer is irrelevant or vague
    if is_irrelevant(answer_text):
        fallback = search_tool.run(input_text)
        answer_text += f"\n\n🌐 *Extra info from the web:* {fallback}"

    # 🔍 Get relevant images using OCR
    matched_images = find_relevant_images_with_keywords(
        st.session_state.image_map,
        input_text,
        matched_pages,
        top_k=5
    )

    # 📊 Get table images for those pages
    table_images = []
    if "table_image_map" in st.session_state:
        for page in matched_pages:
            if page in st.session_state.table_image_map:
                table_images.extend(st.session_state.table_image_map[page])

    # 💾 Save full response
    st.session_state.chat_history.append({
        "question": input_text,
        "answer": answer_text,
        "matched_chunks": matched_chunks,
        "matched_pages": matched_pages,
        "matched_images": matched_images,
        "matched_tables": table_images
    })

    # 🔄 Reset input
    st.session_state.input_text = ""


def hash_image(img_path):
    with open(img_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    

def extract_table_images_from_pdf(pdf_path, output_folder="extracted_tables"):
    os.makedirs(output_folder, exist_ok=True)
    table_images = {}  # {page_number: [image_paths]}

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.find_tables()
            for j, table in enumerate(tables):
                bbox = table.bbox  # (x0, top, x1, bottom)
                cropped = page.within_bbox(bbox).to_image(resolution=200)
                path = os.path.join(output_folder, f"page{i+1}_table{j+1}.png")
                cropped.save(path, format="PNG")
                table_images.setdefault(i + 1, []).append(path)

    return table_images


def render_home():
    st.title("Welcome to")
    st.image("images/immortal_logo.png", width=700)
    st.title("FINACLE PDFs QUERIES")

    st.markdown("""
        <style>
        .scrollable-history {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #2E2E2E;
            color: white;
        }
        .question {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .answer {
            margin-bottom: 10px;
        }
        .stButton button {
            background-color: red;
            color: white;
            width: 100%;
            height: 50px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.chat_history:
        with st.container():
            st.markdown("<div class='scrollable-history'>", unsafe_allow_html=True)

            for chat in st.session_state.chat_history:
                st.markdown(f"<div class='question'>❓ Question:</div> {chat['question']}", unsafe_allow_html=True)
                st.markdown(f"<div class='answer'>✅ Answer:<br>{chat['answer']}</div>", unsafe_allow_html=True)

                relevant_images = chat.get("matched_images", [])
                table_imgs = chat.get("matched_tables", [])

                if table_imgs:
                    st.markdown("**📊 Detected Table(s):**")
                    for img_path in table_imgs:
                        st.image(img_path, use_column_width=True)

                if relevant_images:
                    st.markdown("**🖼️ Relevant Screenshot(s):**")
                    for img_path, score, page in relevant_images:
                        st.image(img_path, use_column_width=True)

                if not relevant_images and not table_imgs:
                    st.info("No matching screenshots or tables found.")

                st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Ask something!")

    with st.form(key='chat_form', clear_on_submit=True):
        st.text_input("Input your question", key="input_text")
        st.form_submit_button(label="Send", on_click=get_chat_response)


