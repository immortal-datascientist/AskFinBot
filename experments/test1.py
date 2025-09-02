######################################## This is Gdrive code #########################################



# # chat_ui.py
# import streamlit as st 
# from langchain.chains import create_retrieval_chain
# from models.llm_models import document_chain
# from ocr_extractor.ocr_image_matcher import find_relevant_images_with_keywords
# from langchain_community.tools import DuckDuckGoSearchRun
# import os
# import hashlib
# from PIL import Image
# import pdfplumber

# search_tool = DuckDuckGoSearchRun()

# def is_irrelevant(answer_text):
#     """Detect irrelevant answers based on common phrases or low length."""
#     if not answer_text or len(answer_text.strip().split()) < 10:
#         return True
    
#     answer_lower = answer_text.lower()
#     irrelevant_keywords = [
#         "not related to the provided context",
#         "i cannot provide",
#         "does not mention",
#         "no relevant information",
#         "unable to answer",
#         "context is about",
#         "outside the scope",
#         "not covered",
#         "information is missing"
#     ]
#     return any(kw in answer_lower for kw in irrelevant_keywords)

# def get_chat_response():
#     input_text = st.session_state.input_text
#     if not input_text:
#         return

#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     response = retrieval_chain.invoke({"input": input_text})

#     matched_chunks = response.get("context", [])
#     matched_pages = [doc.metadata["page"] + 1 for doc in matched_chunks]
#     answer_text = response.get("answer", "").strip()

#     # 🕵️ Check if answer is irrelevant or vague
#     if is_irrelevant(answer_text):
#         fallback = search_tool.run(input_text)
#         answer_text += f"\n\n🌐 *Extra info from the web:* {fallback}"

#     # 🔍 Get relevant images using OCR
#     matched_images = find_relevant_images_with_keywords(
#         st.session_state.image_map,
#         input_text,
#         matched_pages,
#         top_k=5
#     )

#     # 📊 Get table images for those pages
#     table_images = []
#     if "table_image_map" in st.session_state:
#         for page in matched_pages:
#             if page in st.session_state.table_image_map:
#                 table_images.extend(st.session_state.table_image_map[page])

#     # 💾 Save full response
#     st.session_state.chat_history.append({
#         "question": input_text,
#         "answer": answer_text,
#         "matched_chunks": matched_chunks,
#         "matched_pages": matched_pages,
#         "matched_images": matched_images,
#         "matched_tables": table_images
#     })

#     # 🔄 Reset input
#     st.session_state.input_text = ""


# def hash_image(img_path):
#     with open(img_path, "rb") as f:
#         return hashlib.md5(f.read()).hexdigest()
    

# def extract_table_images_from_pdf(pdf_path, output_folder="extracted_tables"):
#     os.makedirs(output_folder, exist_ok=True)
#     table_images = {}  # {page_number: [image_paths]}

#     with pdfplumber.open(pdf_path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             tables = page.find_tables()
#             for j, table in enumerate(tables):
#                 bbox = table.bbox  # (x0, top, x1, bottom)
#                 cropped = page.within_bbox(bbox).to_image(resolution=200)
#                 path = os.path.join(output_folder, f"page{i+1}_table{j+1}.png")
#                 cropped.save(path, format="PNG")
#                 table_images.setdefault(i + 1, []).append(path)

#     return table_images


# def render_home():
#     st.title("Welcome to")
#     st.image("images/immortal_logo.png", width=700)
#     st.title("FINACLE PDFs QUERIES")

#     st.markdown("""
#         <style>
#         .scrollable-history {
#             max-height: 500px;
#             overflow-y: auto;
#             padding: 10px;
#             border: 1px solid #ddd;
#             border-radius: 10px;
#             background-color: #2E2E2E;
#             color: white;
#         }
#         .question {
#             font-size: 18px;
#             font-weight: bold;
#             margin-bottom: 5px;
#         }
#         .answer {
#             margin-bottom: 10px;
#         }
#         .stButton button {
#             background-color: red;
#             color: white;
#             width: 100%;
#             height: 50px;
#             border-radius: 10px;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     if st.session_state.chat_history:
#         with st.container():
#             st.markdown("<div class='scrollable-history'>", unsafe_allow_html=True)

#             for chat in st.session_state.chat_history:
#                 st.markdown(f"<div class='question'>❓ Question:</div> {chat['question']}", unsafe_allow_html=True)
#                 st.markdown(f"<div class='answer'>✅ Answer:<br>{chat['answer']}</div>", unsafe_allow_html=True)

#                 relevant_images = chat.get("matched_images", [])
#                 table_imgs = chat.get("matched_tables", [])

#                 if table_imgs:
#                     st.markdown("**📊 Detected Table(s):**")
#                     for img_path in table_imgs:
#                         st.image(img_path, use_column_width=True)

#                 if relevant_images:
#                     st.markdown("**🖼️ Relevant Screenshot(s):**")
#                     for img_path, score, page in relevant_images:
#                         st.image(img_path, use_column_width=True)

#                 if not relevant_images and not table_imgs:
#                     st.info("No matching screenshots or tables found.")

#                 st.markdown("<hr>", unsafe_allow_html=True)

#             st.markdown("</div>", unsafe_allow_html=True)
#     else:
#         st.info("No chat history yet. Ask something!")

#     with st.form(key='chat_form', clear_on_submit=True):
#         st.text_input("Input your question", key="input_text")
#         st.form_submit_button(label="Send", on_click=get_chat_response)


################################################################################################################S






# llm_models.py (LOCAL LLaMA 3 VERSION)
##################################### llama.cpp manual installation  #################################
# import os
# import re
# from llama_cpp import Llama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.language_models import BaseLanguageModel
# from langchain_core.documents import Document
# from langchain_core.runnables import Runnable
# from langchain.chains.combine_documents import create_stuff_documents_chain

# # Load local LLaMA 3 GGUF model
# llama_instance = Llama(
#     model_path="E:\\AJAY\\ajay_data\\FINACLE_RAG\\model_weights\\llama3\\Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", 
#     n_gpu_layers=10,     # <== 🟢 Optimal for 8GB GPU
#     n_ctx=2048,          # or increase if needed
#     f16_kv=True,
#     use_mmap=True,
#     use_mlock=False,     # Set to True only if you know what you're doing
#     verbose=True
# )

# # Runnable wrapper for LangChain
# class LocalLlamaLLM(Runnable):
#     def __init__(self, llama_instance):
#         self.llama = llama_instance

#     def invoke(self, input, config=None) -> str:
#         # Handle LangChain's ChatPromptValue or raw input
#         if hasattr(input, "to_string"):
#             raw_prompt = input.to_string()
#         elif isinstance(input, dict):
#             raw_prompt = input.get("input", "")
#         elif isinstance(input, str):
#             raw_prompt = input
#         else:
#             raise TypeError(f"Unsupported input type: {type(input)}")

#         # Use LLaMA 3 instruct formatting for better responses
#         formatted_prompt = f"[INST] {raw_prompt.strip()} [/INST]"

#         response = self.llama(
#             formatted_prompt,
#             stop=["</s>"],   # Simplified stop tokens
#             echo=False
#         )
#         return response["choices"][0]["text"].strip()

# # Wrap your LLaMA model
# llm = LocalLlamaLLM(llama_instance)

# # Prompt template for RAG (PDF-based question answering)
# prompt_template = ChatPromptTemplate.from_template(
#     """
#     Answer the question based on the provided context only.
#     Please provide the most accurate, clear, and complete response.

#     <context>
#     {context}
#     <context>
#     Question: {input}
#     """
# )

# # Create LangChain document answering chain
# document_chain = create_stuff_documents_chain(llm, prompt_template)

# # ✅ SQL generation using the same local LLaMA model
# def generate_sql_query(prompt: str) -> str:
#     system_prompt = (
#         "You are a SQL expert. Convert the following natural language request "
#         "into a valid, executable MySQL query.\n"
#         "- Use appropriate table and column names based on the request.\n"
#         "- Assume the database has multiple tables with meaningful names.\n"
#         "- Only return the raw SQL query. Do not explain, format, or wrap it in markdown.\n"
#         "- Do NOT use backticks or code blocks and don't use ';' at the end.\n"
#         "- The output should be syntactically correct MySQL.\n"
#         f"User Request: {prompt.strip()}\n"
#         "SQL:"
#     )

#     formatted_prompt = f"[INST] {system_prompt} [/INST]"

#     response = llm.llama(
#         formatted_prompt,
#         stop=["</s>"],
#         echo=False
#     )
#     return response["choices"][0]["text"].strip()

# # Cleanup helper
# def extract_sql_only(text: str) -> str:
#     lines = text.strip().splitlines()
#     sql_lines = [line for line in lines if not line.strip().lower().startswith("sql")]
#     return "\n".join(sql_lines).strip().strip("`")

###################################################################################################################



#ocr_image_matcher.py (UPDATED WITH KEYWORD EXTRACTION)
##################################### Faster keyword-based image matcher #################################


############################################ This Is with Sentence Transformers ############################################
# import pytesseract
# from PIL import Image
# import spacy
# from sentence_transformers import SentenceTransformer, util

# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# nlp = spacy.load("en_core_web_sm")

# def extract_text_from_image(image_path):
#     try:
#         image = Image.open(image_path).convert("L")  # grayscale
#         text = pytesseract.image_to_string(image)
#         return text.strip().lower()
#     except Exception:
#         return ""

# def extract_keywords(text):
#     doc = nlp(text)
#     return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]

# def find_relevant_images_with_keywords(image_map, user_question, matched_pages, top_k=5):
#     keywords = extract_keywords(user_question)
#     image_matches = []

#     nearby_pages = set()
#     for p in matched_pages:
#         for offset in range(0, 3):  # only future pages
#             if (p + offset) > 0:
#                 nearby_pages.add(p + offset)

#     checked = 0
#     for page in sorted(nearby_pages):
#         if page in image_map:
#             for img_path in image_map[page]:
#                 ocr_text = extract_text_from_image(img_path)
#                 if not ocr_text:
#                     continue

#                 hits = [kw for kw in keywords if kw in ocr_text]
#                 match_score = len(hits)

#                 if match_score >= 1:
#                     image_matches.append((img_path, match_score, page))

#                 checked += 1
#                 if checked >= 20:
#                     break

#     sorted_images = sorted(image_matches, key=lambda x: x[1], reverse=True)
#     return sorted_images[:top_k]
#################################################################################################



# # ocr_image_matcher.py
# import spacy
# from sentence_transformers import SentenceTransformer
# from typing import List, Tuple

# nlp = spacy.load("en_core_web_sm")
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # kept for potential semantic use

# def extract_keywords(text: str) -> List[str]:
#     doc = nlp(text.lower())
#     return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]

# def find_relevant_images_with_keywords(
#     image_map_with_ocr: dict,  # {page_number: [(img_path, ocr_text), ...]}
#     user_question: str,
#     matched_chunks,
#     matched_pages: List[int],
#     top_k: int = 5
# ) -> List[Tuple[str, int, int]]:
#     """
#     Faster image search:
#     - Uses OCR text cached at PDF ingestion
#     - Combines keywords from user question + retrieved chunks
#     - Restricts to same page ±1
#     - Requires match_score >= 2
#     """

#     # CHANGE ✅ Combined keywords from question and context text
#     context_text = " ".join([doc.page_content for doc in matched_chunks])
#     keywords = extract_keywords(user_question + " " + context_text)

#     image_matches = []
#     nearby_pages = set()

#     # CHANGE ✅ Restrict to ±1 page instead of future 3 pages
#     for p in matched_pages:
#         for offset in range(-1, 2):
#             if (p + offset) > 0:
#                 nearby_pages.add(p + offset)

#     # CHANGE ✅ No OCR here — use cached text
#     for page in sorted(nearby_pages):
#         if page in image_map_with_ocr:
#             for img_path, ocr_text in image_map_with_ocr[page]:
#                 if not ocr_text:
#                     continue

#                 hits = [kw for kw in keywords if kw in ocr_text]
#                 match_score = len(hits)

#                 # CHANGE ✅ Require at least 2 keyword matches
#                 if match_score >= 2:
#                     image_matches.append((img_path, match_score, page))

#     sorted_images = sorted(image_matches, key=lambda x: x[1], reverse=True)
#     return sorted_images[:top_k]





############################################ This Is without Sentence Transformers ############################################
# import pytesseract
# from PIL import Image
# import spacy

# # Removed: from sentence_transformers import SentenceTransformer, util   ## Open This
# # Removed: embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# nlp = spacy.load("en_core_web_sm")

# def extract_text_from_image(image_path):
#     try:
#         image = Image.open(image_path).convert("L")  # grayscale
#         text = pytesseract.image_to_string(image)
#         return text.strip().lower()
#     except Exception:
#         return ""

# def extract_keywords(text):
#     doc = nlp(text)
#     return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]

# def find_relevant_images_with_keywords(image_map, user_question, matched_pages, top_k=5):
#     keywords = extract_keywords(user_question)
#     image_matches = []

#     nearby_pages = set()
#     for p in matched_pages:
#         for offset in range(0, 3):  # only future pages
#             if (p + offset) > 0:
#                 nearby_pages.add(p + offset)

#     checked = 0
#     for page in sorted(nearby_pages):
#         if page in image_map:
#             for img_path in image_map[page]:
#                 ocr_text = extract_text_from_image(img_path)
#                 if not ocr_text:
#                     continue

#                 hits = [kw for kw in keywords if kw in ocr_text]
#                 match_score = len(hits)

#                 if match_score >= 1:
#                     image_matches.append((img_path, match_score, page))

#                 checked += 1
#                 if checked >= 20:
#                     break

#     sorted_images = sorted(image_matches, key=lambda x: x[1], reverse=True)
#     return sorted_images[:top_k]
##################################################################################################