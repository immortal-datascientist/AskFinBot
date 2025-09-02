

########################################## working well ###########################################

# chat_ui.py
import os
import streamlit as st
from langchain.chains import create_retrieval_chain
from sql_query.sql_utils import get_all_users, get_user_tables, fetch_table_preview, execute_sql_query
from models.llm_models import document_chain, generate_sql_query, extract_sql_only
from ocr_extractor.ocr_image_matcher import find_relevant_images_with_keywords
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
import hashlib
import pdfplumber
import re
from typing import Any, Dict, Optional, Set


search_tool = DuckDuckGoSearchRun()

# -------------------- NEW: tiny helpers (added below imports) -------------------- #
def _safe_get_page_plus1(meta: Dict[str, Any]) -> Optional[int]:
    """
    Your pipeline stores page as 0-based in metadata['page'] (you were adding +1 later).
    This reads common keys and returns 1-based page index if found, else None.
    """
    if not meta:
        return None
    # Primary key you already use
    if "page" in meta:
        try:
            return int(meta["page"]) + 1
        except Exception:
            pass
    # Fallbacks in case of different key names
    for k in ("page_number", "page_no", "source_page"):
        if k in meta:
            try:
                v = int(meta[k])
                # If it already looks 1-based, keep it as-is
                return v if v > 0 else v + 1
            except Exception:
                continue

    # Sometimes page is embedded in a string (e.g., "path...page=12")
    for k in ("loc", "source", "file", "path"):
        v = meta.get(k)
        if isinstance(v, str):
            m = re.search(r"page\s*=?\s*(\d+)", v, flags=re.I)
            if m:
                try:
                    p = int(m.group(1))
                    return p if p > 0 else p + 1
                except Exception:
                    pass
    return None


def _allowed_pages_from(page_1based: Optional[int], forward: int = 4) -> Set[int]:
    """
    Build allowed window {page, page+1, ..., page+forward}.
    Example: page=35 -> {35,36,37,38,39} when forward=4.
    """
    if page_1based is None:
        return set()
    end = page_1based + max(0, int(forward))
    return set(range(page_1based, end + 1))
# ------------------------------------------------------------------------------- #


def is_irrelevant(answer_text):
    """Detect irrelevant answers based on common phrases or low length."""
    if not answer_text or len(answer_text.strip().split()) < 3:
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

    # Debug: Show selected PDF/collection
    print("DEBUG: Selected PDF/Collection:", st.session_state.get("selected_pdf", "Unknown"))

    # Debug: Check embedding vector size
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    test_vec = embedder.embed_query("test")
    print("DEBUG: Embedding vector size used for retrieval:", len(test_vec))

    retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k": 8})  # ############ Change if want to access more chunks 
    results = retriever.get_relevant_documents(input_text)

    # Debug: Show retrieved chunks
    print(f"DEBUG: Retrieved {len(results)} chunks from Qdrant for query '{input_text}'")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content[:200], "...\n")

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": input_text})

    matched_chunks = response.get("context", [])
    # Your previous computation (kept): turn 0-based -> 1-based
    matched_pages = [doc.metadata.get("page", 0) + 1 for doc in matched_chunks]

    # -------------------- NEW: determine the single source page and allowed window -------------------- #
    # Prefer the top result from 'results' (already sorted by similarity), otherwise fallback to matched_pages[0]
    source_page = None
    if results:
        try:
            source_page = _safe_get_page_plus1(results[0].metadata)
        except Exception:
            source_page = None
    if source_page is None:
        source_page = matched_pages[0] if matched_pages else None

    allowed_pages = _allowed_pages_from(source_page, forward=4)  # page..page+4
    print(f"DEBUG: Source page (1-based): {source_page}, Allowed pages: {sorted(list(allowed_pages))}")

    # Keep matched_pages only within the allowed window (so downstream ops remain consistent)
    matched_pages = [p for p in matched_pages if p in allowed_pages]
    # -------------------------------------------------------------------------------------------------- #

    answer_text = response.get("answer", "").strip()

    if is_irrelevant(answer_text):
        print("DEBUG: Answer deemed irrelevant, falling back to DuckDuckGo.")
        fallback = search_tool.run(input_text)
        answer_text += f"\n\n🌐 *Extra info from the web:* {fallback}"

    # -------------------- IMAGES: restrict to allowed pages -------------------- #
    # Your function already ranks images; we just limit the candidate page list we pass in.
    matched_images = find_relevant_images_with_keywords(
        st.session_state.image_map,
        input_text,
        matched_chunks,
        matched_pages,   # now already trimmed to allowed window
        top_k=5
    )

    # If the image matcher returns tuples (img_path, score, page), filter strictly by allowed_pages again (safety)
    filtered_images = []
    for item in matched_images:
        try:
            img_path, score, page = item
            if page in allowed_pages:
                filtered_images.append(item)
        except Exception:
            # If the item shape is different, keep original behavior but warn
            print("WARN: Unexpected image tuple shape; skipping strict page filter for this item.", item)
            filtered_images.append(item)
    matched_images = filtered_images
    # -------------------------------------------------------------------------- #

    # -------------------- TABLES: restrict to allowed pages -------------------- #
    table_images = []
    if "table_image_map" in st.session_state:
        for page in sorted(list(allowed_pages)):  # only pages in the window
            if page in st.session_state.table_image_map:
                table_images.extend(st.session_state.table_image_map[page])
    # -------------------------------------------------------------------------- #

    # persist
    st.session_state.chat_history.append({
        "question": input_text,
        "answer": answer_text,
        "matched_chunks": matched_chunks,
        "matched_pages": matched_pages,
        "matched_images": matched_images,
        "matched_tables": table_images
    })

    st.session_state.input_text = ""


def hash_image(img_path):
    with open(img_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    


###################################################################################
def _find_footer_cut_y_chat(page: pdfplumber.page.Page) -> float:
    """
    Same as in pdf_upload.py but local to chat_ui.py, returns a y cutoff.
    If not found, returns page.height (i.e., no cropping).
    """
    W, H = page.width, page.height
    min_width = 0.60 * W
    min_y = 0.75 * H

    best_y = None
    for ln in getattr(page, "lines", []):
        if abs(ln.get("y0") - ln.get("y1")) < 1.0:
            width = abs(ln.get("x1") - ln.get("x0"))
            y = (ln.get("y0") + ln.get("y1")) / 2.0
            if width >= min_width and y >= min_y:
                if best_y is None or y > best_y:
                    best_y = y

    if best_y is None:
        for rc in getattr(page, "rects", []):
            x0, y0, x1, y1 = rc.get("x0"), rc.get("y0"), rc.get("x1"), rc.get("y1")
            height = abs(y1 - y0)
            width = abs(x1 - x0)
            y_mid = (y0 + y1) / 2.0
            if height <= 1.5 and width >= min_width and y_mid >= min_y:
                if best_y is None or y_mid > best_y:
                    best_y = y_mid

    return best_y if best_y is not None else H



def extract_table_images_from_pdf(pdf_path, output_folder="extracted_tables"):
    os.makedirs(output_folder, exist_ok=True)
    table_images = {}  # {page_number: [image_paths]}

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # detect footer border and crop ABOVE it
            cut_y = _find_footer_cut_y_chat(page)
            # When not found, cut_y == page.height (i.e., no crop)
            crop_bbox = (0, 0, page.width, max(0, cut_y - 1))
            page_cropped = page.within_bbox(crop_bbox)

            # IMPORTANT: find tables on the cropped page only
            tables = page_cropped.find_tables()
            for j, table in enumerate(tables):
                bbox = table.bbox  # (x0, top, x1, bottom) already above footer
                cropped_img = page_cropped.within_bbox(bbox).to_image(resolution=200)
                path = os.path.join(output_folder, f"page{i+1}_table{j+1}.png")
                cropped_img.save(path, format="PNG")
                table_images.setdefault(i + 1, []).append(path)

    return table_images
# ###################################################################################

def render_home():
    st.title("Welcome to")
    st.image("images/immortal_logo.png", width=700)
    st.title("AskFinBot - FINACLE PDFs QUERIES")

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
                        st.image(img_path, use_container_width=True)

                if relevant_images:
                    st.markdown("**🖼️ Relevant Screenshot(s):**")
                    for img_path, score, page in relevant_images:
                        st.image(img_path, use_container_width=True)

                if not relevant_images and not table_imgs:
                    st.info("No matching screenshots or tables found.")

                st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Ask something!")

    with st.form(key='chat_form', clear_on_submit=True):
        st.text_input("Input your question", key="input_text")
        st.form_submit_button(label="Send", on_click=get_chat_response)









# ############ SQL Part working properly###########
# def render_sql_query():
#     st.title("🔎 Ask SQL Query")

#     # Step 1: Prompt
#     user_prompt = st.text_input(
#         "💬 Describe your data need:",
#         placeholder="e.g., show employees with salary above 30000 and age below 30"
#     )

#     # Step 2: Generate SQL from LLM
#     if st.button("⚙️ Generate SQL"):
#         if user_prompt.strip():
#             raw_sql = generate_sql_query(user_prompt)
#             clean_sql = extract_sql_only(raw_sql)
#             st.session_state.sql_query = clean_sql
#         else:
#             st.warning("Please enter your question before generating SQL.")

#     # Step 3: Editable SQL
#     sql_code = st.text_area(
#         "🛠️ Edit the SQL Query below:",
#         value=st.session_state.get("sql_query", ""),
#         height=150
#     )

#     # Step 4: Run SQL query
#     if st.button("📤 Provide"):
#         df, error = execute_sql_query(sql_code)
#         if error:
#             st.error(f"❌ Error executing query: {error}")
#             st.session_state.sql_result = None
#         else:
#             st.session_state.sql_result = df

#     # Step 5: Show result
#     if st.session_state.get("sql_result") is not None:
#         st.subheader("📋 Result")
#         st.dataframe(st.session_state.sql_result)

# def render_sql_query():
#     st.title("🤖 AskFinBot - Oracle Edition")

#     # --- Section 1: Explore Users ---
#     st.header("1️⃣ Database Users")
#     users = get_all_users()
#     selected_user = st.selectbox("Select a User", users)

#     # --- Section 2: Explore Tables ---
#     if selected_user:
#         tables = get_user_tables(selected_user)
#         st.subheader(f"Tables under user `{selected_user}`")
#         selected_table = st.selectbox("Choose a Table", tables)

#         # --- Section 3: Table Preview ---
#         if selected_table:
#             st.subheader(f"Preview of `{selected_user}.{selected_table}`")
#             df_preview = fetch_table_preview(selected_user, selected_table)
#             st.dataframe(df_preview)

#     # --- Section 4: SQL Query Generation & Execution ---
#     st.header("2️⃣ Ask SQL Query")
#     user_prompt = st.text_input("💬 Describe your data need:", placeholder="e.g., show employees with salary above 30000")

#     if st.button("⚙️ Generate SQL"):
#         if user_prompt.strip():
#             sql_text = generate_sql_query(user_prompt)  # <-- Uses generate_sql_query
#             st.session_state.sql_query = extract_sql_only(sql_text)  # <-- Cleans using extract_sql_only
#         else:
#             st.warning("Please enter your question.")


#     # Editable SQL
#     sql_code = st.text_area("🛠️ Edit the SQL Query below:", value=st.session_state.get("sql_query", ""), height=150)

#     # Button: Run SQL
#     if st.button("📤 Provide"):
#         df, error = execute_sql_query(sql_code)
#         if error:
#             st.error(f"❌ Error executing query: {error}")
#             st.session_state.sql_result = None
#         else:
#             st.session_state.sql_result = df

#     # Show result
#     if st.session_state.get("sql_result") is not None:
#         st.subheader("📋 Result")
#         st.dataframe(st.session_state.sql_result)



###############################################################################################


def render_sql_query():
    st.title("🤖 AskFinBot - Oracle Edition")

    # --- Section 1: Explore Users ---
    st.header("1️⃣ Database Users")
    users = get_all_users()
    if users:
        selected_user = st.selectbox("Select a User", users)
    else:
        st.error("❌ Could not fetch users from the database.")
        return  # Stop execution if users not available

    # --- Section 2: Explore Tables ---
    if selected_user:
        tables = get_user_tables(selected_user)
        if tables:
            st.subheader(f"Tables under user `{selected_user}`")
            selected_table = st.selectbox("Choose a Table", tables)
        else:
            st.warning(f"No tables found for user `{selected_user}`.")
            selected_table = None

        # --- Section 3: Table Preview ---
        if selected_table:
            st.subheader(f"Preview of `{selected_user}.{selected_table}`")
            try:
                df_preview = fetch_table_preview(selected_user, selected_table)
                if df_preview is not None and not df_preview.empty:
                    st.dataframe(df_preview)
                else:
                    st.info("Table is empty or preview couldn't be loaded.")
            except Exception as e:
                st.error(f"⚠️ Error loading preview: {e}")

    # --- Section 4: SQL Query Generation & Execution ---
    st.header("2️⃣ Ask SQL Query")
    user_prompt = st.text_input("💬 Describe your data need:", placeholder="e.g., show employees with salary above 30000")

    if st.button("⚙️ Generate SQL"):
        if user_prompt.strip():
            try:
                sql_text = generate_sql_query(user_prompt)
                cleaned_sql = extract_sql_only(sql_text)
                st.session_state.sql_query = cleaned_sql
                st.success("✅ SQL generated successfully!")
            except Exception as e:
                st.error(f"❌ Failed to generate SQL: {e}")
        else:
            st.warning("Please enter your question.")

    # --- Editable SQL Text Area ---
    sql_code = st.text_area("🛠️ Edit the SQL Query below:", value=st.session_state.get("sql_query", ""), height=150)

    # --- Button: Run SQL ---
    if st.button("📤 Provide"):
        if sql_code.strip():
            df, error = execute_sql_query(sql_code)
            if error:
                st.error(f"❌ Error executing query: {error}")
                st.session_state.sql_result = None
            else:
                st.session_state.sql_result = df
                st.success("✅ Query executed successfully!")
        else:
            st.warning("SQL code is empty.")

    # --- Show Result ---
    if st.session_state.get("sql_result") is not None:
        st.subheader("📋 Result")
        st.dataframe(st.session_state.sql_result)
