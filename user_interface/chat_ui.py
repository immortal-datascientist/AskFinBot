

##################################  Original code ######################################
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
from PIL import Image
import pdfplumber


search_tool = DuckDuckGoSearchRun()

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

##################################### This is Working Well #################################
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

    retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k": 8})  ############ Change if want to access more chunks 
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
    matched_pages = [doc.metadata.get("page", 0) + 1 for doc in matched_chunks]
    answer_text = response.get("answer", "").strip()

    if is_irrelevant(answer_text):
        print("DEBUG: Answer deemed irrelevant, falling back to DuckDuckGo.")
        fallback = search_tool.run(input_text)
        answer_text += f"\n\n🌐 *Extra info from the web:* {fallback}"

    matched_images = find_relevant_images_with_keywords(
        st.session_state.image_map,
        input_text,
        matched_chunks,
        matched_pages,
        top_k=5
    )

    table_images = []
    if "table_image_map" in st.session_state:
        for page in matched_pages:
            if page in st.session_state.table_image_map:
                table_images.extend(st.session_state.table_image_map[page])

    st.session_state.chat_history.append({
        "question": input_text,
        "answer": answer_text,
        "matched_chunks": matched_chunks,
        "matched_pages": matched_pages,
        "matched_images": matched_images,
        "matched_tables": table_images
    })

    st.session_state.input_text = ""
#######################################################################################



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