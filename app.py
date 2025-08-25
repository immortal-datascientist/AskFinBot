
##################################  Original code ######################################
"""
Main entry point for the Streamlit app.
Handles routing to different pages using the sidebar.
Initializes session state variables and delegates rendering to component modules.
If you want to run the UI use this in command prompt "streamlit run app.py"
"""

import streamlit as st
from interface.sidebar import render_sidebar
from user_interface.chat_ui import render_home
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pdf_manager.pdf_storage import list_saved_pdfs, load_vectors_qdrant, load_image_map, load_table_map
from user_interface.chat_ui import render_home, render_sql_query

st.sidebar.title("📂 Select a Stored PDF")
stored_pdfs = list_saved_pdfs()
selected_pdf = st.sidebar.selectbox("Choose PDF:", stored_pdfs)

if selected_pdf:
    st.session_state.vectors = load_vectors_qdrant(selected_pdf)
    st.session_state.image_map = load_image_map(selected_pdf)
    st.session_state.table_image_map = load_table_map(selected_pdf)

# Initialize session state variables if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "area" not in st.session_state:
    st.session_state.area = ""

if "sql_query" not in st.session_state:
    st.session_state.sql_query = ""

if "sql_result" not in st.session_state:
    st.session_state.sql_result = None

# --- Sidebar navigation logic ---
option = render_sidebar()

# --- Page routing ---
if option == "Home":
    st.success(f"Loaded preprocessed PDF: {selected_pdf}")
    render_home()

elif option == "Ask_SQL_query":
    render_sql_query()

