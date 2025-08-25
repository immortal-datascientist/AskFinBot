import streamlit as st
from files.pdf_upload import handle_pdf_upload

def render_sidebar():
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Select a page:", ["Home", "Ask_SQL_query", "Sample 2"])

    st.sidebar.title(" ")
    st.sidebar.title(" ")
    st.sidebar.subheader("Upload PDF")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        handle_pdf_upload(uploaded_file)

    return option