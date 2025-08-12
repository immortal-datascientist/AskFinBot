import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from fpdf import FPDF
import time
import random
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

huggingface_api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "area" not in st.session_state:
    st.session_state.area = ""

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Select a page:", ["Home", "Sample 1", "Sample 2"])    

st.sidebar.title("                                        ")
st.sidebar.title("                                        ")
st.sidebar.title("                                        ")

st.sidebar.subheader("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()

        with open("temp_uploaded_file.pdf", "wb") as f:  
            f.write(uploaded_file.getbuffer())

        st.session_state.loader = PyPDFLoader("temp_uploaded_file.pdf")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

##### groq models #####
#llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-groq-8b-8192-tool-use-preview")
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

# llm = HuggingFaceEndpoint(
#     endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
#     temperature = 0.5
# )

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt_template)





class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        pass

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'C')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()




if option == "Home":
    st.title("Welcome to")

    st.image("images/immortal_logo.png", width=700)

    st.title("FINACLE PDFs QUERIES")

    st.markdown("""
        <style>
        .chat-container {
            height: 350px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #2E2E2E; 
            color: #fff; 
        }
                
        .big-question {
            font-size: 18px;
            font-weight: bold;
            color: #fff; 
        }
                
        .chat-container hr {
            border: 0.5px solid #ccc;
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

    # Display chat history in a scrollable container
    if st.session_state.chat_history:
        chat_display = ""
        for chat in st.session_state.chat_history:
            chat_display += f"<div class='big-question'>Question :- {chat['question']}</div>\n"
            chat_display += f"<div>Answer :- \n\n{chat['answer']}</div>\n"
            chat_display += "<hr>\n"
        
        st.markdown(f"<div class='chat-container'>{chat_display}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='chat-container'>No chat history yet.</div>", unsafe_allow_html=True)

    def update_chat():
        input_text = st.session_state.input_text
        if input_text:
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            #start_time = time.time()
            response = retrieval_chain.invoke({"input": input_text})
            #end_time = time.time()

            st.session_state.chat_history.append({"question": input_text, "answer": response["answer"]})

            # total_time = end_time - start_time
            # st.write(f"Response time: {total_time:.2f} seconds")

            # Clear the input text
            st.session_state.input_text = ""

    with st.form(key='chat_form', clear_on_submit=True):
        st.text_input("Input your question", key="input_text")
        submit = st.form_submit_button(label="Send", on_click=update_chat)