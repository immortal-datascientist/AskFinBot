import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model="llama3-8b-8192", 
    max_tokens=2048,  
    temperature=0.5
)

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