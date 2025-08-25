# llm_models.py

import os
import subprocess
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

os.environ["OLLAMA_GPU"] = "1"
os.environ["OLLAMA_GPU_LAYERS"] = "35"



# --- GPU check function ---
def check_gpu():
    """Check if Ollama is using GPU."""
    try:
        # Check Ollama's process list
        ollama_check = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if "GPU" in ollama_check.stdout or "cuda" in ollama_check.stdout.lower():
            return True

        # Fallback: Check NVIDIA GPU usage
        nvidia_check = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name,used_gpu_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if "ollama" in nvidia_check.stdout.lower():
            return True
    except FileNotFoundError:
        print("⚠️ NVIDIA tools not found — is the driver installed?")
    except Exception as e:
        print(f"⚠️ GPU check failed: {e}")

    return False

# --- Load LLM only once per session ---
if "llm" not in st.session_state:
    st.info("🚀 Loading Llama3 model into GPU... Please wait.")
    st.session_state.llm = OllamaLLM(
        model="llama3",
        temperature=0.5
    )

    if check_gpu():
        st.success("✅ Model loaded and running on GPU!")
    else:
        st.warning("⚠️ Model loaded, but GPU not detected. Running on CPU.")

llm = st.session_state.llm




# # ✅ Local LLM using Ollama
# llm = OllamaLLM(
#     model="llama3",  
#     temperature=0.5
# )

# --- Helper to preserve PDF formatting ---
def build_context(docs):
    """
    Joins document chunks with double newlines to preserve bullet points,
    code indentation, and section spacing.
    """
    return "\n\n".join([getattr(doc, "page_content", "") for doc in (docs or [])])


prompt_template = ChatPromptTemplate.from_template(
    """
You are a strict assistant that answers ONLY from the given PDF context.

Rules:
- If the answer exists in the context, return the **full relevant section** exactly as written, including any surrounding code, variable declarations, or examples that directly relate to the question.
- Merge all provided context chunks in the correct logical order to ensure the complete answer is returned.
- If there is **any code, SQL, or PL/SQL**, wrap it inside triple backticks with the correct language tag so it renders with syntax highlighting in Markdown. 
  Examples:
```sql
SELECT * FROM customers;
- Preserve all original formatting, indentation, and line breaks — especially for code, tables, or examples.
- Do NOT rephrase, summarize, or omit any part of the relevant text.
- Do NOT add general knowledge, assumptions, or hypothetical examples.
- If the answer is NOT fully present in the context, reply exactly with:
  "⚠️ Not enough information in the provided PDF. Use DuckDuckGo search."

<context>
{context}
</context>

Question: {input}

Answer:
"""
)

document_chain = create_stuff_documents_chain(llm, prompt_template)


def generate_sql_query(prompt: str) -> str:
    system_prompt = (
        "You are a SQL expert. Convert the following natural language request "
        "into a valid, executable MySQL query.\n"
        "- Use appropriate table and column names based on the request.\n"
        "- Assume the database has multiple tables with meaningful names.\n"
        "- Only return the raw SQL query. Do not explain, format, or wrap it in markdown.\n"
        "- Do NOT use backticks or code blocks and don't use ';' at the end.\n"
        "- The output should be syntactically correct MySQL.\n"
        f"User Request: {prompt}\n"
        "SQL:"
    )

    return llm.invoke(system_prompt).strip()


def extract_sql_only(text: str) -> str:
    # Basic extractor to get SQL from LLM response
    lines = text.strip().splitlines()
    sql_lines = [line for line in lines if not line.strip().lower().startswith("sql")]
    return "\n".join(sql_lines).strip().strip("`")









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



