
# import streamlit as st
# from langchain.chains import create_retrieval_chain
# from models.llm_models import document_chain


# def get_chat_response():
#     input_text = st.session_state.input_text
#     if input_text:
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)

#         response = retrieval_chain.invoke({"input": input_text})

#         st.session_state.chat_history.append({"question": input_text, "answer": response["answer"]})

#         st.session_state.input_text = ""


# def render_home():
#     st.title("Welcome to")
#     st.image("images/immortal_logo.png", width=700)
#     st.title("FINACLE PDFs QUERIES")

#     st.markdown("""
#         <style>
#         .chat-container {
#             height: 350px;
#             overflow-y: auto;
#             padding: 10px;
#             border: 1px solid #ddd;
#             border-radius: 10px;
#             background-color: #2E2E2E; 
#             color: #fff; 
#         }
#         .big-question {
#             font-size: 18px;
#             font-weight: bold;
#             color: #fff; 
#         }
#         .chat-container hr {
#             border: 0.5px solid #ccc;
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
#         chat_display = ""
#         for chat in st.session_state.chat_history:
#             chat_display += f"<div class='big-question'>Question :- {chat['question']}</div>\n"
#             chat_display += f"<div>Answer :- \n\n{chat['answer']}</div>\n"
#             chat_display += "<hr>\n"
#         st.markdown(f"<div class='chat-container'>{chat_display}</div>", unsafe_allow_html=True)

#         # Show images if available
#         if 'image_map' in st.session_state:
#             st.markdown("**Related Screenshots:**")
#             for page_num, img_list in st.session_state.image_map.items():
#                 for img_path in img_list:
#                     st.image(img_path, use_column_width=True)

#     else:
#         st.markdown("<div class='chat-container'>No chat history yet.</div>", unsafe_allow_html=True)

#     with st.form(key='chat_form', clear_on_submit=True):
#         st.text_input("Input your question", key="input_text")
#         st.form_submit_button(label="Send", on_click=get_chat_response)



import streamlit as st
from langchain.chains import create_retrieval_chain
from models.llm_models import document_chain


def get_chat_response():
    input_text = st.session_state.input_text
    if input_text:
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"input": input_text})

        source_pages = set()
        if "context" in response:
            for doc in response["context"]:
                if "page" in doc.metadata:
                    source_pages.add(doc.metadata["page"] + 1)

        st.session_state.chat_history.append({
            "question": input_text,
            "answer": response["answer"],
            "pages": list(source_pages)
        })

        st.session_state.input_text = ""



# def render_home():
#     st.title("Welcome to")
#     st.image("images/immortal_logo.png", width=700)
#     st.title("FINACLE PDFs QUERIES")

#     st.markdown("""
#         <style>
#         .chat-container {
#             height: auto;
#             overflow-y: auto;
#             padding: 10px;
#             border: 1px solid #ddd;
#             border-radius: 10px;
#             background-color: #2E2E2E; 
#             color: #fff; 
#         }
#         .big-question {
#             font-size: 18px;
#             font-weight: bold;
#             color: #fff; 
#         }
#         .chat-container hr {
#             border: 0.5px solid #ccc;
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
#         chat_display = ""
#         for chat in st.session_state.chat_history:
#             chat_display += f"<div class='big-question'>Question :- {chat['question']}</div>\n"
#             chat_display += f"<div>Answer :- \n\n{chat['answer']}</div>\n"
#             chat_display += "<hr>\n"
#         st.markdown(f"<div class='chat-container'>{chat_display}</div>", unsafe_allow_html=True)

#         if 'image_map' in st.session_state and "pages" in chat:
#             st.markdown(f"**Related Screenshots:**")
#             for page in chat["pages"]:
#                 if page in st.session_state.image_map:
#                     for img_path in st.session_state.image_map[page]:
#                         st.image(img_path, use_column_width=True)
#     else:
#         st.markdown("<div class='chat-container'>No chat history yet.</div>", unsafe_allow_html=True)

#     with st.form(key='chat_form', clear_on_submit=True):
#         st.text_input("Input your question", key="input_text")
#         st.form_submit_button(label="Send", on_click=get_chat_response)


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

                if 'image_map' in st.session_state and "pages" in chat:
                    related_images_found = False
                    for page in chat["pages"]:
                        if page in st.session_state.image_map:
                            for img_path in st.session_state.image_map[page]:
                                st.image(img_path, use_column_width=True)
                                related_images_found = True

                    if not related_images_found:
                        st.info("No related screenshots found.")

                st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Ask something!")

    # Chat form
    with st.form(key='chat_form', clear_on_submit=True):
        st.text_input("Input your question", key="input_text")
        st.form_submit_button(label="Send", on_click=get_chat_response)



# def render_home():
#     st.title("Welcome to")
#     st.image("images/immortal_logo.png", width=700)
#     st.title("FINACLE PDFs QUERIES")

#     st.markdown("""
#         <style>
#         .chat-scroll {
#             max-height: 500px;
#             overflow-y: scroll;
#             border: 1px solid #ddd;
#             border-radius: 10px;
#             padding: 10px;
#             background-color: #2E2E2E;
#             color: white;
#         }
#         .chat-block {
#             margin-bottom: 20px;
#             padding-bottom: 10px;
#             border-bottom: 1px solid #444;
#         }
#         .big-question {
#             font-size: 18px;
#             font-weight: bold;
#             color: #fff; 
#         }
#         .chat-image {
#             margin-top: 10px;
#             margin-bottom: 10px;
#             border-radius: 8px;
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
#         chat_html = "<div class='chat-scroll'>"

#         for chat in st.session_state.chat_history:
#             chat_html += "<div class='chat-block'>"
#             chat_html += f"<div class='big-question'>Question :- {chat['question']}</div>\n"
#             chat_html += f"<div>Answer :- <br>{chat['answer']}</div>\n"

#             # Display related images right below answer
#             if 'image_map' in st.session_state and "pages" in chat:
#                 for page in chat["pages"]:
#                     if page in st.session_state.image_map:
#                         for img_path in st.session_state.image_map[page]:
#                             chat_html += f"<img src='file://{img_path}' class='chat-image' width='100%'><br>"

#             chat_html += "</div>"

#         chat_html += "</div>"
#         st.markdown(chat_html, unsafe_allow_html=True)
#     else:
#         st.markdown("<div class='chat-scroll'>No chat history yet.</div>", unsafe_allow_html=True)

#     with st.form(key='chat_form', clear_on_submit=True):
#         st.text_input("Input your question", key="input_text")
#         st.form_submit_button(label="Send", on_click=get_chat_response)