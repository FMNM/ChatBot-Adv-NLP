# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:28:58 2024

@author: Matthew Crowley

title: MATTChat-App.py

description:
    A fully realised RAG chatbot. That responds to questions using information 
    information in the vector store embeddings created in 2-build_vector_db.ipynb.

    References:
        LangChain API docs - https://python.langchain.com/docs/tutorials/qa_chat_history/
        RAG Streamlit WebApp - https://github.com/vikrambhat2/RAG-Implementation-with-ConversationUI/tree/main/Streamlit%20Applications
"""

#=============================================================================
# 0) Import modules 
#=============================================================================
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores import FAISS
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from pathlib import Path
import os
import streamlit as st

# Loading langsmithkey from separate file
from langsmithkey import *


#=============================================================================
# 1) Identify database path 
#=============================================================================
matt_path=Path(os.getcwd())
root_path=matt_path.parents[0]
db_path=root_path.joinpath(r'data\vectorstore\db_faiss')
word_path=root_path.joinpath(r'data\apra_standards\word')


#=============================================================================
# 2) Define HTML templates 
#=============================================================================
bot_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px;">
    <div style="flex-shrink: 0; margin-right: 10px;">
        <img src="https://uxwing.com/wp-content/themes/uxwing/download/communication-chat-call/answer-icon.png" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div style="background-color: #f1f1f1; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

user_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-end;">
    <div style="flex-shrink: 0; margin-left: 10px;">
        <img src="https://cdn.iconscout.com/icon/free/png-512/free-q-characters-character-alphabet-letter-36051.png?f=webp&w=512" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

button_style = """
<style>
    .small-button {
        display: inline-block;
        padding: 5px 10px;
        font-size: 12px;
        color: white;
        background-color: #007bff;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-right: 5px;
    }
    .small-button:hover {
        background-color: #0056b3;
    }
</style>
"""

#=============================================================================
# 3) Load vectorstore, retriever and LLM
#=============================================================================
embedding_model='sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
llm = ChatOllama(model="llama3.2")



#=============================================================================
# 4) Contextualize question
#=============================================================================

# contextualize_q_system_prompt
#   Gets the model to reformulate the latest user query
#   as a standalone question using history as part of 
#   the context. Allowing the appearance of "memory"
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# contextualize_q_prompt 
#   Integrate contextualize_q_system_prompt with current prompt
#   and history points
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Create a chain that takes conversation history and returns documents. (from API docs)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

#=============================================================================
# 5) Answer question
#=============================================================================

# System prompt
#   primes the model to answer questions in a useful manner
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# qa prompt
#   combines system_prompt, chat history and human "input" 
#   questions together for each interaction with the model
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# contextualize_q_system_prompt
#   Gets the model to reformulate the latest user query
#   as a standalone question using history as part of 
#   the context. Allowing the appearance of "memory"
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# contextualize_q_prompt 
#   like qa_prompt above, integrate contextualize_q_system_prompt 
#   with current prompt and history points
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Create a chain that takes conversation history and returns documents. (from API docs)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Create a chain for passing a list of Documents to a model. (from API docs)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create retrieval chain that retrieves documents and then passes them on. (from API docs)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


#=============================================================================
# 6) Statefully manage chat history
#=============================================================================

class State(TypedDict):
    """
        This dict class represents the state of the application.
        It has the same input and output keys as `rag_chain`.
    """
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


def call_model(state: State):
    """
        A simple node that runs the `rag_chain`.

        Inputs:
            State [State] - Custom dictionary defined as above

        Returns:
            Latest addition to the chat history including 
            input message and response.
    """
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


# Our graph consists only of one node:
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Compile the graph with a checkpointer object.
# This persists the state, in this case in memory.
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

#=============================================================================
# 7) Example invocations of the app (demonstrated in 3-MATTChat-review-v2.ipynb)
#=============================================================================

# config = {"configurable": {"thread_id": "abc123"}}

# result = app.invoke(
#     {"input": "What is Task Decomposition?"},
#     config=config,
# )
# print(result["answer"])

#=============================================================================
# 8) Set up streamlit app
#=============================================================================
st.title("MATTChat - Your AI guide to APRA reporting standards!")

# Sidebar for file upload
uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.sidebar.button("Process PDFs"):
        split_docs = prepare_and_split_docs(uploaded_files)
        vector_db = ingest_into_vectordb(split_docs)
        retriever = vector_db.as_retriever()
        st.sidebar.success("Documents processed and vector database created!")

        # Initialize the conversation chain
        conversational_chain = get_conversation_chain(retriever)
        st.session_state.conversational_chain = conversational_chain

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_docs' not in st.session_state:
    st.session_state.show_docs = {}

if 'similarity_scores' not in st.session_state:
    st.session_state.similarity_scores = {}

# Function to toggle the document visibility
def toggle_docs(index):
    st.session_state.show_docs[index] = not st.session_state.show_docs.get(index, False)

# Chat input
user_input = st.text_input("Ask a question about APRA standards:")

if st.button("Submit"):
    st.markdown(button_style, unsafe_allow_html=True)
    if user_input and 'conversational_chain' in st.session_state:
        session_id = "abc123"  # Static session ID for this demo; you can make it dynamic if needed
        conversational_chain = st.session_state.conversational_chain
        response = conversational_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        context_docs = response.get('context', [])
        st.session_state.chat_history.append({"user": user_input, "bot": response['answer'],  "context_docs": context_docs})

# Display chat history
if st.session_state.chat_history:
    for index, message in enumerate(st.session_state.chat_history):
        # Render the user message using the template
        st.markdown(user_template.format(msg=message['user']), unsafe_allow_html=True)
        
        # Render the bot message using the bot template
        st.markdown(bot_template.format(msg=message['bot']), unsafe_allow_html=True)

        # Initialize session state for each message
        if f"show_docs_{index}" not in st.session_state:
            st.session_state[f"show_docs_{index}"] = False
        if f"similarity_score_{index}" not in st.session_state:
            st.session_state[f"similarity_score_{index}"] = None

        # Layout for the buttons in a single row (horizontal alignment)
        cols = st.columns([1, 1])  # Create two equal columns for buttons

        # Render "Show Source Docs" button
        with cols[0]:
            if st.button(f"Show/Hide Source Docs", key=f"toggle_{index}"):
                # Toggle the visibility of source documents for this message
                st.session_state[f"show_docs_{index}"] = not st.session_state[f"show_docs_{index}"]

        # Render "Answer Relevancy" button
        with cols[1]:
            if st.button(f"Calculate Answer Relevancy", key=f"relevancy_{index}"):
                if st.session_state[f"similarity_score_{index}"] is None:
                    score = calculate_similarity_score(message['bot'], message['context_docs'])
                    st.session_state[f"similarity_score_{index}"] = score

        # Check if source documents should be shown
        if st.session_state[f"show_docs_{index}"]:
            with st.expander("Source Documents"):
                for doc in message.get('context_docs', []):
                    st.write(f"Source: {doc.metadata['source']}")
                    st.write(doc.page_content)

        # Display similarity score if available
        if st.session_state[f"similarity_score_{index}"] is not None:
            st.write(f"Similarity Score: {st.session_state[f'similarity_score_{index}']:.2f}")
