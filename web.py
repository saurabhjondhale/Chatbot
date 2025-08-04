import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# === App Title ===
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("📚 Chat with Your Text File")

# === Upload File ===
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

if uploaded_file:
    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.read())

    # Load document
    loader = TextLoader("temp.txt")
    documents = loader.load_and_split()

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents, embedding=embeddings)

    # Load LLM
    llm = OllamaLLM(model="llama3")
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Chat input
    st.subheader("💬 Ask Your Questions")
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Your question", key="input")

    if user_input:
        with st.spinner("Thinking..."):
            answer = qa.invoke(user_input)
        st.session_state.history.append((user_input, answer))

    # Display conversation
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
