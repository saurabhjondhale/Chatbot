import os
from langchain_community.document_loaders import TextLoader
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
#from langchain_community.llms import Ollama  # 🔁 Local LLM instead of OpenAI

# === CONFIG ===
TXT_PATH = ""

# === STEP 1: Load .txt File ===
loader = TextLoader(TXT_PATH)
documents = loader.load_and_split()

# === STEP 2: Create Local Embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embedding=embeddings)

# === STEP 3: Use Ollama for LLM ===
llm = OllamaLLM(model="mistral")  # You can replace "llama3" with "mistral", "gemma", etc.

# === STEP 4: Create QA Chain ===
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# === STEP 5: Ask Questions ===
while True:
    query = input("Ask your question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa.invoke(query)
    print("\nAnswer:", answer)
