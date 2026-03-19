from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os


# 🔹 SET YOUR GROQ API KEY HERE
GROQ_API_KEY = ""


# 🔹 Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# 🔹 Create vector store
def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# 🔹 Build QA chain
def build_qa_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,   # ✅ API used here
        model_name="llama3-8b-8192"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa_chain


# 🔹 Ask question
def ask_question(qa_chain, query):
    return qa_chain.run(query)