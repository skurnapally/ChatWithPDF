# generator.py

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_faiss_index(pdf_path, save_path):
    print(f"📄 Loading and splitting: {pdf_path}")
    chunks = load_and_split_pdf(pdf_path)

    print("🔍 Creating FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"✅ FAISS index saved at: {save_path}")

# 🚀 Run this only when you want to generate the index
if __name__ == "__main__":
    pdf_path = "AttentionisAllyouneed.pdf"
    index_path = "faiss_index"
    create_faiss_index(pdf_path, index_path)
