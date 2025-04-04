# interpreter.py

import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import warnings 
warnings.filterwarnings("ignore")
import streamlit as st

# 🔐 Load environment variables from .env file
#load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def ask_question_with_gemini(faiss_path, query):
    print("🔄 Loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

    print("🤖 Initializing Gemini model...")
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    print(f"❓ Asking: {query}")
    result = qa_chain(query)
    return result["result"]

# 🚀 Run this file to interact with the document
if __name__ == "__main__":
   
    index_path = "faiss_index"
    #question = "Summarize the main points of the document."
    question = "what is a transformer model"
    answer = ask_question_with_gemini(index_path, question)
    print("\n💡 Answer:", answer)
