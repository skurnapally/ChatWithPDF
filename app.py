# app.py

import os
import tempfile
import base64
import shutil
import streamlit as st
from generator import create_faiss_index
from interpreter import ask_question_with_gemini
from dotenv import load_dotenv

# Load environment variables (for Google API key)
load_dotenv()

st.set_page_config(page_title="📄 Document Q&A with Gemini", layout="centered")

# 🔧 Custom background using local image
with open("background.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

page_bg_img = f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-attachment: fixed;
    padding: 2rem;
    border-radius: 10px;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("📄 Document Retriever using Gemini 2.0 + FAISS")

# Use session state to persist FAISS index creation and track uploaded file
if "faiss_created" not in st.session_state:
    st.session_state["faiss_created"] = False
if "faiss_path" not in st.session_state:
    st.session_state["faiss_path"] = None
if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None
if "temp_dir" not in st.session_state:
    st.session_state["temp_dir"] = None
if "answers" not in st.session_state:
    st.session_state["answers"] = []

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Check if the uploaded file is new
    if uploaded_file.name != st.session_state["uploaded_filename"]:
        # Remove old temp_dir and FAISS index if it exists
        if st.session_state["temp_dir"]:
            shutil.rmtree(st.session_state["temp_dir"], ignore_errors=True)

        # Reset state
        st.session_state["faiss_created"] = False
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.session_state["answers"] = []

    if not st.session_state["faiss_created"]:
        temp_dir = tempfile.mkdtemp()
        faiss_path = os.path.join(temp_dir, "faiss_index")
        pdf_path = os.path.join(temp_dir, uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("📚 Processing document and creating FAISS index..."):
            create_faiss_index(pdf_path, faiss_path)

        st.session_state["faiss_created"] = True
        st.session_state["faiss_path"] = faiss_path
        st.session_state["temp_dir"] = temp_dir
        st.success("✅ Document indexed successfully!")

# Input question (only visible after FAISS index is created)
if st.session_state["faiss_created"]:
    user_question = st.text_input("Ask a question based on the uploaded document")

    # Question handler
    if user_question:
        with st.spinner("🤖 Thinking..."):
            try:
                answer = ask_question_with_gemini(st.session_state["faiss_path"], user_question)
                st.session_state["answers"].append((user_question, answer))
                st.markdown(f"### 💡 Answer:\n{answer}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")