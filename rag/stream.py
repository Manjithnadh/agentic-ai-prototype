import streamlit as st
from rag1 import create_faiss, ask_ai

st.title("multi rag model")

upload_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf","txt"],
    accept_multiple_files=True
)

# Get current file names
current_file_names = [f.name for f in upload_files] if upload_files else []

# Check if we need to rebuild embeddings (files changed)
if ("retriever" not in st.session_state or st.session_state.get("upload_files") != current_file_names):
    
    if upload_files:
        st.session_state["retriever"] = create_faiss(upload_files)
        st.session_state["upload_files"] = current_file_names
    elif "retriever" in st.session_state:
        # Clear retriever if all files are removed
        del st.session_state["retriever"]
        del st.session_state["upload_files"]

query = st.text_input("ask question about the files")

if query and "retriever" in st.session_state:
    answer = ask_ai(st.session_state["retriever"], query)
    st.write("### answer:")
    st.write(answer)