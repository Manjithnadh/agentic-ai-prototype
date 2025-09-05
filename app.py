import os
import streamlit as st
from agents.agents1 import app as agent_app, set_qa_chain

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="Agentic AI Assistant", layout="centered")

st.title("🤖 Agentic AI Chatbot")
st.write("Ask me about **drugs (SQL)**, **documents (RAG)**")

# -------------------- File Upload for RAG --------------------
uploaded_file = st.file_uploader("📂 Upload a PDF or TXT to enable RAG", type=["pdf", "txt"])

if uploaded_file:
    os.makedirs("uploaded_files", exist_ok=True)
    file_path = os.path.join("uploaded_files", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    set_qa_chain(file_path)  # dynamically activate RAG
    st.success(f"✅ File `{uploaded_file.name}` uploaded. RAG enabled!")

# -------------------- Initialize chat history --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- Chat Input --------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.chat_history.append(("You", user_input))

    # Run through LangGraph app
    result = agent_app.invoke({"query": user_input})
    bot_reply = result["response"]

    # Append bot response
    st.session_state.chat_history.append(("Bot", bot_reply))

# -------------------- Display Chat --------------------
chat_box = st.container()
with chat_box:
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")
