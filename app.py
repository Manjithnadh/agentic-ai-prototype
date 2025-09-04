import os
import streamlit as st
from agents.agents import app as agent_app, set_qa_chain

st.set_page_config(page_title="Agentic AI Assistant", layout="centered")

# -------------------- Title --------------------
st.title("🤖 Agentic AI Chatbot")
st.write("Ask me about **drugs (SQL)**, **documents (RAG)**, or anything else (Fallback).")

# -------------------- File Upload for RAG --------------------
uploaded_file = st.file_uploader("📂 Upload a PDF or TXT to enable RAG", type=["pdf", "txt"])

if uploaded_file:
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    os.makedirs("uploaded_files", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    set_qa_chain(file_path)  # dynamically activate RAG
    st.success(f"✅ File `{uploaded_file.name}` uploaded. RAG enabled!")

# -------------------- Chat Interface --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.chat_history.append(("You", user_input))

    # Run through LangGraph app
    result = agent_app.invoke({"query": user_input})
    bot_reply = result["response"]

    # Save bot reply
    st.session_state.chat_history.append(("Bot", bot_reply))

# -------------------- Display Chat --------------------
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")
