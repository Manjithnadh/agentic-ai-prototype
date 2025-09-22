import os


import streamlit as st
from agents.agents1 import app as agent_app, memory
from tools.RAG_tool import  create_vectorstore,build_qa  # ✅ use qa_chain
from langchain_openai import ChatOpenAI

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Medical AI Assistant",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    st.markdown("#### 📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT/DOCX files for document queries",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("uploaded_files", exist_ok=True)
        saved_paths = []

        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploaded_files", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)
            

        # ✅ Use your functions from RAG_tool
            vectorstore = create_vectorstore("uploaded_files")
            st.write("Number of chunks in vectorstore:", len(vectorstore.docstore._dict))# returns FAISS
            qa = build_qa(vectorstore)                            # returns RetrievalQA
            st.session_state.qa_chain = qa
            
        
        st.success(f"✅ Uploaded {len(saved_paths)} files successfully")

    st.markdown("---")
    st.markdown("#### ℹ️ System Info")
    st.caption("""
    - **SQL Agent** → Medicine database queries  
    - **RAG Agent** → Document-based questions  
    - **Fallback** → General responses
    """)

    if st.button("🔄 Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

# ----------------- HEADER -----------------
st.markdown("""
<div class="header-container" style="
    background:white; padding:1rem 2rem; border-bottom:1px solid #e0e0e0;
    position:sticky; top:0; z-index:100;">
    <h2 style="margin:0; color:#2c3e50;">⚕️ Medical AI Assistant</h2>
    <p style="margin:0; color:#7f8c8d; font-size:0.9rem;">
        Ask about medicines, uploaded documents, or general queries
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- CHAT HISTORY INIT -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- CHAT DISPLAY -----------------
chat_container = st.container()

with chat_container:
    if st.session_state.chat_history:
        for sender, msg in st.session_state.chat_history:
            if sender == "user":
                st.markdown(f"""
                <div style="background:#e3f2fd; padding:12px 16px; border-radius:12px; 
                            margin:12px 0; margin-left:20%; border:1px solid #bbdefb;">
                    <p style="margin:0;"><strong>👤 You:</strong> {msg}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f5f5f5; padding:12px 16px; border-radius:12px; 
                            margin:12px 0; margin-right:20%; border:1px solid #e0e0e0;">
                    <p style="margin:0;"><strong>🤖 Assistant:</strong> {msg}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#666; margin-top:2rem;">
            <h3>Welcome to Medical AI Assistant</h3>
            <p>Start a conversation by typing your question below 👇</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- CHAT INPUT -----------------
user_input = st.chat_input("Type your message here...", key="main_chat_input")

if user_input:
    # Show user message immediately
    st.session_state.chat_history.append(("user", user_input))
    st.rerun()

# ----------------- PROCESS NEW MESSAGE -----------------
if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
    last_user_input = st.session_state.chat_history[-1][1]

    with st.spinner("🤖 Processing your request..."):
        try:
            # Save user query into memory
            memory.save_context({"input": last_user_input}, {"output": ""})

            # Call agent
            result = agent_app.invoke({
                "query": last_user_input,
                "conversation": memory.load_memory_variables({}).get("history", "")
            })

            # Normalize output
            if isinstance(result, dict):
                bot_reply = result.get("response") or result.get("output") or str(result)
            else:
                bot_reply = str(result)

            # Save response to memory + UI
            memory.save_context({"input": last_user_input}, {"output": bot_reply})
            st.session_state.chat_history.append(("bot", bot_reply))

        except Exception as e:
            import traceback
            print("ERROR TRACE:", traceback.format_exc())
            bot_reply = f"❌ Error: {str(e)}"
            st.session_state.chat_history.append(("bot", bot_reply))

    st.rerun()
