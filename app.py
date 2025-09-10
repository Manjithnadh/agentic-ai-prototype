import os
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
import streamlit as st
from agents.agents1 import app as agent_app, set_qa_chain

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(
    page_title="Medical AI Assistant", 
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean chat interface
st.markdown("""
<style>
    .main {
        padding: 0;
    }
    .header-container {
        background: white;
        padding: 1rem 2rem;
        border-bottom: 1px solid #e0e0e0;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .chat-container {
        padding: 1rem 2rem;
        max-height: calc(100vh - 180px);
        overflow-y: auto;
        margin-top: 20px;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem 2rem;
        border-top: 1px solid #e0e0e0;
        z-index: 100;
    }
    .sidebar-open .input-container {
        left: 250px;
    }
    .sidebar-closed .input-container {
        left: 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 12px 0;
        margin-left: 20%;
        border: 1px solid #bbdefb;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 12px 0;
        margin-right: 20%;
        border: 1px solid #e0e0e0;
    }
    .message-content {
        margin: 0;
        line-height: 1.5;
        color: #333;
    }
    /* Remove default Streamlit padding */
    .stApp {
        padding-top: 0;
    }
    /* Ensure main content is properly positioned */
    .main .block-container {
        padding-top: 0;
        padding-bottom: 80px;
    }
    /* Welcome message styling */
    .welcome-container {
        text-align: center;
        padding: 3rem 1rem;
        color: #666;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- JavaScript to detect sidebar state --------------------
st.components.v1.html("""
<script>
function detectSidebarState() {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    const inputContainer = document.querySelector('.input-container');
    
    if (sidebar && inputContainer) {
        const isVisible = sidebar.offsetWidth > 0;
        if (isVisible) {
            inputContainer.classList.add('sidebar-open');
            inputContainer.classList.remove('sidebar-closed');
        } else {
            inputContainer.classList.add('sidebar-closed');
            inputContainer.classList.remove('sidebar-open');
        }
    }
}
setInterval(detectSidebarState, 100);
window.addEventListener('load', detectSidebarState);
</script>
""", height=0)

# -------------------- Sidebar for File Upload & Settings --------------------
with st.sidebar:
    st.markdown('### ⚙️ System Configuration')
    
    # File Upload for RAG
    st.markdown('#### 📄 Document Upload')
    uploaded_file = st.file_uploader(
        "Upload PDF/TXT for document queries", 
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        with st.spinner("Processing document..."):
            os.makedirs("uploaded_files", exist_ok=True)
            file_path = os.path.join("uploaded_files", uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            set_qa_chain(file_path)
        
        st.success(f"✅ **{uploaded_file.name}** uploaded successfully")
    
    st.markdown("---")
    
    # System information
    st.markdown('#### ℹ️ System Info')
    st.caption("""
    - **SQL Agent**: Medicine database queries
    - **RAG Agent**: Document-based questions  
    - **Fallback**: General responses
    """)
    
    # Clear chat button
    if st.button("🔄 Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

# -------------------- Main Content Area --------------------
# Header inside main container
st.markdown("""
<div class="header-container">
    <h2 style="margin:0; color:#2c3e50; font-size:1.5rem;">Medical AI Assistant</h2>
    <p style="margin:0; color:#7f8c8d; font-size:0.9rem;">Ask about medicines, documents, or general questions</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat messages container
chat_container = st.container()

with chat_container:
    # Display chat history
    if st.session_state.chat_history:
        for i, (sender, msg) in enumerate(st.session_state.chat_history):
            if sender == "user":
                st.markdown(f"""
                <div class="user-message">
                    <p class="message-content"><strong>👤 You:</strong> {msg}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <p class="message-content"><strong>🤖 Assistant:</strong> {msg}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        # Welcome message when no chat history
        st.markdown("""
        <div class="welcome-container">
            <h3>Welcome to Medical AI Assistant</h3>
            <p>Start a conversation by typing a message below</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Fixed Chat Input --------------------
user_input = st.chat_input("Type your message here...", key="chat_input")

if user_input:
    # Add user message to chat
    st.session_state.chat_history.append(("user", user_input))
    
    # Process the query
    with st.spinner("🤖 Processing your request..."):
        try:
            result = agent_app.invoke({"query": user_input})
            bot_reply = result["response"]
            
            # Add bot response to chat
            st.session_state.chat_history.append(("bot", bot_reply))
            
        except Exception as e:
            error_msg = "Sorry, I encountered an error processing your request. Please try again."
            st.session_state.chat_history.append(("bot", error_msg))
    
    # Rerun to update the chat display
    st.rerun()