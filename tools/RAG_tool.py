import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os
load_dotenv()

# -------------------- Load & Split Documents --------------------
def load_and_split(file_path: str):
    """Load a file (PDF or TXT) and split into chunks."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# -------------------- FAISS Vectorstore --------------------
def create_vectorstore(file_path: str, db_path: str = "faiss_index"):
    """Build FAISS index from file."""
    chunks = load_and_split(file_path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(db_path)  # Save index for reuse
    return vectorstore

def load_vectorstore(db_path: str = "faiss_index"):
    """Load saved FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# -------------------- Build QA Chain --------------------
def build_qa(file_path: str = None, db_path: str = "faiss_index"):
    """Build QA chain with Gemini + FAISS."""
    if file_path:
        vectorstore = create_vectorstore(file_path, db_path)
    else:
        vectorstore = load_vectorstore(db_path)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0,api_key=os.getenv("GOOGLE_API_KEY"))
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# -------------------- Exportable function --------------------
def get_qa_chain(file_path: str = None, db_path: str = "faiss_index"):
    """Return QA chain; uses existing index if no file provided."""
    return build_qa(file_path, db_path)
