import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def load_and_split(file_path):
    
    """Load file and split into chunks"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)


def create_vectorstore(file_path, db_path="faiss_index"):
    """Build FAISS index from file"""
    chunks = load_and_split(file_path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index for reuse
    vectorstore.save_local(db_path)
    return vectorstore


def load_vectorstore(db_path="faiss_index"):
    """Load saved FAISS index"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


def build_qa(file_path=None, db_path="faiss_index"):
    """Build QA chain with Gemini + FAISS"""
    # Use existing FAISS if available, else create new
    if file_path:
        vectorstore = create_vectorstore(file_path, db_path)
    else:
        vectorstore = load_vectorstore(db_path)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())


if __name__ == "__main__":
    # Build the QA chain
    qa_chain = build_qa(file_path="../data/Frontend_JD.pdf")  # or None to load existing FAISS
    
    print("RAG Chatbot ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        result = qa_chain({"query": query})
        print("Bot:", result["result"])

