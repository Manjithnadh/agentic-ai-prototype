import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings    
load_dotenv()

# Session management
class RAGSession:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
    
    def initialize(self, folder_path: str):
        if self.vectorstore is None:
            self.vectorstore = create_vectorstore(folder_path)
            self.qa_chain = build_qa(self.vectorstore)
    
    def query(self, question: str):
        if self.qa_chain is None:
            return "RAG not initialized. Upload documents first."
        return self.qa_chain.run(question)
    
    def show_embeddings(self, num=3):
        if self.vectorstore:
        # Get the actual embedding vectors
            index = self.vectorstore.index
            print(f"Embedding shape: {index.ntotal} vectors, {index.d} dimensions")
        # Show first few vectors
            for i in range(min(num, index.ntotal)):
                vector = index.reconstruct(i)
                print(f"Vector {i}: {vector[:5]}...")


rag_session = RAGSession()


def load_and_split(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def create_vectorstore(folder_path: str):
    all_chunks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith((".pdf", ".txt")):
            all_chunks.extend(load_and_split(file_path))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(all_chunks, embeddings)

def build_qa(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def query_documents(question: str, folder_path: str = "uploaded_files"):
    rag_session.initialize(folder_path)
    return rag_session.query(question)


if __name__ == "__main__":
    print("🤖 RAG System Test Mode")
    print("Make sure you have files in the 'uploaded_files' folder")

    rag_session.initialize("uploaded_files")
    rag_session.show_embeddings()  # Add this line
    
    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() in ['exit', 'quit']:
            break
        
        try:
            answer = query_documents(question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")