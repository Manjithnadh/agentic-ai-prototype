import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings    
load_dotenv()


# -------------Load & Split Documents ---------------------------------------->

def load_and_split(file_path: str):
    """Load a file (PDF or TXT) and split into chunks."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def create_vectorstore(folder_path: str):
    all_chunks= []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith((".pdf",".txt")):
            all_chunks.extend(load_and_split(file_path))
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(all_chunks,embeddings)

def build_qa(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature= 0, api_key=os.getenv("OPENAI_API_KEY"))
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    return RetrievalQA.from_chain_type(llm = llm, retriever=retriever)

# if __name__ == "__main__":
#     folder = "uploaded_files" 
#     vectorstore = create_vectorstore(folder)
#     qa = build_qa(vectorstore)

#     print("Chat ready! Type 'exit' to quit.")
#     while True:
#         q = input("Ask: ")
#         if q.lower() in ["exit", "quit"]:
#             break
#         print(qa.run(q))
