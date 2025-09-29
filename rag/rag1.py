import os
import shutil
from dotenv import load_dotenv
import PyPDF2
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

def load_and_split(file):
    docs = []

    # Case 1: Streamlit UploadedFile (file-like object)
    if hasattr(file, "read"):
        if file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    docs.append(Document(page_content=text, metadata={"page": page_num}))
        else:  # text file uploaded
            text = file.read().decode("utf-8")
            docs.append(Document(page_content=text, metadata={"source": file.name}))

    # Case 2: File path on disk
    else:
        if file.endswith(".pdf"):
            with open(file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        docs.append(Document(page_content=text, metadata={"page": page_num}))
        else:  # text file path
            loader = TextLoader(file, encoding="utf-8")
            docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def create_faiss(files):
    all_chunks=[]
    for file in files:
        all_chunks.extend(load_and_split(file))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def ask_ai(retriever, question):
    llm = ChatOpenAI(
        model = "gpt-4o-mini",
        openai_api_key = os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.run(question)
    return answer




    

