import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from agentic_ai_project.tools.RAG_tool import create_vectorstore, load_vectorstore

load_dotenv()

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
    # Example terminal chat
    file = "uploads/Frontend_jd.pdf"   # replace with your file
    qa = build_qa(file)

    print("RAG Agent ready. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa.run(query)
        print(f"AI: {answer}")
