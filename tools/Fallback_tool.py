import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool


# -------------------- Load ENV + Initialize LLM --------------------
load_dotenv(dotenv_path=r"C:\Users\Manjith.Mullapudi\PycharmProjects\agentic-ai-prototype\PythonProject23\agentic_ai_project\.env")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# -------------------- Fallback Agent --------------------
fallback_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a polite assistant. 
The user asked: "{query}"

If the query is not related to medicines, drugs, or knowledge from the RAG agent,
respond with: "I’m sorry, I can only answer queries related to medicines or drug information."
"""
)

fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt)


@tool("fallback_agent", return_direct=True)
def fallback_agent(query: str) -> str:
    """Handles unrelated queries with a polite response."""
    try:
        return fallback_chain.run(query=query)
    except Exception as e:
        return f"Fallback Agent Error: {e}"


# -------------------- Example Run --------------------
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk me anything (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        response = fallback_agent(user_input)
        print("\nAssistant:", response)
