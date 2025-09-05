from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# -------------------- Initialize LLM --------------------
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


def fallback_agent(query: str) -> str:
    """Handles unrelated queries with a polite response."""
    try:
        return fallback_chain.run(query=query)
    except Exception as e:
        return f"Fallback Agent Error: {e}"
