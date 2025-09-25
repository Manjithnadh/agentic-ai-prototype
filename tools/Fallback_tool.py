from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.tools import tool
import os
load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


fallback_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a polite fallback assistant.

The user said: "{query}"

Rules:
- If the message is a greeting or wish (hi, hello, hey, good morning, good night, happy birthday, etc.), reply politely with a suitable greeting or wish back.
- If the message is not about medicines, drugs, greetings, or wishing, or RAG node always respond with:
  Im sorry, I can only answer queries related to medicines or drug information.
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
