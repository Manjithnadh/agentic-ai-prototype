from PythonProject23.agentic_ai_project.tools.db_tool import agent_executor
from PythonProject23.agentic_ai_project.tools.RAG_tool import get_qa_chain
from PythonProject23.agentic_ai_project.tools.Fallback_tool import fallback_chain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

# -------------------- Setup --------------------
load_dotenv(dotenv_path=r"C:\Users\Manjith.Mullapudi\PycharmProjects\agentic-ai-prototype\PythonProject23\agentic_ai_project\.env")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

qa_chain =None # will be set dynamically after file upload

def set_qa_chain(file_path):
    global qa_chain
    qa_chain = get_qa_chain(file_path=file_path)



# -------------------- Router --------------------
def router(state: dict):
    """Decide which agent to use based on query content."""
    query = state.get("query", "").lower()
    if any(word in query for word in ["drug", "medicine", "tablet", "pill"]):
        return "sql"
    elif any(word in query for word in ["resume", "job", "jd", "document"]):
        return "rag"
    else:
        return "fallback"

def route_node(state: dict):
    """Pass-through node so state stays a dict with 'query'."""
    return state

# -------------------- Agent Nodes --------------------
def sql_node(state):
    return {"response": agent_executor.run(state["query"])}

def rag_node(state):
    if qa_chain is None:
        return {"response": "❌ No document uploaded. Please upload a file first."}
    return {"response": qa_chain.run(state["query"])}

def fallback_node(state):
    return {"response": fallback_chain.run(query=state["query"])}

# -------------------- Build Graph --------------------
graph = StateGraph(dict)
graph.add_node("router", route_node)
graph.add_node("sql", sql_node)
graph.add_node("rag", rag_node)
graph.add_node("fallback", fallback_node)

graph.set_entry_point("router")
graph.add_conditional_edges(
    "router", router,
    {
        "sql": "sql",
        "rag": "rag",
        "fallback": "fallback",
    }
)

graph.add_edge("sql", END)
graph.add_edge("rag", END)
graph.add_edge("fallback", END)

app = graph.compile()

# -------------------- Run (CLI mode) --------------------
if __name__ == "__main__":
    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        result = app.invoke({"query": q})
        print("Bot:", result["response"])
