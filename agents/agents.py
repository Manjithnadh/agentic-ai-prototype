import os
import os
from dotenv import load_dotenv
load_dotenv()
google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from tools.db_tool import agent_executor
from tools.RAG_tool import get_qa_chain
from tools.Fallback_tool import fallback_chain




llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


qa_chain = None


def set_qa_chain(file_path):
    global qa_chain
    qa_chain = get_qa_chain(file_path=file_path)



def router(state: dict):
    """Decide which agent to use based on query content."""
    query = state.get("query", "").lower()

    if any(word in query for word in ["drug", "medicine", "tablet", "pill"]):
        return "sql"

    if qa_chain is not None:
        return "rag"
    else:
        return "fallback"


def route_node(state: dict):
    """Pass-through node so state stays a dict with 'query'."""
    return state


# -------------------- Agent Nodes --------------------
def sql_node(state):
    """SQL Agent node with conversational memory."""
    # Get chat history safely
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    # Convert chat history to plain text
    history_text = ""
    for msg in chat_history:
        if isinstance(msg, dict):
            # Some memory stores messages as dicts with 'content'
            history_text += f"{msg.get('content', '')}\n"
        elif hasattr(msg, "content"):
            history_text += f"{msg.content}\n"
        else:
            history_text += f"{str(msg)}\n"

    # Combine user query with history
    prompt = state["query"] + "\nPrevious conversation:\n" + history_text

    # Run agent
    response = agent_executor.run(prompt)

    # Save current context in memory
    memory.save_context({"input": state["query"]}, {"output": response})

    return {"response": response}


def rag_node(state):
    """RAG Agent node with conversational memory."""
    if qa_chain is None:
        return {"response": "❌ No document uploaded. Please upload a file first."}

    chat_history = memory.load_memory_variables({})["chat_history"]
    history_text = "\n".join([msg.content if hasattr(msg, "content") else str(msg) for msg in chat_history])

    prompt = state["query"] + "\nPrevious conversation:\n" + history_text
    response = qa_chain.run(prompt)
    memory.save_context({"input": state["query"]}, {"output": response})
    return {"response": response}


def fallback_node(state):
    """Fallback Agent node with conversational memory."""
    chat_history = memory.load_memory_variables({})["chat_history"]
    history_text = "\n".join([msg.content if hasattr(msg, "content") else str(msg) for msg in chat_history])

    prompt = state["query"] + "\nPrevious conversation:\n" + history_text
    response = fallback_chain.run(query=prompt)
    memory.save_context({"input": state["query"]}, {"output": response})
    return {"response": response}


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
    print("🤖 Bot is ready! Type 'exit' to quit.")
    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        result = app.invoke({"query": q})
        print("Bot:", result["response"])
