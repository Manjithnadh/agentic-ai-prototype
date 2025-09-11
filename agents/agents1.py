import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from tools.db_tool import agent_executor
from tools.RAG_tool import get_qa_chain
from tools.Fallback_tool import fallback_chain

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0,api_key=os.getenv("GOOGLE_API_KEY"))


memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history", 
    return_messages=True,
    max_token_limit=2000  # Prevents memory from growing too large
)

def get_chat_history_text():
    """Safely extract chat history as text with error handling."""
    try:
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        history_text = ""
        for msg in chat_history:
            if hasattr(msg, 'content'):
                history_text += f"{msg.content}\n"
            elif isinstance(msg, dict) and 'content' in msg:
                history_text += f"{msg['content']}\n"
            else:
                 history_text += f"{str(msg)}\n"
        
        return history_text.strip()
    except Exception as e:
        print(f"Memory error: {e}")
        return ""


qa_chain = None


def set_qa_chain(file_path):
    global qa_chain
    qa_chain = get_qa_chain(file_path=file_path)



def router(state: dict):
    """AI-driven router using Gemini instead of keyword lists."""
    query = state.get("query", "")

    routing_prompt = f"""
    You are a router that decides where to send a user query.
    
    Tools available:
    - "sql": for medicine, drugs, prescriptions, dosage, side effects, or health-related queries.
    - "rag": for queries about uploaded documents, PDFs, files, or text content.
    - "fallback": for general conversation or anything else.

    User query: {query}

    Answer with only one word: sql, rag, or fallback.
    """

    decision = llm.predict(routing_prompt).strip().lower()

    # sanitize
    if decision not in ["sql", "rag", "fallback"]:
        decision = "fallback"

    return decision


def route_node(state: dict):
    """Pass-through node so state stays a dict with 'query'."""
    return state


# -------------------- Agent Nodes --------------------
def sql_node(state):
    """SQL Agent node with enhanced error handling and context."""
    try:
        history_text = get_chat_history_text()
        prompt = f"Context from previous conversation:\n{history_text}\n\nCurrent query: {state['query']}"
        
        response = agent_executor.run({"input": prompt})
        
        memory.save_context({"input": state["query"]}, {"output": response})
        return {"response": response}
    
    except Exception as e:
        error_msg = f"Sorry, I encountered an error accessing the drug database: {str(e)}"
        return {"response": error_msg}


def rag_node(state):
    """RAG Agent node with better context handling."""
    if qa_chain is None:
        return {"response": "Please upload a document first using the upload feature."}
    
    try:
        history_text = get_chat_history_text()
        enhanced_query = f"Based on our conversation: {history_text}\n\nNow: {state['query']}"
        
        response = qa_chain.run(enhanced_query)
        memory.save_context({"input": state["query"]}, {"output": response})
        return {"response": response}
    
    except Exception as e:
        return {"response": f"Error processing your document query: {str(e)}"}

def fallback_node(state):
    """Fallback Agent node with context awareness."""
    try:
        history_text = get_chat_history_text()
        context = f"Conversation history:\n{history_text}\n\nCurrent query: {state['query']}"
        
        response = fallback_chain.run(query=context)
        memory.save_context({"input": state["query"]}, {"output": response})
        return {"response": response}
    
    except Exception as e:
        return {"response": "I apologize, I'm having trouble processing your request right now."}

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
