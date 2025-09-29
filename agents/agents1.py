import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.RAG_tool import ask_ai
import streamlit as st

# Import your existing tools
from tools.db_tool import agent_executor, query_sqlite_db
from tools.Fallback_tool import fallback_agent

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize memory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True
)
# Router function
def route_node(state):
    conversation = memory.load_memory_variables({}).get("history", "")
    
    prompt = PromptTemplate.from_template("""
    You are a routing agent for a medicine information system. Consider the conversation so far:
    {conversation}

    User just asked: {query}

   Decide the correct route:

            - "sql" → if the user asks about medicines, drugs, dosages, side effects, conditions, 
                top rated drugs, filtering, or numeric queries that can be answered 
                from the structured database/CSV/tables.

            - "rag" → if uploaded documents (PDF/TXT/DOCX) are available AND the query 
                could reasonably be answered from those documents. 
                This includes cases where the user does not explicitly mention 
                the file but asks questions like “summarize this”, 
                “what does it say about X”, or “explain section 2”. 
                In general, if the query relates to content that might exist 
                inside the uploaded documents, choose RAG.

            - "fallback" → only if neither the SQL tool nor the RAG tool can provide 
               a relevant answer to the query (e.g., pure chit-chat, 
               completely unrelated topics).

            Respond with only one: sql, rag, or fallback.
             """)
    
    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({
        "query": state["query"],
        "conversation": conversation
    })
    return {"decision": decision.strip().lower(), "query": state["query"],"conversation": conversation}

# Node functions
def sql_node(state):
    
    user_query = state.get("query")
    conversation = state.get("conversation", "")
    try:
        result =  query_sqlite_db(f"Conversation: {conversation}\nQuestion: {user_query}")
        return {"response": result}
    except Exception as e:
        return {"response": f"SQL Error: {e}"}

def rag_node(state):
    # Use the retriever from session state (from Streamlit app)
    retriever = None
    
    # Try to get retriever from different possible sources
    if "retriever" in st.session_state:
        retriever = st.session_state["retriever"]
    elif hasattr(state, 'get') and state.get("retriever"):
        retriever = state.get("retriever")
    
    if not retriever:
        return {"response": "No documents available. Please upload files first."}

    user_query = state.get("query")
    try:
        # Use your existing ask_ai function
        result = ask_ai(retriever, user_query)
        return {"response": result}
    except Exception as e:
        return {"response": f"RAG Error: {e}"}

def fallback_node(state):
    user_query = state.get("query")
    conversation=state.get("conversation","")
    try:
        result = fallback_agent(f"Conversation: {conversation}\nQuestion: {user_query}")
        return {"response": result}
    except Exception as e:
        return {"response": f"Fallback Error: {e}"}

# Router conditional function
def router(state):
    return state["decision"]

# Build the graph
graph = StateGraph(dict)
graph.add_node("router", route_node)
graph.add_node("sql", sql_node)
graph.add_node("rag", rag_node)
graph.add_node("fallback", fallback_node)

graph.set_entry_point("router")
graph.add_conditional_edges(
    "router",
    router,
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
    print("🤖 Medicine Query Bot is ready! Type 'exit' to quit.")
    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Update memory
        memory.save_context({"input": q}, {"output": ""})
        
        # Get response
        result = app.invoke({"query": q})
        print("Bot:", result["response"])
        
        # Update memory with response
        memory.save_context({"input": q}, {"output": result["response"]})