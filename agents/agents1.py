import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.RAG_tool import build_qa, rag_session
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

            You are a routing AI deciding which tool to use. The available tools are:

            1. Tool Name: Medicine Database Query Tool
               Description: Contains detailed data about medicines, including:
               - Medicine Name
               - Composition
               - Uses
               - Side effects
               - Image URL
               - Manufacturer
               - Excellent Review %, Average Review %, Poor Review %
               Rules: Use this tool ONLY if the user's query is specifically about medicines, their uses, side effects, composition, manufacturer, or ratings.
            
            2. Tool Name: RAG Document Query Tool
               Description: Retrieves answers from uploaded PDFs or text documents using a vectorstore. Can answer detailed questions about content in these documents.
               Rules: Use this tool for queries about information in uploaded documents or general knowledge provided in your RAG system.
            
            3. Tool Name: Fallback Agent
               Description: Handles queries unrelated to medicines or uploaded documents. Gives polite responses to greetings or out-of-scope questions.
               Rules: Activate this tool ONLY if the query is unrelated to medicines or uploaded documents.
            
            
          

            Respond with only one: sql, rag, or fallback.
             """)

    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({
        "query": state["query"],
        "conversation": conversation
    })
    return {"decision": decision.strip().lower(), "query": state["query"], "conversation": conversation}


# Node functions
def sql_node(state):
    user_query = state.get("query")
    conversation = state.get("conversation", "")
    try:
        result = query_sqlite_db(f"Conversation: {conversation}\nQuestion: {user_query}")
        return {"response": result}
    except Exception as e:
        return {"response": f"SQL Error: {e}"}


def rag_node(state):
    # Check if we have a QA chain
    qa = st.session_state.get("qa_chain") or rag_session.qa_chain

    if not qa:
        # Try to initialize RAG if vectorstore exists but QA chain is missing
        if rag_session.vectorstore and not rag_session.qa_chain:
            qa = build_qa(rag_session.vectorstore)
            rag_session.qa_chain = qa
            st.session_state["qa_chain"] = qa

    if not qa:
        return {"response": "No documents available. Please upload files first."}

    user_query = state.get("query")
    try:
        result = qa.invoke({"query": user_query})
        if isinstance(result, dict):
            response = result.get("result") or result.get("output") or str(result)
        else:
            response = str(result)
        return {"response": response}
    except Exception as e:
        return {"response": f"RAG Error: {e}"}


def fallback_node(state):
    user_query = state.get("query")
    conversation = state.get("conversation", "")
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