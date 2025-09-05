import pandas as pd
import os
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentType
from dotenv import load_dotenv

load_dotenv()

# -------------------- Initialize LLM --------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
# -------------------- Load CSV into SQLite --------------------
def load_csv_to_sql(file_path: str, db_path: str = "drug_data.db"):
    """Load a CSV file into a SQLite database."""
    df = pd.read_csv(file_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("drugs", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

# Load your CSV (update path as needed)
load_csv_to_sql("data/Medicine_Details.csv")

# -------------------- Initialize SQL Agent --------------------
from langchain.prompts import PromptTemplate

sql_prompt = PromptTemplate(
    input_variables=["input", "table_info"],
    template="""
You are an expert SQL assistant. Use the following table info to answer the question.

Table info:
{table_info}

Question:
{input}

Output Format:
Always respond in valid JSON with the following keys:
- "query": the SQL query you would run
- "answer": the plain text answer for the user

Example:
{{"query": "SELECT name FROM drugs WHERE type='tablet';", "answer": "Paracetamol"}}
"""
)
agent_executor = create_sql_agent(
    llm=llm,
    db=SQLDatabase.from_uri("sqlite:///drug_data.db"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,

)
# -------------------- Optional helper --------------------
def query_sqlite_db(query: str) -> str:
    """Executes a query via the SQL agent."""
    try:
        return agent_executor.run(query)
    except Exception as e:
        return f"SQL Agent Error: {e}"
