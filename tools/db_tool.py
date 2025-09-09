import pandas as pd
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentType
import os
from dotenv import load_dotenv
load_dotenv()

DB_PATH = "drug_data.db"
TABLE_NAME = "drugs"
CSV_PATH = "data\Medicine_Details.csv" 

# -------------------- Initialize LLM --------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------- Load CSV into SQLite --------------------
def load_csv_to_sql(file_path: str, db_path: str = "drug_data.db"):
    """Load a CSV file into a SQLite database."""
    df = pd.read_csv(file_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("drugs", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def database_exists(db_path: str, table_name: str) -> bool:
    """Check if the database file and table already exist."""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    except:
        return False
    
if not database_exists(DB_PATH, TABLE_NAME):
    print("Database not found. Creating new database from CSV...")
    load_csv_to_sql(CSV_PATH, DB_PATH)
else:
    print("Database already exists. Skipping CSV load.")

# -------------------- Initialize SQL Database --------------------
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# -------------------- Initialize SQL Agent --------------------
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,  # This is already set, but keep it
    max_iterations=10,
    early_stopping_method="force",
)

# -------------------- Enhanced helper with better error handling --------------------
def query_sqlite_db(user_query: str) -> str:
    """Executes a query via the SQL agent and returns a user-friendly response."""
    try:
        result = agent_executor.run(user_query)
        return result
    except Exception as e:
        # Check if it's an output parsing error
        error_str = str(e)
        if "Could not parse LLM output" in error_str:
            # Extract the actual useful response from the error message
            if "This is the error:" in error_str:
                # Extract the part after "This is the error:"
                useful_output = error_str.split("This is the error:")[1].strip()
                # Remove the backticks if present
                useful_output = useful_output.replace("`", "")
                return useful_output
            else:
                return "I found some information, but had trouble formatting it. Please try asking more specifically."
        elif "ValueError" in error_str:
            return "I couldn't process that question. Could you try rephrasing it?"
        else:
            return f"An unexpected error occurred: {error_str}"

# -------------------- Test the agent --------------------
if __name__ == "__main__":
    test_query = "What are the uses of some common drugs?"
    print(f"Query: {test_query}")
    result = query_sqlite_db(test_query)
    print(f"Result: {result}")