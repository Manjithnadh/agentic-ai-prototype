import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)
from langchain.tools import tool
# Load environment
load_dotenv()

DB_PATH = "drug_data.db"
TABLE_NAME = "drugs"
CSV_PATH = "data/Medicine_Details.csv"

# ------------------ DB Setup Functions ------------------

def load_csv_to_sql(file_path: str, db_path: str = DB_PATH):
    """Load CSV into SQLite database."""
    df = pd.read_csv(file_path)
    conn = sqlite3.connect(db_path)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Loaded data from {file_path} to {db_path}")

def database_exists(db_path: str, table_name: str) -> bool:
    """Check if DB + table exist."""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (table_name,)
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    except Exception:
        return False



# Create DB if not exists
if not database_exists(DB_PATH, TABLE_NAME):
    print("Database not found. Creating new database from CSV...")
    load_csv_to_sql(CSV_PATH, DB_PATH)
else:
    print("Database already exists. Skipping CSV load.")



llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ------------------ SQL Database Tools Setup ------------------

# Initialize SQL Database
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}",sample_rows_in_table_info=0,include_tables=[TABLE_NAME])

# Create individual tools
query_tool = QuerySQLDatabaseTool(db=db)
info_tool = InfoSQLDatabaseTool(db=db)
list_tool = ListSQLDatabaseTool(db=db)
query_checker_tool = QuerySQLCheckerTool(db=db, llm=llm)

# Create toolkit
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm,
)

# Get all tools from the toolkit
tools = toolkit.get_tools()

# ------------------ Agent Setup ------------------

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate",
)

# ------------------ Query Function ------------------
@tool("sql_tool", return_direct=True)
def query_sqlite_db(user_query: str) -> str:
    """
    Query the structured medicine database (Medicine_Details.csv loaded into SQLite).

    The database contains the following fields:
    - Medicine Name
    - Composition
    - Uses
    - Side_effects
    - Image URL
    - Manufacturer
    - Excellent Review %
    - Average Review %
    - Poor Review %

    This tool converts natural language into SQL queries on the medicines table and retrieves results.
    """
    try:
        result = agent_executor.invoke({"input": user_query})
        return result.get("output", "No output received from agent")
    except Exception as e:
        return f"Error: {str(e)}"



