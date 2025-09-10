import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

DB_PATH = "drug_data.db"
TABLE_NAME = "drugs"
CSV_PATH = "data\Medicine_Details.csv" 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

def load_csv_to_sql(file_path: str, db_path: str = "drug_data.db"):
    df = pd.read_csv(file_path)
    conn = sqlite3.connect(db_path)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def database_exists(db_path: str, table_name: str) -> bool:
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


db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

response_schemas = [
    ResponseSchema(name="sql_query", description="The SQL query to run against the database"),
    ResponseSchema(name="answer", description="Final user-friendly answer to the query"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

CUSTOM_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful assistant that translates natural language questions into SQL queries 
     for a SQLite database. 

     Always respond ONLY in JSON with the following keys:
     - sql_query: the SQL query to run
     - answer: a clear and concise natural language answer based on the query results

     {format_instructions}
     """),
    ("human", "{input}")
])

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool


db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

sql_tool = QuerySQLDatabaseTool(db=db)

agent_executor = initialize_agent(
    tools=[sql_tool],  
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prompt=CUSTOM_SQL_PROMPT,
    handle_parsing_errors=True,
)


def query_sqlite_db(user_query: str) -> str:
    try:
        result = agent_executor.run(user_query)
        parsed = output_parser.parse(result)
        sql = parsed.get("sql_query", "")
        ans = parsed.get("answer", "")
        return f"SQL: {sql}\nAnswer: {ans}"
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    test_query = "What are the uses of some common drugs?"
    print(f"Query: {test_query}")
    result = query_sqlite_db(test_query)
    print(f"Result:\n{result}")
