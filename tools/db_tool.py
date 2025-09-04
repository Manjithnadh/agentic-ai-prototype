import pandas as pd
import sqlite3

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import  AgentType

from langchain_community.agent_toolkits import create_sql_agent

from langchain_community.utilities import SQLDatabase


from dotenv import load_dotenv
import os
from langchain.tools import tool

load_dotenv(dotenv_path=r"C:\Users\Manjith.Mullapudi\PycharmProjects\agentic-ai-prototype\PythonProject23\agentic_ai_project\.env")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)





def load_csv_to_sql():
    """Load a CSV file into a SQLite database from a given file path."""
    file_path=r"C:\Users\Manjith.Mullapudi\PycharmProjects\agentic-ai-prototype\PythonProject23\agentic_ai_project\data\Medicine_Details.csv"


    df=pd.read_csv(file_path)

    conn=sqlite3.connect('drug_data.db')

    df.to_sql('drugs',conn,if_exists="replace",index=False)

    conn.commit()
    conn.close()

load_csv_to_sql()

agent_executor=create_sql_agent(
    llm=llm,
    db = SQLDatabase.from_uri("sqlite:///drug_data.db"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


@tool("query_sqlite_db", return_direct=True)
def query_sqlite_db(query: str) -> str:
    """Converts natural language into an SQL query, executes it, and returns results from the SQLite DB."""
    if agent_executor is None:
        return "Error: SQL Agent is not initialized."
    try:
        return agent_executor.run(query)
    except Exception as e:
        return f"SQL Agent Error: {e}"

query="what is the top 5 best rated drugs"

a=agent_executor.run(query)

print(a)