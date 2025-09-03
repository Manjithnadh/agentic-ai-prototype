from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from PythonProject23.agentic_ai_project.tools.db_tool import query_sqlite_db
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.agent_toolkits import create_sql_agent
from langchain.tools import Tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from PythonProject23.agentic_ai_project.tools import db_tool
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"C:\Users\Manjith.Mullapudi\PycharmProjects\agentic-ai-prototype\.env")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

print("Loaded GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY")[:8], "...")

tools = query_sqlite_db



agent_executor=create_sql_agent(
    llm=llm,
    db = SQLDatabase.from_uri("sqlite:///drug_data.db"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


db_tool.agent_executor=agent_executor

query="what is best rating drugs"

a=agent_executor.run(query)
print(a)
