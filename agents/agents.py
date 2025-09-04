
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv(dotenv_path=r"C:\Users\Manjith.Mullapudi\PycharmProjects\agentic-ai-prototype\PythonProject23\agentic_ai_project\.env")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)