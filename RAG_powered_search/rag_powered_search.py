from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from utils import handle_user_query

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

message = handle_user_query("jv", "Ora Iso")
result = llm.invoke(message)
print(result.content)