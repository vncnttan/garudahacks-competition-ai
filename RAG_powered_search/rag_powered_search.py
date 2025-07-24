from langchain_google_genai import ChatGoogleGenerativeAI
from RAG_powered_search.utils import handle_user_query

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def llm_answer(query: str, language_src: str, language_dst: str = "id") -> str:
    message = handle_user_query(query, language_src, language_dst)
    result = llm.invoke(message)
    return result.content