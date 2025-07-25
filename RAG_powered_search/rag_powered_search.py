from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from RAG_powered_search.utils import handle_user_query
from RAG_powered_search.import_postgresql_to_milvus import get_embedding
from langchain_milvus import Milvus
import os
from constants import translation_lang
import httpx
from urllib.parse import quote
import cohere

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
COLLECTION_NAME = "MAKNA_GLOSSARIUM"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": os.getenv('MILVUS_URI', 'tcp://localhost:19530'), "token": os.getenv('MILVUS_TOKEN', '')},
    collection_name=COLLECTION_NAME,
    text_field="content",
    index_params={
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
        "json_cast_type": "array_double"
    }
)

def llm_answer(query: str, language_src: str, language_dst: str = "id") -> str:
    message = handle_user_query(query, language_src, language_dst)
    result = llm.invoke(message)
    return result.content


co = cohere.Client()

async def llm_search_from_milvus(query: str, language_dst: str, language_src: str = "id") -> str:
    search_kwargs = {"k": 5}
    if language_dst:
        search_kwargs["filter"] = {"language_code": language_dst}

    if language_src != "id":
        try:
            encoded_text = quote(query)
            sl_code = translation_lang.get(language_src, {}).get('sl_code', 'id')
            url = f"https://ftapi.pythonanywhere.com/translate?sl={sl_code}&dl=id&text={encoded_text}"

            async with httpx.AsyncClient() as client:
                translation_api_response = await client.get(url, timeout=20.0)
                translation_api_response.raise_for_status()
                data = translation_api_response.json()
                query = data.get('destination-text', 'Translation failed.')

        except httpx.RequestError as exc:
            print(f"An error occurred while requesting translation: {exc}")
        except Exception as e:
            print(f"An error occurred during translation processing: {e}")

    print("Search Query:", query)
    results = vector_store.similarity_search(
        query=query,
        **search_kwargs
    )

    docs = []
    for res in results:
        docs.append(res.page_content)

    rerank_results = co.rerank(model="rerank-v3.5", query=query, documents=docs, top_n=5)

    formatted_results = []
    for rr in rerank_results.results:
        index = rr.index
        score = rr.relevance_score
        content = results[index].page_content
        id = results[index].metadata.get("id", "unknown")
        formatted_results.append({
            "content": content,
            "id": id,
            "score": score
        })
        
    return formatted_results