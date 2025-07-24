from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from RAG_powered_search.utils import handle_user_query
from RAG_powered_search.import_postgresql_to_milvus import get_embedding
from langchain_milvus import Milvus
import os
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

def llm_search_from_milvus(query: str, language_src: str) -> str:
    results = vector_store.similarity_search(
        query=query,
        k=5,
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