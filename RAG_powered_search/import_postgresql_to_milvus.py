import os
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from openai import OpenAI

load_dotenv()

POSTGRES_CONN = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", 5432),
    "user": os.getenv("POSTGRES_USER", "your_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
    "dbname": os.getenv("POSTGRES_DB_NAME", "your_db_name")
}
POSTGRES_QUERY = """
SELECT w.id, definition, "languageCode"
FROM "Word" w
JOIN "Language" l ON w."languageId" = l."id";
"""

ZILLIZ_URI = os.getenv('MILVUS_URI', 'tcp://localhost:19530')
ZILLIZ_TOKEN = os.getenv('MILVUS_TOKEN', 'your_milvus_token')
COLLECTION_NAME = "MAKNA_GLOSSARIUM"

USE_OPENAI = True
EMBEDDING_DIM = 1536 

def get_embedding(text):
    if USE_OPENAI:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding
    else:
         raise NotImplementedError("Local model not configured.")

client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

def setup_milvus():
    if COLLECTION_NAME in client.list_collections():
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return 

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="id",
        index_type="AUTOINDEX"
    )

    index_params.add_index(
        field_name="vector", 
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field(field_name="language_code", datatype=DataType.VARCHAR, max_length=64)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

    res = client.get_load_state(
        collection_name=COLLECTION_NAME
    )

    print(f"Collection '{COLLECTION_NAME}' created. with schema: {schema}")
    print(f"Result: {res}")

    return 

def fetch_postgres_data():
    conn = psycopg2.connect(**POSTGRES_CONN)
    cursor = conn.cursor()
    cursor.execute(POSTGRES_QUERY)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results 

def insert_to_milvus(query_results):
    data = []
    for row in tqdm(query_results, desc="Embedding & inserting"):
        id, content, language_code = row
        try:
            emb = get_embedding(content)
            data.append({
                "id": id,
                "content": content,
                "vector": emb,
                "language_code": language_code
            })
        except Exception as e:
            print(f"[ERROR] ID {id}: {e}")
            continue
    
    res = client.insert(
        collection_name=COLLECTION_NAME,
        data=data
    )
    print(f"Inserted {res} items into Milvus.")

if __name__ == "__main__":
    print("Connecting to Milvus...")
    setup_milvus()

    print("Fetching data from Postgres...")
    rows = fetch_postgres_data()

    print("Inserting data to Milvus...")
    insert_to_milvus(rows)
