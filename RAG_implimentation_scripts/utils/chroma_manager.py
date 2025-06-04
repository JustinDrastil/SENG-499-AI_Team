import chromadb
import hashlib
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def get_collection(name="rag_docs"):
    client = chromadb.PersistentClient(path="./chroma_store")
    return client.get_or_create_collection(name=name)

def add_documents(collection, docs, metadatas, ids):
    embeddings = embedding_model.encode(docs, convert_to_numpy=True).tolist()
    collection.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)

def delete_documents(collection, ids):
    for id in ids:
        collection.delete(id)

def compute_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def embed_query(text):
    return embedding_model.encode([text], convert_to_numpy=True)[0].tolist()
