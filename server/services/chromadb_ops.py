import chromadb
from services.embedding import embed_texts, rerank
from utils.text_utils import chunk_text, compute_hash

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def add_documents(data):
    collection_name = data["collection_name"]
    text = data["text"]
    collection = chroma_client.get_or_create_collection(name=collection_name)

    chunks = chunk_text(text, collection_name)
    embeddings = embed_texts([chunk["text"] for chunk in chunks])

    collection.add(
        documents=[chunk["text"] for chunk in chunks],
        ids=[chunk["id"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=embeddings
    )
    return {"status": "added", "count": len(chunks)}

def delete_documents(data):
    collection_name = data["collection_name"]
    ids = data["ids"]
    collection = chroma_client.get_or_create_collection(name=collection_name)
    collection.delete(ids=ids)
    return {"status": "deleted", "count": len(ids)}

def search_documents(data):
    collection_name = data["collection_name"]
    query = data["query"]
    collection = chroma_client.get_or_create_collection(name=collection_name)
    results = collection.query(query_texts=[query], n_results=5)
    docs = results["documents"][0]
    scores = rerank(query, docs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return {"results": [{"text": doc, "score": score} for doc, score in ranked]}
