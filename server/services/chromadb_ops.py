import chromadb
from services.embedding import embed_texts, rerank
from utils.text_utils import chunk_text, compute_hash

chroma_client = chromadb.PersistentClient(path="../database/chroma_store")

def add_documents(data):
    collection_name = data["collection_name"]
    text = data["text"]
    doc_tag = data["doc_tag"]

    collection = chroma_client.get_or_create_collection(name=collection_name)

    chunks = chunk_text(text, collection_name, doc_tag)
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
    doc_tag = data["doc_tag"]

    collection = chroma_client.get_collection(name=collection_name)
    metadata_filter = {"doc_tag": doc_tag}
    
    collection.delete(where=metadata_filter)

    return {"status": "deleted", "doc_tag": doc_tag}

def search_documents(data):
    collection_name = data["collection_name"] #dont need anymore
    query = data["query"]
    # chat hiostory
    # onc api token

    collection = chroma_client.get_collection(name=collection_name)

    query_embedding = embed_texts([query])
    results = collection.query(query_embeddings=query_embedding, n_results=100)
    docs = results["documents"][0]

    scores = rerank(query, docs)
    scores = [float(score) for score in scores]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_ranked = ranked[:10]

    return {"results": [{"text": doc, "score": score} for doc, score in top_ranked]} #returns data