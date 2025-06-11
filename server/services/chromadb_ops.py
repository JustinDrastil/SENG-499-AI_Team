import chromadb
from services.embedding import embed_texts, rerank
from utils.text_utils import chunk_text, compute_hash
from services.llm import generate_response

chroma_client = chromadb.PersistentClient(path="../database/chroma_store")

def add_documents(data):
    collection_name = data["collection_name"]
    text = data["text"]
    doc_tag = data["doc_tag"]

    collection = chroma_client.get_or_create_collection(name=collection_name)

    chunks = chunk_text(text, collection_name, doc_tag)
    embeddings = embed_texts([chunk["text"] for chunk in chunks], batch_size=8)

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

    context_parts = []
    for i, (doc, _) in enumerate(top_ranked):
        context_parts.append(f"Source {i+1}:\n{doc.strip()}")
    context_text = "\n\n".join(context_parts)

    ai_response = generate_response(context_text, query, model_key="api")

    return {
        "answer": ai_response
    }