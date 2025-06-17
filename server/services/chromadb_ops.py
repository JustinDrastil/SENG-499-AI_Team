import chromadb
import torch
import json
from services.embedding import embed_texts, rerank
from utils.text_utils import chunk_text, compute_hash, is_valid_url
from services.llm import generate_response, build_second_llm_prompt
from services.fetch_onc_data import fetch_onc_data

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
    results = collection.query(query_embeddings=query_embedding, n_results=50)
    docs = results["documents"][0]

    scores = rerank(query, docs)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    scores = [float(score) for score in scores]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_ranked = ranked[:10]

    context_parts = []
    for i, (doc, _) in enumerate(top_ranked):
        context_parts.append(f"Source {i+1}:\n{doc.strip()}")
    context_text = "\n\n".join(context_parts)

    ai_response = generate_response(context_text, query, model_key="api", token="5e3aec6d-8ed0-49bc-9e96-7980704c17ef")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # print(ai_response)
    clean_response = ai_response.strip("`\n")
    start_index = clean_response.find("http")
    clean = clean_response[start_index:]
    # print(clean)

    if is_valid_url(clean):
        # Step 1: Get JSON from ONC
        api_json = fetch_onc_data(clean)
        # Step 2: Format prompt and use second LLM to generate final response
        second_prompt = build_second_llm_prompt(data["query"], json.dumps(api_json, indent=2))
        final_answer = generate_response(second_prompt, query=data["query"], model_key="answer")
        return { "answer": final_answer }
    else:
        return { "answer": clean_response }