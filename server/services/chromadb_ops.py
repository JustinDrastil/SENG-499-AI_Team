import chromadb
import torch
import json
from services.embedding import embed_texts
from utils.text_utils import chunk_text, is_valid_url, has_location_entity
from utils.json_utils import compress_onc_json_response
from utils.time_utils import extract_timeframe_range
from services.llm import generate_response, build_second_llm_prompt, check_prompt_length
from services.fetch_onc_data import fetch_onc_data

chroma_client = chromadb.PersistentClient(path="../database/chroma_store")

# /add endpoint function, adds one document to the specified collection
def add_document(data):
    # required fields for adding a document
    collection_name = data["collection_name"]
    document_name = data["document_name"]
    text = data["text"]

    # segments document into chunks, batch-embeds chunks into vectors for chromadb
    chunks = chunk_text(collection_name, document_name, text)
    embeddings = embed_texts([chunk["text"] for chunk in chunks], batch_size=8)

    # will create new collection if it does not exist
    collection = chroma_client.get_or_create_collection(name=collection_name)

    collection.add(
        documents=[chunk["text"] for chunk in chunks],
        ids=[chunk["id"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=embeddings
    )
    return {"status": "added", "count": len(chunks)}

# /delete endpoint function, deletes document from the specified collection
def delete_document(data):

    # required fields for deleting a document
    collection_name = data["collection_name"]
    document_name = data["document_name"]

    # ensure collection exists
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        return {"error": f"Collection '{collection_name}' not found: {str(e)}"}

    metadata_filter = {"document_name": document_name}
    
    collection.delete(where=metadata_filter)

    return {"status": "deleted", "document_name": document_name}

# /search endpoint function, performs RAG on user query and returns an answer of type either 0,1,2
def search_documents(data):

    # required fields for /search
    collection_name = data["collection_name"]
    query = data["query"]
    message_history = data.get("message_history")
    token = data["token"]

    start_time, end_time = extract_timeframe_range(query)
    if start_time and end_time:
        query += f" from {start_time} to {end_time}"

    if "cambridge bay" not in query.lower() and not has_location_entity(query):
        query += " in Cambridge Bay"

    # ensure collection exists
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        return {"error": f"Collection '{collection_name}' not found: {str(e)}"}

    query_embedding = embed_texts([query])
    results = collection.query(query_embeddings=query_embedding, n_results=200)
    docs = results["documents"][0]

    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"Source {i+1}:\n{doc.strip()}")
    context_text = "\n\n".join(context_parts)

    ai_response = generate_response(context_text, query, "api", token, message_history)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    clean_response = ai_response.strip("`\n")
    start_index = clean_response.find("http")
    clean = clean_response[start_index:]
    clean = clean + token


    if is_valid_url(clean):
        # Step 1: Get JSON from ONC
        api_json = fetch_onc_data(clean)

        # Step 2: Format prompt and use second LLM to generate final response
        second_prompt = build_second_llm_prompt(data["query"], json.dumps(api_json, indent=2))

        # Step 3: Check if the prompt is too long
        if(check_prompt_length(second_prompt)):
            try:
                # Step 3.1: Compress the JSON
                api_json = compress_onc_json_response(api_json)
                # Step 3.2: Rebuild prompt with compressed JSON
                second_prompt = build_second_llm_prompt(data["query"], json.dumps(api_json, indent=2))
                # return { "answer": f"You can find the data using the following link: {clean}", "type": 2}
            except Exception as e:
                return {"error": f"Failed to compress ONC data: {str(e)}"}
            
        final_answer = generate_response(second_prompt, query=data["query"], model_key="answer")

        no_data_keywords = ["no information found", "could not find", "not available", "no data"]
        if any(kw in final_answer.lower() for kw in no_data_keywords):
            # valid call, but ONC has no data 
            return { "answer": final_answer, "type": 0 }
        else:
            # successful full answer
            return { "answer": final_answer, "type": 2}
    else:
        # Detect clarification questions
        clarification_keywords = ["please specify", "could refer to", "which one", "ambiguous", "what kind", "do you want to"]
        if any(kw in clean_response.lower() for kw in clarification_keywords):
            # clarification question
            return { "answer": clean_response, "type": 1 }
        else:
            # invalid api call
            return { "answer": clean_response, "type": 0 }

# /collections endpoint function, returns list of collections in chromadb
def list_collections():
    collections = chroma_client.list_collections()
    return {
        "Collection Names": [col.name for col in collections],
        "Number of Collections": len(collections)
    }