from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

model = SentenceTransformer('BAAI/bge-large-en-v1.5')
reranker = CrossEncoder('BAAI/bge-reranker-base')

chroma_client = chromadb.PersistentClient(path="./chroma_db")

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask server is running! Use /add, /delete, or /search endpoints with a 'collection_name' parameter."

def compute_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def chunk_text(text, collection_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    doc = Document(page_content=text)
    chunks = splitter.split_documents([doc])

    chunk_data = []
    for idx, chunk in enumerate(chunks):
        hash_input = f"{collection_name}-{idx}-{chunk.page_content}"
        chunk_data.append({
            "text": chunk.page_content,
            "chunk_id": idx,
            "hash": compute_hash(hash_input)
        })

    return chunk_data

@app.route('/add', methods=['POST'])
def add_data():
    data = request.json
    collection_name = data.get("collection_name")
    raw_text = data.get("text")

    if not collection_name or not raw_text:
        return jsonify({"error": "Missing collection_name or text"}), 400

    chunk_data = chunk_text(raw_text, collection_name)
    texts = [item["text"] for item in chunk_data]
    ids = [item["hash"] for item in chunk_data]
    metadatas = [{"chunk_id": item["chunk_id"], "hash": item["hash"]} for item in chunk_data]

    collection = chroma_client.get_or_create_collection(name=collection_name)
    embeddings = model.encode(texts).tolist()
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts,
        ids=ids
    )

    return jsonify({
        "message": "Data added successfully",
        "collection": collection_name,
        "count": len(texts)
    })

@app.route('/delete', methods=['POST'])
def delete_data():
    data = request.json
    collection_name = data.get("collection_name")
    metadata_filter = data.get("metadata_filter", {})
    
    if not collection_name:
        return jsonify({"error": "No collection_name provided"}), 400
    
    collection = chroma_client.get_collection(name=collection_name)
    collection.delete(where=metadata_filter)
    
    return jsonify({
        "message": "Data deleted successfully",
        "collection": collection_name
    })

@app.route('/search', methods=['POST'])
def search_data():
    data = request.json
    collection_name = data.get("collection_name")
    query_text = data.get("query_text", "")
    n_results = data.get("n_results", 2)
    top_k = min(1, n_results)
    
    if not collection_name:
        return jsonify({"error": "No collection_name provided"}), 400
    if not query_text:
        return jsonify({"error": "No query text provided"}), 400
    
    collection = chroma_client.get_collection(name=collection_name)
    query_embedding = model.encode(query_text).tolist()
    
    initial_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    pairs = [(query_text, doc) for doc in initial_results['documents'][0]]
    rerank_scores = reranker.predict(pairs)
    
    ranked_results = list(zip(
        initial_results['ids'][0],
        initial_results['documents'][0],
        initial_results['metadatas'][0],
        initial_results['distances'][0],
        rerank_scores
    ))
    
    ranked_results.sort(key=lambda x: x[4], reverse=True)
    top_results = ranked_results[:top_k]
    
    return jsonify({
        "query": query_text,
        "collection": collection_name,
        "results": {
            "ids": [x[0] for x in top_results],
            "documents": [x[1] for x in top_results],
            "metadatas": [x[2] for x in top_results],
            "embedding_distances": [float(x[3]) for x in top_results],
            "rerank_scores": [float(x[4]) for x in top_results]
        }
    })

@app.route('/collections', methods=['GET'])
def list_collections():
    try:
        collections = chroma_client.list_collections()
        collection_names = [col.name for col in collections]
        return jsonify({"collections": collection_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/view', methods=['POST'])
def view_collection():
    data = request.json
    collection_name = data.get("collection_name")
    limit = data.get("limit", 10)

    if not collection_name:
        return jsonify({"error": "No collection_name provided"}), 400

    try:
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.get(include=["documents", "metadatas", "ids"])
        documents = results["documents"][:limit]
        metadatas = results["metadatas"][:limit]
        ids = results["ids"][:limit]
        return jsonify({
            "collection": collection_name,
            "count": len(ids),
            "entries": [
                {"id": i, "document": d, "metadata": m}
                for i, d, m in zip(ids, documents, metadatas)
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
