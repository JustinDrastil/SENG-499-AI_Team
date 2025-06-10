import chromadb
import hashlib
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def get_collection(name="rag_docs"):
    client = chromadb.PersistentClient(path="./chroma_store")
    return client.get_or_create_collection(name=name)

def add_documents(collection, docs, metadatas, ids, batch_size=5000):
    total = len(docs)
    embeddings = []
    print("[Add] [Process Beginning]")

    # Step 1: Compute embeddings
    for i, doc in enumerate(docs):
        emb = embedding_model.encode(doc, convert_to_numpy=True).tolist()
        embeddings.append(emb)
        if ((i + 1) % max(1, total // 10)) == 0 or ((i + 1) == total):
            percent = int(((i + 1) / total) * 100)
            print(f"[Add] Encoding progress: {percent}% ({i + 1}/{total})")

    # Step 2: Add in safe-size batches
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            documents=docs[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"[Add] Batch added: {i}–{end - 1}")

    print("[Add] [Process Completed] All documents added to collection.")

def delete_documents(collection, ids):
    for id in ids:
        collection.delete(id)

def compute_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def embed_query(text):
    return embedding_model.encode([text], convert_to_numpy=True)[0].tolist()
