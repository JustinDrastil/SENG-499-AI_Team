from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import os

DB_FOLDER = "./"
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

def load_reranker():
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return tokenizer, model

def rerank_documents(query, documents, tokenizer, model):
    pairs = [[query, doc] for doc in documents]
    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        scores = model(**inputs).logits.view(-1).float().cpu().numpy()
    return scores

def get_database_connection():
    try:
        client = PersistentClient(path=DB_FOLDER)
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu",
            normalize_embeddings=True
        )
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
        return collection
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        raise

def query_documents(collection, query_text, tokenizer, model, n_retrieve=3, n_return=1):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_retrieve
    )
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    rerank_scores = rerank_documents(query_text, documents, tokenizer, model)
    combined = []
    for i, (doc, meta, chroma_score, rerank_score) in enumerate(zip(
        documents,
        metadatas,
        results['distances'][0],
        rerank_scores
    ), 1):
        combined.append({
            "content": doc,
            "metadata": meta,
            "chroma_score": float(1 - chroma_score),
            "rerank_score": float(rerank_score)
        })
    combined.sort(key=lambda x: x["rerank_score"], reverse=True)
    return combined[:n_return]

def main():
    try:
        print("Loading reranker model...")
        reranker_tokenizer, reranker_model = load_reranker()
        collection = get_database_connection()
        print(f"\nConnected to database with {collection.count()} documents")
        print(f"Using embedding model: {EMBEDDING_MODEL}")
        print(f"Using reranker model: {RERANKER_MODEL}")
        print("Type 'exit' to quit\n")
        while True:
            query = input("Enter your search query: ").strip()
            if query.lower() in ('exit', 'quit'):
                break
            if not query:
                continue
            results = query_documents(collection, query, reranker_tokenizer, reranker_model)
            print(f"\nBest matching document for: '{query}'")
            for result in results:
                print(f"\n[Final Score: {result['rerank_score']:.4f}]")
                print(f"[Initial Retrieval Score: {result['chroma_score']:.2f}]")
                print(f"Source: {result['metadata'].get('source', 'N/A')}")
                print(f"Content: {result['content']}")
            print()
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()