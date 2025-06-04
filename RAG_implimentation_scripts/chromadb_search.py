from utils.chroma_manager import get_collection, embed_query
from utils.llama3_inference import run_llama3
from utils.rerank import rerank_documents

query = input("Enter your query: ")

# Get collection and search
collection = get_collection()
query_embedding = embed_query(query)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    include=["documents", "metadatas"]
)

# Prepare documents for re-ranking
docs = results["documents"][0] if results["documents"] else []
reranked_docs = rerank_documents(query, docs, top_k=5)
context = "\n\n".join(reranked_docs)

# Use LLaMA 3 to generate an answer
answer = run_llama3(query, context)

print("\n--- Answer ---\n")
print(answer)
