from utils.chroma_manager import get_collection, add_documents, compute_hash

# Define your documents
docs = [
    "The ocean covers more than 70% of the Earth's surface.",
    "The deepest part of the ocean is the Mariana Trench, reaching nearly 11,000 meters.",
    "Oceans produce over half of the world's oxygen through marine plants.",
    "The Great Barrier Reef is the largest living structure on Earth.",
    "Over 80% of the ocean remains unexplored by humans."
]

metadatas = []
ids = []

# Generate unique ids and metadata using hash-based deduplication
for i, doc in enumerate(docs):
    doc_hash = compute_hash(doc)
    ids.append(doc_hash)
    metadatas.append({"source": "ocean_facts", "hash": doc_hash})

# Initialize ChromaDB collection
collection = get_collection()

# Add documents (Chroma will skip if hash IDs already exist)
add_documents(collection, docs, metadatas, ids)

print("Documents (deduplicated) added to ChromaDB.")
