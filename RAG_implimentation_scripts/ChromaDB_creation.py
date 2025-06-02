from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os
DB_FOLDER = "./"

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5",
    normalize_embeddings=True
)

client = PersistentClient(path=DB_FOLDER)

collection = client.get_or_create_collection(
    name="my_documents",
    embedding_function=embedding_func
)

collection.add(
    documents=[
        "The ocean covers more than 70% of the Earth's surface.",
        "The deepest part of the ocean is the Mariana Trench, reaching nearly 11,000 meters.",
        "Oceans produce over half of the world's oxygen through marine plants.",
        "The Great Barrier Reef is the largest living structure on Earth.",
        "Over 80% of the ocean remains unexplored by humans."
    ],
    metadatas=[
        {"source": "ocean_facts"},
        {"source": "ocean_facts"},
        {"source": "ocean_facts"},
        {"source": "ocean_facts"},
        {"source": "ocean_facts"}
    ],
    ids=["fact1", "fact2", "fact3", "fact4", "fact5"]
)

print(f"ChromaDB successfully created in {DB_FOLDER} with {collection.count()} documents")
print("Using embedding model: BAAI/bge-large-en-v1.5")