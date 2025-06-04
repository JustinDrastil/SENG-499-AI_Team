import os
import ollama
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_DB_DIR = "chroma_db"
DOCUMENTS_DIR = "documents"
COLLECTION_NAME = "rag_collection"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "tinyllama"
TOP_K = 6

def load_and_split_documents(doc_folder):
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    all_chunk_groups = []

    for filename in os.listdir(doc_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(doc_folder, filename)
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()

            chunks = splitter.split_documents(docs)
            chunk_texts = [chunk.page_content for chunk in chunks]
            all_chunk_groups.append(chunk_texts)

            print(f"[Embedding] {filename}: {len(chunk_texts)} chunks.")

    print(f"[Embedding] Loaded {len(all_chunk_groups)} document(s).")
    return all_chunk_groups



def normalize(text):
    return text.lower().strip()

def get_embedding(text: str):
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return []

def embed_documents(doc_folder: str = DOCUMENTS_DIR):
    print("[Embedding] Loading and splitting documents...")
    all_chunk_groups = load_and_split_documents(doc_folder)
    if not all_chunk_groups:
        print("[Embedding] No chunks found.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # Force fresh collection each time
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    chunk_id = 0
    for doc_idx, chunk_group in enumerate(all_chunk_groups):
        chunk_id = 0
        for chunk in chunk_group:
            embedding = get_embedding(chunk)
            if embedding:
                collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"doc{doc_idx}_chunk{chunk_id}"],
                metadatas=[{"doc_id": doc_idx, "chunk_id": chunk_id}])
                print(f"[Embedding] Added doc {doc_idx} chunk {chunk_id}")
                chunk_id += 1

    print("[Embedding] All documents embedded and stored.")


def retrieve_similar_chunks(query: str, top_k: int = TOP_K):
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return list(zip(results["documents"][0], results["metadatas"][0]))


def ask_llm(query: str, context: str):
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question. If the context is not helpful, still try your best.

Context:
{context}

Question: {query}

Answer:"""
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

def main():
    print("Running embedding step...")
    embed_documents()

    while True:
        user_input = input("\nAsk a question (type 'exit' to quit):\n\nYou: ").strip()
        if user_input.lower() == 'exit':
            break

        print(f"[INFO] Query: {user_input}")
        retrieved_chunks = retrieve_similar_chunks(normalize(user_input))
        context = "\n\n".join(chunk for chunk, _ in retrieved_chunks)
        response = ask_llm(user_input, context)
        print("\nContext:")
        for i, (chunk, metadata) in enumerate(retrieved_chunks):
            print(f"DOC{metadata['doc_id']}_CHUNK{metadata['chunk_id']}: \n{chunk}\n")
        print(f"\nAnswer:\n{response}\n")

if __name__ == "__main__":
    main()
