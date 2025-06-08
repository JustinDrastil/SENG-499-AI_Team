from utils.chroma_manager import get_collection, add_documents, compute_hash

DOCUMENTS_DIR = "documents"

def load_and_split_documents(doc_folder):
    import os
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    chunk_data = []

    print(f"[Embedding] [Process Beginning]")
    for filename in os.listdir(doc_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(doc_folder, filename)
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()

            chunks = splitter.split_documents(docs)

            for idx, chunk in enumerate(chunks):
                chunk_data.append({
                    "text": chunk.page_content,
                    "source": filename,
                    "chunk_id": idx,
                    "hash": compute_hash(chunk.page_content)
                })

            print(f"[Embedding] {filename}: {len(chunks)} chunks.")

    print(f"[Embedding] [Process Completed]: {len(chunk_data)} total chunks.")
    return chunk_data

# Define your document chunks
chunk_data = load_and_split_documents(DOCUMENTS_DIR)
docs = [item["text"] for item in chunk_data]

# Generate unique ids and metadata using hash-based deduplication
ids = [item["hash"] for item in chunk_data]
metadatas = [{"source": item["source"], "hash": item["hash"]} for item in chunk_data]

# Initialize ChromaDB collection
collection = get_collection()

# Add documents (Chroma will skip if hash IDs already exist)
add_documents(collection, docs, metadatas, ids)

print("Documents (deduplicated) added to ChromaDB.")
