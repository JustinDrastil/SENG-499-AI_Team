from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

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
        chunk_data.append({
            "id": compute_hash(f"{collection_name}-{idx}-{chunk.page_content}"),
            "text": chunk.page_content,
            "metadata": {"chunk": idx, "collection": collection_name}
        })
    return chunk_data
