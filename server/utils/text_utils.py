from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import re

def compute_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def is_valid_url(text):
    # Basic check for a well-formed HTTP/HTTPS URL
    return re.match(r'^https?://[^\s]+$', text.strip()) is not None
    
def preprocess_text(text): 
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    text = re.sub(r'\[\s*[a-z]+\s*\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.replace('\x00', '').replace('\ufeff', '')
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def chunk_text(text, collection_name, doc_tag, min_chunk_chars=100):
    cleaned_text = preprocess_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    doc = Document(page_content=cleaned_text)
    chunks = splitter.split_documents([doc])

    chunk_data = []
    for idx, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if len(content) < min_chunk_chars:
            continue  # Skip short or noisy chunks

        chunk_data.append({
            "id": compute_hash(f"{collection_name}-{idx}-{content}"),
            "text": content,
            "metadata": {
                "chunk": idx,
                "collection": collection_name,
                "doc_tag": doc_tag
            }
        })
    return chunk_data
