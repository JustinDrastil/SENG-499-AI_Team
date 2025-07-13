from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import re

def compute_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def has_location_entity(text):
    location_keywords = ["bay", "inlet", "strait", "cambridge", "saanich", "pacific", "atlantic"]
    return any(re.search(rf"\b{kw}\b", text.lower()) for kw in location_keywords)

def is_valid_url(text):
    # Basic check for a well-formed HTTP/HTTPS URL
    return re.match(r'^https?://[^\s]+$', text.strip()) is not None

 # Determine intent based on simple keyword heuristic
def is_general_question(query):
    keywords = [
        "what is", "why is", "explain", "how does", "define",
        "describe", "importance of", "what does", "how do", "tell me about"
    ]
    return any(kw in query.lower() for kw in keywords)
   
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

def chunk_text(collection_name, document_name, text, min_chunk_chars=100):
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
                "collection_name": collection_name,
                "document_name": document_name
            }
        })
    return chunk_data
