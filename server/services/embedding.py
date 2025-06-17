from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

model = SentenceTransformer('BAAI/bge-large-en-v1.5')
reranker = CrossEncoder('BAAI/bge-reranker-base')

def embed_texts(texts, batch_size=8):
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch).tolist()
        all_embeddings.extend(embeddings)

        # Optional: Clean MPS cache (only relevant if using Apple M1/M2 GPU)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Optional: Print progress
        print(f"[Embed] Progress: {min(i+batch_size, total)}/{total}")

    return all_embeddings