from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer('BAAI/bge-large-en-v1.5')
reranker = CrossEncoder('BAAI/bge-reranker-base')

def embed_texts(texts):
    return model.encode(texts).tolist()

def rerank(query, docs):
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    return scores
