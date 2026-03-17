# rag_model.py
import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = "data"
KB_FILE = os.path.join(DATA_DIR, "mental_health_knowledge.json")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")
KB_CACHE_PATH = os.path.join(ARTIFACTS_DIR, "kb_cache.pkl")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_knowledge_base():
    with open(KB_FILE, "r", encoding="utf-8") as f:
        kb = json.load(f)
    # Create a list of "question + answer" text to embed/search
    docs = []
    for entry in kb:
        text = (entry.get("question", "") or "") + " " + (entry.get("answer", "") or "")
        docs.append(text.strip())
    return kb, docs


def save_embeddings():
    """Build TF-IDF vectorizer + matrix and persist them."""
    kb, docs = load_knowledge_base()
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(docs)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(KB_CACHE_PATH, "wb") as f:
        pickle.dump(kb, f)
    return True


def load_embeddings():
    """Load precomputed artifacts. If missing, build them."""
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MATRIX_PATH) and os.path.exists(KB_CACHE_PATH)):
        save_embeddings()
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(KB_CACHE_PATH, "rb") as f:
        kb = pickle.load(f)
    return kb, vectorizer, tfidf_matrix


def answer_query(query, top_k=1):
    """
    Return the top_k most relevant answers from the knowledge base for the given query.
    Returns a list of dicts: [{"id":..., "question":..., "answer":..., "score":...}, ...]
    """
    kb, vectorizer, tfidf_matrix = load_embeddings()
    query_vec = vectorizer.transform([query])
    # cosine similarity via linear_kernel (fast for TF-IDF)
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    best_indices = cosine_similarities.argsort()[::-1][:top_k]
    results = []
    for idx in best_indices:
        entry = kb[idx]
        results.append({
            "id": entry.get("id"),
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "score": float(cosine_similarities[idx])
        })
    return results
