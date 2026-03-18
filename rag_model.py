# rag_model.py
import os
import json
import pickle
import re
import numpy as np

DATA_DIR = "data"
KB_FILE = os.path.join(DATA_DIR, "mental_health_knowledge.json")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "semantic_embeddings.pkl")
KB_CACHE_PATH = os.path.join(ARTIFACTS_DIR, "kb_cache.pkl")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Semantic model setup ---
try:
    from sentence_transformers import SentenceTransformer, util
    # all-MiniLM-L6-v2 is small (80MB), fast, and excellent for semantic similarity
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_AVAILABLE = True
    print("✅ Semantic embeddings enabled (all-MiniLM-L6-v2)")
except Exception as e:
    SEMANTIC_AVAILABLE = False
    print(f"⚠️  sentence-transformers unavailable, falling back to TF-IDF: {e}")

# --- Spell correction setup ---
try:
    from textblob import TextBlob
    SPELL_CORRECTION = True
except ImportError:
    SPELL_CORRECTION = False


def correct_query(text):
    if not SPELL_CORRECTION:
        return text
    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text


def normalize(text):
    return re.sub(r'\s+', ' ', text.lower().strip())


def load_knowledge_base():
    with open(KB_FILE, "r", encoding="utf-8") as f:
        kb = json.load(f)
    # For embedding: use question only (cleaner semantic signal)
    docs = [normalize(entry.get("question", "") or "") for entry in kb]
    return kb, docs


# ── SEMANTIC PATH ──────────────────────────────────────────────────────────────

def save_embeddings():
    kb, docs = load_knowledge_base()

    if SEMANTIC_AVAILABLE:
        # Embed all questions
        embeddings = _model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)
    else:
        # Fall back to TF-IDF
        _save_tfidf(kb, docs)

    with open(KB_CACHE_PATH, "wb") as f:
        pickle.dump(kb, f)
    return True


def load_embeddings():
    if not os.path.exists(KB_CACHE_PATH):
        save_embeddings()

    with open(KB_CACHE_PATH, "rb") as f:
        kb = pickle.load(f)

    if SEMANTIC_AVAILABLE:
        if not os.path.exists(EMBEDDINGS_PATH):
            save_embeddings()
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        return kb, embeddings, None   # third slot unused in semantic path
    else:
        return _load_tfidf(kb)


def answer_query(query, top_k=1):
    """
    Semantic search: understands meaning, not just word overlap.
    Falls back to TF-IDF if sentence-transformers is unavailable.
    """
    # Step 1: spell correct + normalize
    corrected = correct_query(query)
    clean = normalize(corrected)

    kb, embeddings_or_vectorizer, tfidf_matrix = load_embeddings()

    if SEMANTIC_AVAILABLE:
        # Encode the query
        query_embedding = _model.encode(clean, convert_to_numpy=True)
        # Cosine similarity against all KB question embeddings
        scores = util.cos_sim(query_embedding, embeddings_or_vectorizer)[0].numpy()
    else:
        # TF-IDF fallback
        from sklearn.metrics.pairwise import linear_kernel
        query_vec = embeddings_or_vectorizer.transform([clean])
        scores = linear_kernel(query_vec, tfidf_matrix).flatten()

    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in best_indices:
        entry = kb[idx]
        results.append({
            "id": entry.get("id"),
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "score": float(scores[idx]),
            "corrected_query": corrected if corrected.lower() != query.lower() else None
        })
    return results


# ── TF-IDF FALLBACK ────────────────────────────────────────────────────────────

VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")


def _save_tfidf(kb, docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_df=0.90, min_df=1, ngram_range=(1, 3), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(docs)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)


def _load_tfidf(kb):
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MATRIX_PATH)):
        _, docs = load_knowledge_base()
        _save_tfidf(kb, docs)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    return kb, vectorizer, tfidf_matrix
