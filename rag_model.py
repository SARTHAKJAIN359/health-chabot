# rag_model.py
import os
import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = "data"
KB_FILE = os.path.join(DATA_DIR, "mental_health_knowledge.json")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")
KB_CACHE_PATH = os.path.join(ARTIFACTS_DIR, "kb_cache.pkl")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Spell correction setup ---
try:
    from textblob import TextBlob
    SPELL_CORRECTION = True
except ImportError:
    SPELL_CORRECTION = False
    print("⚠️  textblob not installed — spell correction disabled")


def correct_query(text):
    """Fix common typos in user query before searching."""
    if not SPELL_CORRECTION:
        return text
    try:
        corrected = str(TextBlob(text).correct())
        return corrected
    except Exception:
        return text


def normalize(text):
    """Lowercase and collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def load_knowledge_base():
    with open(KB_FILE, "r", encoding="utf-8") as f:
        kb = json.load(f)
    docs = []
    for entry in kb:
        # Include synonyms/alternate phrasings by repeating key words
        q = entry.get("question", "") or ""
        a = entry.get("answer", "") or ""
        text = f"{q} {q} {a}"   # weight question twice for better matching
        docs.append(normalize(text))
    return kb, docs


def save_embeddings():
    """Build TF-IDF vectorizer + matrix and persist them."""
    kb, docs = load_knowledge_base()
    vectorizer = TfidfVectorizer(
        max_df=0.90,
        min_df=1,
        ngram_range=(1, 3),          # up to trigrams — catches "how to sleep better"
        analyzer='word',
        sublinear_tf=True,           # dampens high-frequency terms
        strip_accents='unicode',
    )
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
    if not (os.path.exists(VECTORIZER_PATH) and
            os.path.exists(MATRIX_PATH) and
            os.path.exists(KB_CACHE_PATH)):
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
    Return the top_k most relevant answers from the knowledge base.
    Applies spell correction and normalized matching before searching.
    """
    kb, vectorizer, tfidf_matrix = load_embeddings()

    # Step 1: spell-correct the raw query
    corrected = correct_query(query)

    # Step 2: normalize
    clean = normalize(corrected)

    # Step 3: TF-IDF search
    query_vec = vectorizer.transform([clean])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    best_indices = cosine_similarities.argsort()[::-1][:top_k]

    results = []
    for idx in best_indices:
        entry = kb[idx]
        results.append({
            "id": entry.get("id"),
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "score": float(cosine_similarities[idx]),
            "corrected_query": corrected if corrected.lower() != query.lower() else None
        })
    return results
