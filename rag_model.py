# rag_model.py
import os
import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
KB_FILE = os.path.join(DATA_DIR, "mental_health_knowledge.json")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")
KB_CACHE_PATH = os.path.join(ARTIFACTS_DIR, "kb_cache.pkl")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Spell correction
try:
    from textblob import TextBlob
    SPELL_CORRECTION = True
except ImportError:
    SPELL_CORRECTION = False

# Synonym expansion — maps user words to related KB terms
SYNONYMS = {
    "sad": "depression depressed unhappy low mood",
    "depress": "depression depressed sad hopeless",
    "anxious": "anxiety anxious worried nervous panic",
    "anxiety": "anxiety anxious worried nervous panic stress",
    "stress": "stress burnout overwhelmed pressure",
    "burnout": "burnout exhausted overwhelmed stress signs recover",
    "tired": "tired exhausted fatigue burnout",
    "sleep": "sleep insomnia rest tired improve",
    "panic": "panic attack anxiety breathing grounding",
    "trauma": "trauma ptsd flashback",
    "lonely": "loneliness lonely isolated connection social",
    "grief": "grief loss coping mourning",
    "ocd": "obsessive compulsive disorder rituals thoughts",
    "adhd": "attention deficit hyperactivity disorder focus",
    "bipolar": "bipolar disorder mood swings mania",
    "therapy": "therapy cbt counseling treatment",
    "boundary": "boundaries healthy relationships limit",
    "mindful": "mindfulness meditation present calm",
    "breathe": "breathing exercises calm anxiety",
    "journal": "journaling writing emotions",
    "exercise": "exercise physical activity mood",
    "negative": "negative thoughts cognitive cbt reframe",
    "esteem": "self esteem confidence worth",
    "grounding": "grounding techniques anxiety 5 senses",
    "dissociat": "dissociation disconnect trauma",
    "resilience": "resilience cope recover setback",
    "creative": "creativity art music therapy",
    "work": "work stress burnout manage",
    "family": "family talk communicate support",
    "eat": "eating disorder anorexia bulimia food",
    "social media": "social media anxiety depression comparison",
    "relationship": "relationship breakup loss grief",
}


def expand_query(text):
    lower = text.lower()
    extras = []
    for keyword, expansion in SYNONYMS.items():
        if keyword in lower:
            extras.append(expansion)
    if extras:
        return text + " " + " ".join(extras)
    return text


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
    docs = []
    for entry in kb:
        q = entry.get("question", "") or ""
        a = entry.get("answer", "") or ""
        # Weight question 3x for better intent matching
        docs.append(normalize(f"{q} {q} {q} {a}"))
    return kb, docs


def save_embeddings():
    kb, docs = load_knowledge_base()
    vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=1,
        ngram_range=(1, 3),
        sublinear_tf=True,
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
    kb, vectorizer, tfidf_matrix = load_embeddings()
    corrected = correct_query(query)
    expanded = expand_query(corrected)
    clean = normalize(expanded)
    query_vec = vectorizer.transform([clean])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in best_indices:
        entry = kb[int(idx)]
        results.append({
            "id": entry.get("id"),
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "score": float(scores[idx]),
        })
    return results
