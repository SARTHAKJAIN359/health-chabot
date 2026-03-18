# rag_model.py
import os
import json
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
KB_FILE = os.path.join(DATA_DIR, "mental_health_knowledge.json")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")
KB_CACHE_PATH = os.path.join(ARTIFACTS_DIR, "kb_cache.pkl")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Spell correction ---
try:
    from textblob import TextBlob
    SPELL_CORRECTION = True
except ImportError:
    SPELL_CORRECTION = False

# --- Synonym map: expands user query with related terms ---
SYNONYMS = {
    "sad": "sad depression depressed unhappy low mood",
    "depress": "depression depressed sad hopeless low mood",
    "anxious": "anxiety anxious worried nervous stress",
    "anxiety": "anxiety anxious worried nervous panic stress",
    "stress": "stress stressed burnout overwhelmed pressure",
    "burnout": "burnout exhausted tired overwhelmed stress recover",
    "tired": "tired exhausted fatigue burnout energy",
    "sleep": "sleep insomnia rest tired fatigue",
    "panic": "panic attack anxiety fear heart racing",
    "trauma": "trauma ptsd flashback abuse past",
    "lonely": "lonely loneliness isolated alone connection",
    "anger": "anger angry irritable mood emotional",
    "grief": "grief loss mourning bereavement death sad",
    "self harm": "self harm hurt cutting injury",
    "suicide": "suicide suicidal crisis helpline",
    "ocd": "ocd obsessive compulsive thoughts rituals",
    "adhd": "adhd attention focus hyperactivity",
    "bipolar": "bipolar mood swings mania depression",
    "therapy": "therapy therapist counseling cbt treatment",
    "medication": "medication medicine psychiatrist treatment",
    "eat": "eating disorder anorexia bulimia food body",
    "boundary": "boundaries limits healthy relationships",
    "mindful": "mindfulness meditation present calm",
    "breathe": "breathing breath calm relax anxiety",
    "journal": "journaling writing emotions thoughts",
    "exercise": "exercise physical activity mood mental health",
    "diet": "diet nutrition food mental health brain",
    "social media": "social media screen time comparison anxiety",
    "relationship": "relationship breakup loss grief cope",
    "work": "work stress burnout career job pressure",
    "family": "family talk communicate support mental health",
    "friend": "friend support helping struggling crisis",
    "negative": "negative thoughts cognitive cbt reframe",
    "concentration": "concentration focus adhd attention",
    "memory": "memory focus cognitive mental health",
    "dissociat": "dissociation disconnect trauma ptsd",
    "psychosis": "psychosis hallucination delusion schizophrenia",
    "schizophrenia": "schizophrenia psychosis delusion hallucination",
    "resilience": "resilience cope recover setback strength",
    "confidence": "confidence self esteem worth value",
    "esteem": "self esteem confidence worth value",
    "grounding": "grounding technique anxiety panic present 5 senses",
    "cbt": "cbt cognitive behavioral therapy negative thoughts",
    "dbt": "dbt dialectical behavior therapy emotional regulation",
    "emdr": "emdr trauma therapy ptsd processing",
    "nature": "nature outdoors walk green mental health",
    "creative": "creative art music therapy expression",
    "gut": "gut health brain food diet mental health",
    "neurodiver": "neurodiversity adhd autism dyslexia brain",
    "compas": "compassion fatigue caregiver burnout empathy",
}


def expand_query(text):
    """Add synonym expansions to improve TF-IDF matching."""
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
        # Weight question 3x — the question is the best semantic signal
        text = f"{q} {q} {q} {a}"
        docs.append(normalize(text))
    return kb, docs


def save_embeddings():
    kb, docs = load_knowledge_base()
    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=1,
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

    # Step 1: spell correct
    corrected = correct_query(query)

    # Step 2: expand with synonyms
    expanded = expand_query(corrected)

    # Step 3: normalize
    clean = normalize(expanded)

    # Step 4: TF-IDF similarity
    query_vec = vectorizer.transform([clean])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in best_indices:
        entry = kb[int(idx)]
        results.append({
            "id": entry.get("id"),
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "score": float(scores[idx]),
            "corrected_query": corrected if corrected.lower() != query.lower() else None
        })
    return results
