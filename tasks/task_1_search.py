import json
import os
import re
import csv
from typing import Dict, List, Any, Optional, Tuple

# --- Ensure modern SQLite (Chroma requires sqlite >= 3.35) ---
# This safely swaps in the bundled pysqlite3 (installed via pysqlite3-binary)
# if the host system ships an older sqlite3. No-op if already recent.
try:  # pragma: no cover
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # type: ignore
except Exception:
    pass

import chromadb
from chromadb.config import Settings

# Lazy import of sentence-transformers will happen inside _get_embedding_model
try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None  # Defer errors until used

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
VDB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "outputs", "chroma")
PERSONAS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "personas")
COLLECTION_NAME = "personas_collection"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

_embedding_model: Optional["SentenceTransformer"] = None  # type: ignore
_client: Optional[Any] = None


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _get_embedding_model():
    """
    Lazy-load the embedding model only when actually needed.
    Keeps startup fast for pages that don't use embeddings.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer  # local import (lazy)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_client() -> Any:
    global _client
    if _client is None:
        os.makedirs(VDB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=VDB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def initialize_search_engine(force_reindex: bool = False) -> Dict[str, Any]:
    """
    Initialize persistent ChromaDB with persona embeddings.

    - Creates a persistent collection if not present.
    - Indexes all persona .txt files located in assets/personas.
    - If force_reindex=True, drops existing collection and re-creates.
    """
    client = _get_client()

    # Drop & recreate if forcing reindex
    if force_reindex:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection_names = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in collection_names:
        return {"status": "ok", "message": "Collection already initialized"}

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    files = [f for f in os.listdir(PERSONAS_DIR) if f.lower().endswith(".txt")]
    files.sort()

    texts: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for idx, filename in enumerate(files):
        file_path = os.path.join(PERSONAS_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception:  # pragma: no cover
            content = ""
        if not content:
            continue
        texts.append(content)
        ids.append(f"persona_{idx}")
        metadatas.append({"filename": filename})

    if texts:
        model = _get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    return {"status": "ok", "message": f"Indexed {len(texts)} personas"}


def _extract_negative_constraints(query: str) -> List[str]:
    """
    Extract simple negative constraints like:
      'no smokers', "don't want smokers", 'avoid lawyers', 'without relocation'.

    Returns a list of lowercase raw substrings (very naive; may over-match).
    """
    lowered = query.lower()
    negatives: List[str] = []
    patterns = [
        r"don't want ([a-z0-9\- ,]+)",
        r"do not want ([a-z0-9\- ,]+)",
        r"no ([a-z0-9\- ,]+)",
        r"avoid ([a-z0-9\- ,]+)",
        r"without ([a-z0-9\- ,]+)",
    ]
    for pat in patterns:
        for match in re.findall(pat, lowered):
            parts = [w.strip() for w in match.split(",")]
            for p in parts:
                if p:
                    negatives.append(p)
    # Normalize & dedupe
    cleaned = []
    for n in negatives:
        n2 = n.strip()
        if not n2:
            continue
        cleaned.append(n2)
    return sorted(set(cleaned))


def _decompose_query_with_llm(query: str) -> List[str]:
    """
    Use Gemini to decompose a complex query into 3–5 focused sub-queries.
    Falls back to [query] on any error or if LLM key not available.
    """
    prompt = (
        "Analyze the following user query. Decompose it into 3 to 5 simpler, self-contained sub-queries "
        "that cover its key facets. The goal is to run a separate search for each of these to find the best "
        "possible candidates. Return ONLY a valid JSON object with the schema: "
        '{"sub_queries": ["query 1", "query 2", "..."]}. '
        "For example, if the user asks for 'a musician in Europe interested in tech', your output should be "
        '{"sub_queries": ["musician who lives in Europe", "person with an interest in technology and engineering", '
        '"creative professional with technical skills"]}.\n\n'
        f"User query:\n{query}"
    )

    try:
        if genai is None:
            raise RuntimeError("google-generativeai not installed or failed to import")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY/GEMINI_API_KEY env var")
        genai.configure(api_key=api_key)
        model_g = genai.GenerativeModel("gemini-1.5-flash")
        response = model_g.generate_content(prompt)
        raw_text = response.text if hasattr(response, "text") else str(response)

        json_text = raw_text.strip()
        if json_text.startswith("```") and json_text.endswith("```"):
            json_text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", json_text, flags=re.DOTALL)
        match = re.search(r"\{[\s\S]*\}$", json_text)
        if match:
            json_text = match.group(0)
        parsed = json.loads(json_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("sub_queries"), list) and parsed["sub_queries"]:
            clean: List[str] = []
            seen = set()
            for s in parsed["sub_queries"]:
                if isinstance(s, str):
                    s2 = s.strip()
                    if s2 and s2 not in seen:
                        seen.add(s2)
                        clean.append(s2)
            if clean:
                return clean[:5]
    except Exception:
        pass

    return [query]


def _load_persona_index(index_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Load personas_index.csv into a dict keyed by filename.
    """
    if index_path is None:
        index_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets",
            "personas",
            "personas_index.csv",
        )
    index: Dict[str, Dict[str, str]] = {}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("file")
                if fname:
                    index[fname] = row
    except Exception:
        index = {}
    return index


def _extract_positive_signals(query: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Extract positive (must-have) and nice-to-have signals from the query.
    Very coarse heuristic keyed to a few hard-coded domains.
    """
    q = query.lower()
    ai_terms = [
        "ai", "ml", "machine learning", "artificial intelligence", "llm", "language model",
        "nlp", "natural language", "computer vision", "cv", "mlo ps", "mloops", "mops"
    ]
    physics_terms = [
        "physics", "physicist", "physics curriculum", "quantum", "thermodynamics", "mechanics"
    ]
    tennis_terms = ["tennis"]

    must: Dict[str, List[str]] = {}
    nice: Dict[str, List[str]] = {}

    if any(t in q for t in ai_terms):
        must["ai"] = ai_terms
    if any(t in q for t in physics_terms):
        must["physics"] = physics_terms
    if any(t in q for t in tennis_terms):
        nice["tennis"] = tennis_terms

    return must, nice


def _score_candidate(
    doc_text: str,
    filename: str,
    must: Dict[str, List[str]],
    nice: Dict[str, List[str]],
    index_rows: Dict[str, Dict[str, str]],
) -> float:
    """
    Compute a simple lexical/tag score complementing vector similarity (which Chroma already applied).
    """
    text = (doc_text or "").lower()
    score = 0.0

    for _, terms in must.items():
        if not any(term in text for term in terms):
            score -= 2.0
        else:
            score += 1.0

    for _, terms in nice.items():
        if any(term in text for term in terms):
            score += 0.5

    row = index_rows.get(filename, {})
    tags = (row.get("tags") or "").lower()
    profession = (row.get("profession") or "").lower()

    if "ai/ml" in tags or any(k in profession for k in ["ai", "ml", "nlp", "computer vision", "llm"]):
        score += 1.0
    if any(k in profession for k in ["physics", "physicist"]):
        score += 1.0

    return score


# ------------------------------------------------------------------
# Public search function
# ------------------------------------------------------------------
def perform_search(query: str) -> Dict[str, Any]:
    """
    Persona search pipeline:
      1. Ensure collection exists (auto-init if missing).
      2. LLM-based query decomposition (fallback: original).
      3. Iterative vector retrieval & aggregation (dedupe by filename).
      4. Negative constraint filtering.
      5. Lexical re-scoring & optional must-have filtering.
      6. Optional Gemini summarization (strict JSON). Fallback includes action_points.

    Returns JSON: {"results": [{ "name": "...", "compatibility_percentage": "...",
                                 "insights": "...", "action_points": "..." }, ...]}
    """
    if not query or not query.strip():
        return {"results": []}

    # 1. Ensure collection exists
    client = _get_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        initialize_search_engine()
        collection = client.get_collection(COLLECTION_NAME)

    # 2. Decompose query
    sub_queries = _decompose_query_with_llm(query)

    # 3. Retrieval & aggregation
    model = _get_embedding_model()
    aggregated_docs: List[str] = []
    aggregated_meta: List[Dict[str, Any]] = []
    seen_filenames: set = set()
    negatives = _extract_negative_constraints(query)

    for sub_q in sub_queries:
        try:
            emb = model.encode([sub_q], convert_to_numpy=True).tolist()[0]
            res = collection.query(query_embeddings=[emb], n_results=15)
            docs_i: List[str] = res.get("documents", [[]])[0] if res else []
            metas_i: List[Dict[str, Any]] = res.get("metadatas", [[]])[0] if res else []
        except Exception:
            docs_i, metas_i = [], []

        for d, m in zip(docs_i, metas_i):
            fname = (m or {}).get("filename", "")
            if fname and fname in seen_filenames:
                continue
            lowered = (d or "").lower()
            if negatives and any(neg for neg in negatives if neg in lowered):
                continue
            if fname:
                seen_filenames.add(fname)
            aggregated_docs.append(d)
            aggregated_meta.append(m)

    if not aggregated_docs:
        return {"results": []}

    # 4. Re-rank
    must, nice = _extract_positive_signals(query)
    index_rows: Dict[str, Dict[str, str]] = {}  # Not using CSV ranking now

    base_scores = {i: (len(aggregated_docs) - i) * 0.01 for i in range(len(aggregated_docs))}
    scored: List[Tuple[float, int, str, Dict[str, Any]]] = []
    for i, (doc, meta) in enumerate(zip(aggregated_docs, aggregated_meta)):
        fname = (meta or {}).get("filename", "")
        lex_score = _score_candidate(doc, fname, must, nice, index_rows)
        total_score = lex_score + base_scores.get(i, 0.0)
        scored.append((total_score, i, doc, meta))

    def matches_must(doc_text: str) -> bool:
        t = (doc_text or "").lower()
        for _, terms in must.items():
            if not any(term in t for term in terms):
                return False
        return True

    if must:
        must_matches = [s for s in scored if matches_must(s[2])]
        if len(must_matches) >= 5:
            scored = must_matches

    scored.sort(key=lambda x: x[0], reverse=True)
    shortlisted = scored[:5]
    shortlisted_docs = [s[2] for s in shortlisted]
    shortlisted_meta = [s[3] for s in shortlisted]

    if not shortlisted_docs:
        return {"results": []}

    # 5. Prepare summarization prompt
    personas_block = "\n\n".join(
        [
            f"Persona {i+1} (file: {m.get('filename','?')}):\n{d}"
            for i, (d, m) in enumerate(zip(shortlisted_docs, shortlisted_meta))
        ]
    )
    system_instruction = (
        "You are a matching assistant. Given a user query and candidate personas, analyze compatibility. "
        'Return strictly JSON: {"results":[{"name":"...","compatibility_percentage":"...","insights":"...",'
        '"action_points":"..."}]} '
        "action_points: a friendly, specific question the user can ask to start a conversation. "
        "Only output valid JSON. No markdown."
    )
    user_prompt = (
        f"User query:\n{query}\n\nCandidates:\n{personas_block}\n\n"
        "Rank top matches. Name must be extracted. Provide numeric percentage (string)."
    )

    # 6. Gemini summarization with strong fallback
    try:
        if genai is None:
            raise RuntimeError("google-generativeai not installed or failed to import")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY/GEMINI_API_KEY env var")
        genai.configure(api_key=api_key)
        model_g = genai.GenerativeModel("gemini-1.5-flash")
        response = model_g.generate_content([system_instruction, user_prompt])
        raw_text = response.text if hasattr(response, "text") else str(response)
    except Exception:
        return _fallback_results(shortlisted_docs)

    json_text = raw_text.strip()
    if json_text.startswith("```") and json_text.endswith("```"):
        json_text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", json_text, flags=re.DOTALL)
    match = re.search(r"\{[\s\S]*\}$", json_text)
    if match:
        json_text = match.group(0)

    try:
        parsed = json.loads(json_text)
        if (
            isinstance(parsed, dict)
            and isinstance(parsed.get("results"), list)
        ):
            # Ensure each result has required fields; auto-fill if missing
            for r in parsed["results"]:
                if "action_points" not in r:
                    r["action_points"] = "Ask them about one of their highlighted interests to start a conversation."
                if "compatibility_percentage" not in r:
                    r["compatibility_percentage"] = "75"
            return parsed
    except Exception:
        pass

    return _fallback_results(shortlisted_docs)


def _fallback_results(shortlisted_docs: List[str]) -> Dict[str, Any]:
    """
    Deterministic fallback if Gemini unavailable or JSON parsing fails.
    Ensures schema compliance including action_points.
    """
    results = []
    for i, d in enumerate(shortlisted_docs[:3]):
        name_match = re.search(r"Name:\s*([^\n]+)", d)
        name = name_match.group(1).strip() if name_match else f"Candidate {i+1}"
        base_pct = max(50, 90 - i * 10)
        results.append(
            {
                "name": name,
                "compatibility_percentage": str(base_pct),
                "insights": "Heuristic match based on lexical + similarity scoring (LLM unavailable).",
                "action_points": "Open by referencing a shared domain interest (e.g., AI, physics, or hobbies).",
            }
        )
    return {"results": results}