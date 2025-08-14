import json
import os
import re
import csv
from typing import Dict, List, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional at import time
    genai = None  # Defer errors until use


VDB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "outputs", "chroma")
PERSONAS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "personas")
COLLECTION_NAME = "personas_collection"


_embedding_model: Optional[SentenceTransformer] = None
_client: Optional[Any] = None
_collection = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_client() -> Any:
    global _client
    if _client is None:
        os.makedirs(VDB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=VDB_DIR, settings=Settings(anonymized_telemetry=False))
    return _client


def initialize_search_engine() -> Dict[str, Any]:
    """Initialize persistent ChromaDB with persona embeddings.

    - Creates a persistent collection if not present and indexes persona .txt files
    - Safe to call multiple times; skips re-indexing if docs already present
    """
    client = _get_client()
    collection_names = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in collection_names:
        # Already initialized
        return {"status": "ok", "message": "Collection already initialized"}

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    files = [f for f in os.listdir(PERSONAS_DIR) if f.lower().endswith(".txt")]
    files.sort()
    texts, ids, metadatas = [], [], []
    for idx, filename in enumerate(files):
        file_path = os.path.join(PERSONAS_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as e:  # pragma: no cover
            content = ""
        if not content:
            continue
        texts.append(content)
        ids.append(f"persona_{idx}")
        metadatas.append({"filename": filename})

    if texts:
        model = _get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    return {"status": "ok", "message": f"Indexed {len(texts)} personas"}


def _extract_negative_constraints(query: str) -> List[str]:
    """Extract simple negative constraints like 'no smokers', "don't want smokers".

    Returns a list of lowercase keywords to exclude.
    """
    lowered = query.lower()
    negatives: List[str] = []
    # Simple patterns; extendable
    patterns = [
        r"don't want ([a-z\- ]+)",
        r"do not want ([a-z\- ]+)",
        r"no ([a-z\- ]+)",
        r"avoid ([a-z\- ]+)",
        r"without ([a-z\- ]+)",
    ]
    for pat in patterns:
        for match in re.findall(pat, lowered):
            negatives.extend([w.strip() for w in match.split(",")])
    # Normalize common terms
    normalized = []
    for token in negatives:
        token = token.strip()
        if not token:
            continue
        normalized.append(token)
    # Deduplicate
    return sorted(set(normalized))


def _decompose_query_with_llm(query: str) -> List[str]:
    """Use Gemini to decompose a complex query into 3-5 focused sub-queries.

    Returns a list of sub-queries. Falls back to [query] on any failure.
    """
    prompt = (
        "Analyze the following user query. Decompose it into 3 to 5 simpler, self-contained sub-queries "
        "that cover its key facets. The goal is to run a separate search for each of these to find the best "
        "possible candidates. Return ONLY a valid JSON object with the schema: {\"sub_queries\": [\"query 1\", \"query 2\", ...]}. "
        "For example, if the user asks for 'a musician in Europe interested in tech', your output should be "
        "{\"sub_queries\": [\"musician who lives in Europe\", \"person with an interest in technology and engineering\", \"creative professional with technical skills\"]}.\n\n"
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
            # Normalize whitespace and deduplicate
            clean = []
            seen = set()
            for s in parsed["sub_queries"]:
                if not isinstance(s, str):
                    continue
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
    """Load `personas_index.csv` into a dict keyed by filename.

    Returns: { filename: {"name": ..., "age": ..., "location": ..., "profession": ..., "tags": ... } }
    """
    if index_path is None:
        index_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "assets", "personas", "personas_index.csv"
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
        # If index not available, proceed without it
        index = {}
    return index


def _extract_positive_signals(query: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Extract positive signals from the query.

    Returns a tuple of (must_have_terms, nice_to_have_terms) where each is a mapping
    of category -> list of terms. Categories used internally: "ai", "physics", "tennis".
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

    # Heuristic: if a term group is explicitly mentioned, treat as must-have for AI and physics.
    if any(t in q for t in ai_terms):
        must["ai"] = ai_terms
    if any(t in q for t in physics_terms):
        must["physics"] = physics_terms

    # Tennis is a soft preference
    if any(t in q for t in tennis_terms):
        nice["tennis"] = tennis_terms

    return must, nice


def _score_candidate(doc_text: str, filename: str, must: Dict[str, List[str]], nice: Dict[str, List[str]],
                     index_rows: Dict[str, Dict[str, str]]) -> float:
    """Compute a lexical/tag score to combine with vector similarity.

    - Heavily penalize missing must-have categories
    - Reward presence of nice-to-have terms
    - Boost if CSV tags indicate relevant domains (e.g., AI/ML)
    """
    text = (doc_text or "").lower()
    score = 0.0

    # Must-have checks (hard constraints): subtract if missing
    for category, terms in must.items():
        if not any(term in text for term in terms):
            score -= 2.0  # strong penalty per missing must-have
        else:
            score += 1.0

    # Nice-to-have checks
    for _, terms in nice.items():
        if any(term in text for term in terms):
            score += 0.5

    # CSV tag boosts
    row = index_rows.get(filename, {})
    tags = (row.get("tags") or "").lower()
    profession = (row.get("profession") or "").lower()

    if "ai/ml" in tags or any(k in profession for k in ["ai", "ml", "nlp", "computer vision", "llm"]):
        score += 1.0
    if any(k in profession for k in ["physics", "physicist"]):
        score += 1.0

    return score


def perform_search(query: str) -> Dict[str, Any]:
    """Search personas and have an LLM produce a ranked JSON via Multi-Query Retrieval.

    Steps:
    1) Decompose the user query into multiple sub-queries using Gemini
    2) For each sub-query, perform vector search (top 10-15) and aggregate unique candidates
    3) Apply negative constraint filtering on aggregated texts
    4) Re-rank the combined pool using positive-signal scoring vs the original query
    5) Send top 5 to Gemini for insights; parse and return JSON
    """
    if not query or not query.strip():
        return {"results": []}

    client = _get_client()
    collection = client.get_collection(COLLECTION_NAME)

    # Step 1: Decompose
    sub_queries = _decompose_query_with_llm(query)

    # Step 2: Parallel (iterative) retrieval and aggregation with deduplication
    model = _get_embedding_model()
    aggregated_docs: List[str] = []
    aggregated_meta: List[Dict[str, Any]] = []
    seen_filenames: set = set()

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
            # Step 3: Apply negative filters early to reduce pool size
            lowered = (d or "").lower()
            negatives = _extract_negative_constraints(query)
            if negatives and any(neg in lowered for neg in negatives):
                continue
            if fname:
                seen_filenames.add(fname)
            aggregated_docs.append(d)
            aggregated_meta.append(m)

    # If nothing aggregated, return empty
    if not aggregated_docs:
        return {"results": []}

    # Step 4: Re-rank the aggregated pool
    must, nice = _extract_positive_signals(query)
    index_rows = {}  # do not rely on CSV index for ranking

    # Base score gives slight preference to earlier (higher-ranked) retrievals across sub-queries
    base_scores = {i: (len(aggregated_docs) - i) * 0.01 for i in range(len(aggregated_docs))}

    scored: List[Tuple[float, int, str, Dict[str, Any]]] = []
    for i, (doc, meta) in enumerate(zip(aggregated_docs, aggregated_meta)):
        fname = (meta or {}).get("filename", "")
        lex_score = _score_candidate(doc, fname, must, nice, index_rows)
        total_score = lex_score + base_scores.get(i, 0.0)
        scored.append((total_score, i, doc, meta))

    # Optionally require must-haves if we have enough candidates that satisfy them
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

    # Prepare prompt
    personas_block = "\n\n".join(
        [f"Persona {i+1} (file: {m.get('filename','?')}):\n{d}" for i, (d, m) in enumerate(zip(shortlisted_docs, shortlisted_meta))]
    )
    system_instruction = (
        "You are a matching assistant. Given a user query and candidate personas, "
        "analyze compatibility. Return strictly a JSON object with the schema: "
        "{\"results\": [{\"name\": \"...\", \"compatibility_percentage\": \"...\", \"insights\": \"...\", \"action_points\": \"...\"}]} "
        "The 'action_points' field should contain a specific, friendly question the user can ask to start a conversation "
        "based on the shared interests you identified in the 'insights'. Only output valid JSON. Do not include markdown."
    )
    user_prompt = (
        f"User query:\n{query}\n\nCandidates:\n{personas_block}\n\n"
        "Rank top matches. Name must be extracted from persona. Provide a numeric percentage as a string."
    )

    # Call Gemini
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
    except Exception as e:
        # Fallback: create a deterministic JSON using cosine order if LLM fails
        fallback = {
            "results": [
                {
                    "name": re.search(r"Name:\s*([^\n]+)", d).group(1).strip() if re.search(r"Name:\s*([^\n]+)", d) else "Unknown",
                    "compatibility_percentage": str(90 - i * 10),
                    "insights": "Auto-generated due to LLM error."
                }
                for i, d in enumerate(shortlisted_docs[:3])
            ]
        }
        return fallback

    # Ensure we extract JSON only
    json_text = raw_text.strip()
    # Remove code fences if any
    if json_text.startswith("```") and json_text.endswith("```"):
        json_text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", json_text, flags=re.DOTALL)
    # Find first JSON object
    match = re.search(r"\{[\s\S]*\}$", json_text)
    if match:
        json_text = match.group(0)
    try:
        parsed = json.loads(json_text)
        # Basic validation
        if isinstance(parsed, dict) and "results" in parsed and isinstance(parsed["results"], list):
            return parsed
    except Exception:
        pass

    # Final fallback minimal structure
    return {
        "results": [
            {
                "name": re.search(r"Name:\s*([^\n]+)", d).group(1).strip() if re.search(r"Name:\s*([^\n]+)", d) else "Unknown",
                "compatibility_percentage": "75",
                "insights": "Returned via fallback parser."
            }
            for d in shortlisted_docs[:3]
        ]
    }


