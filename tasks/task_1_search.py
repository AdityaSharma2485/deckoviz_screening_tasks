import json
import os
import re
from typing import Dict, List, Any

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


_embedding_model: SentenceTransformer | None = None
_client: chromadb.Client | None = None
_collection = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_client() -> chromadb.Client:
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


def perform_search(query: str) -> Dict[str, Any]:
    """Search personas and have an LLM produce a ranked JSON.

    Steps:
    1) Embed query and retrieve top 10 from ChromaDB
    2) Apply negative constraint filtering on texts
    3) Send top 5 filtered to Gemini with an instruction to return JSON
    4) Parse and return JSON
    """
    if not query or not query.strip():
        return {"results": []}

    client = _get_client()
    collection = client.get_collection(COLLECTION_NAME)

    model = _get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    documents: List[str] = results.get("documents", [[]])[0] if results else []
    metadatas: List[Dict[str, Any]] = results.get("metadatas", [[]])[0] if results else []

    negatives = _extract_negative_constraints(query)
    if negatives:
        filtered = []
        filtered_meta = []
        for doc, meta in zip(documents, metadatas):
            lowered = doc.lower()
            if any(neg in lowered for neg in negatives):
                continue
            filtered.append(doc)
            filtered_meta.append(meta)
        documents, metadatas = filtered, filtered_meta

    shortlisted_docs = documents[:5]
    shortlisted_meta = metadatas[:5]

    if not shortlisted_docs:
        return {"results": []}

    # Prepare prompt
    personas_block = "\n\n".join(
        [f"Persona {i+1} (file: {m.get('filename','?')}):\n{d}" for i, (d, m) in enumerate(zip(shortlisted_docs, shortlisted_meta))]
    )
    system_instruction = (
        "You are a matching assistant. Given a user query and candidate personas, "
        "analyze compatibility. Return strictly a JSON object with the schema: "
        "{\"results\": [{\"name\": \"...\", \"compatibility_percentage\": \"...\", \"insights\": \"...\"}]} "
        "Only output valid JSON. Do not include markdown."
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


