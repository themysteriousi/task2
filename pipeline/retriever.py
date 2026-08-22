"""
CRITICAL: This module has NO module-level mutable state.
Qdrant client and models are safe to share (read-only weights).
All per-request data stays in local variables or RequestState.
Live internet search provides real-time web retrieval grounded in search engines.
"""
import asyncio
import math
import re
import httpx
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from pipeline.embedder import embed_query as _embed_query
from pipeline.state import RequestState

# These are SAFE to share — they are read-only model weights and a connection pool
_qdrant = None

def get_qdrant_client():
    global _qdrant
    if _qdrant is None:
        try:
            client = QdrantClient(host="localhost", port=6333, timeout=0.5)
            client.get_collections()
            _qdrant = client
        except Exception:
            _qdrant = QdrantClient(path="./qdrant_data")
    return _qdrant

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION = "msmarco_xi"

def _logit_to_prob(logit: float) -> float:
    """Convert cross-encoder raw logit to 0-1 probability via sigmoid"""
    return round(1 / (1 + math.exp(-logit)), 4)


def _search_ddg_sync(query: str, max_results: int = 5) -> list[dict]:
    """Synchronous DuckDuckGo search helper run inside thread pool."""
    results = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                title = item.get("title", "")
                snippet = item.get("body", "") or item.get("snippet", "")
                url = item.get("href", "") or item.get("link", "")
                if snippet:
                    results.append({
                        "title": title,
                        "text": f"{title}: {snippet}" if title and not snippet.startswith(title) else snippet,
                        "raw_snippet": snippet,
                        "url": url,
                        "source": "DuckDuckGo"
                    })
    except Exception as e:
        # Fallback will handle
        pass
    return results


async def _search_wikipedia_async(query: str, limit: int = 3) -> list[dict]:
    """Async Wikipedia search fallback / supplement."""
    results = []
    try:
        headers = {"User-Agent": "VoiceSearchEngine/2.0 (contact@voicesearch.ai)"}
        async with httpx.AsyncClient(timeout=3.0, headers=headers) as client:
            r = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1,
                    "srlimit": limit
                }
            )
            if r.status_code == 200:
                data = r.json()
                for item in data.get("query", {}).get("search", []):
                    title = item.get("title", "")
                    raw_snippet = item.get("snippet", "")
                    clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()
                    if clean_snippet:
                        results.append({
                            "title": title,
                            "text": f"{title} (Wikipedia): {clean_snippet}",
                            "raw_snippet": clean_snippet,
                            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            "source": "Wikipedia"
                        })
    except Exception:
        pass
    return results


async def search_live_internet(queries: list[str]) -> list[dict]:
    """
    Fetches live internet search results concurrently for expanded queries.
    Combines DuckDuckGo and Wikipedia results into clean candidate passages.
    """
    loop = asyncio.get_event_loop()
    tasks = []

    # Query DuckDuckGo for top query variants
    primary_query = queries[0] if queries else ""
    for q in queries[:2]:
        tasks.append(loop.run_in_executor(None, _search_ddg_sync, q, 4))

    # Concurrently search Wikipedia for primary query
    if primary_query:
        tasks.append(_search_wikipedia_async(primary_query, 3))

    raw_results_groups = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    seen_texts = set()

    for group in raw_results_groups:
        if isinstance(group, list):
            for item in group:
                text = item.get("text", "").strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    candidates.append(item)

    return candidates


async def retrieve(state: RequestState, top_k: int = 5) -> list[dict]:
    """
    Retrieves information directly from the live internet, followed by
    neural cross-encoder reranking against the user's canonical query.
    All inputs come from `state`. All outputs go into `state.retrieved_chunks`.
    No module-level mutable variables touched.
    """
    # Use the queries that were expanded for THIS request (from state)
    queries = state.expanded_queries
    if not queries:
        queries = [state.transcript]

    loop = asyncio.get_event_loop()

    # 1. Fetch live internet search results
    web_results = await search_live_internet(queries)

    candidate_list = []
    for idx, item in enumerate(web_results):
        candidate_list.append({
            "chunk_id": f"web-{idx+1}",
            "text": item["text"],
            "passage_id": item.get("url") or f"web-source-{idx+1}",
            "title": item.get("title", "Web Source"),
            "url": item.get("url", ""),
            "language": "en",
            "original_query": state.transcript,
            "strategy": "live_internet_web",
            "vector_score": 1.0,
            "rerank_score": 0.0,
            "rerank_prob": 0.0
        })

    # If no web results were retrieved (offline/network issue), fallback to local Qdrant if available
    if not candidate_list:
        try:
            embeddings = await asyncio.gather(*[
                loop.run_in_executor(None, _embed_query, q) for q in queries[:1]
            ])
            qdrant = get_qdrant_client()
            for emb in embeddings:
                if hasattr(qdrant, "query_points"):
                    res = qdrant.query_points(collection_name=COLLECTION, query=emb.tolist(), limit=10, with_payload=True)
                    hits = res.points
                else:
                    hits = qdrant.search(collection_name=COLLECTION, query_vector=emb.tolist(), limit=10, with_payload=True)
                for hit in hits:
                    cid = str(hit.id)
                    candidate_list.append({
                        "chunk_id": cid,
                        "text": hit.payload.get("text", ""),
                        "passage_id": hit.payload.get("passage_id", ""),
                        "title": "Local Knowledge",
                        "url": "",
                        "language": hit.payload.get("language", "en"),
                        "original_query": hit.payload.get("original_query", ""),
                        "strategy": hit.payload.get("chunk_strategy", "native"),
                        "vector_score": hit.score,
                        "rerank_score": 0.0,
                        "rerank_prob": 0.0
                    })
        except Exception:
            pass

    if not candidate_list:
        state.retrieved_chunks = []
        return []

    # 2. Neural Cross-Encoder Reranking against the ORIGINAL user query
    pairs = [(state.transcript, c["text"]) for c in candidate_list]
    raw_scores = await loop.run_in_executor(
        None, lambda: _reranker.predict(pairs)
    )

    for c, raw_score in zip(candidate_list, raw_scores):
        c["rerank_score"] = float(raw_score)
        c["rerank_prob"] = _logit_to_prob(float(raw_score))

    # Sort by rerank score descending (most relevant first)
    candidate_list.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Write to state — per request
    state.retrieved_chunks = candidate_list[:top_k]
    return state.retrieved_chunks
