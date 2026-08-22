"""
pipeline/embedder.py
Multilingual-E5 Embedder enforcing mandatory 'query: ' and 'passage: ' prefixes.
"""

import os
from functools import lru_cache
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Model choice: multilingual-e5-large (1024 dims) or multilingual-e5-small (384 dims)
MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Lazy load and cache SentenceTransformer model."""
    return SentenceTransformer(MODEL_NAME)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a search query.
    CRITICAL: E5 models strictly REQUIRE the 'query: ' prefix.
    """
    model = get_model()
    clean_query = query.strip()
    return model.encode(f"query: {clean_query}", normalize_embeddings=True)


def embed_passage(passage: str) -> np.ndarray:
    """
    Embed a single document passage.
    CRITICAL: E5 models strictly REQUIRE the 'passage: ' prefix.
    """
    model = get_model()
    clean_passage = passage.strip()
    return model.encode(f"passage: {clean_passage}", normalize_embeddings=True)


def embed_passages_batch(passages: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Batch embed passages for indexing. Always prefixes with 'passage: '.
    """
    model = get_model()
    prefixed = [f"passage: {p.strip()}" for p in passages]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


class E5Embedder:
    """Compatibility wrapper class for pipeline usage."""
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = get_model()
        self.vector_dim = self.model.get_sentence_embedding_dimension()

    def encode_query(self, query: str) -> List[float]:
        return embed_query(query).tolist()

    async def encode_query_async(self, query: str) -> List[float]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_query, query)

    def encode_passages(self, passages: List[str], batch_size: int = 32) -> List[List[float]]:
        embs = embed_passages_batch(passages, batch_size=batch_size)
        return embs.tolist()
