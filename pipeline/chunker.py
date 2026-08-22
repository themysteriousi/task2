"""
pipeline/chunker.py
Multi-strategy chunker implementing:
Strategy A (native), Strategy B (sentence-split), Strategy C (fixed-overlap), Strategy D (semantic-merge).
"""

from dataclasses import dataclass
import re
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    chunk_id: str
    passage_id: str
    text: str
    strategy: str        # "native" | "sentence" | "fixed_overlap" | "semantic_merge"
    language: str
    original_query: str  # query this passage was labeled relevant for
    char_count: int
    score: float = 0.0   # filled in after retrieval / reranking

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "passage_id": self.passage_id,
            "text": self.text,
            "strategy": self.strategy,
            "language": self.language,
            "original_query": self.original_query,
            "char_count": self.char_count,
            "score": self.score,
        }


def select_strategy(text: str, char_count: int) -> str:
    """Select appropriate chunking strategy based on passage length."""
    if char_count <= 400:
        return "native"        # MSMARCO passages average ~300-400 chars: use as-is
    if char_count <= 800:
        return "sentence"      # 2-3 sentence split with overlap
    if char_count <= 1500:
        return "fixed_overlap" # 150-word window with 30-word overlap
    return "semantic_merge"    # Group by embedding topic shifts


def chunk_passage(
    passage_id: str,
    text: str,
    language: str = "en",
    original_query: str = "",
) -> List[Chunk]:
    """Apply the right strategy per passage length and characteristics."""
    char_count = len(text.strip())
    if char_count == 0:
        return []

    strategy = select_strategy(text, char_count)

    # Strategy A: Native passage
    if strategy == "native":
        return [
            Chunk(
                chunk_id=f"{passage_id}_0",
                passage_id=passage_id,
                text=text.strip(),
                strategy="native",
                language=language,
                original_query=original_query,
                char_count=char_count,
            )
        ]

    # Strategy B: Sentence boundary with 1-sentence overlap
    elif strategy == "sentence":
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 15]
        if len(sentences) <= 2:
            return [
                Chunk(
                    chunk_id=f"{passage_id}_0",
                    passage_id=passage_id,
                    text=text.strip(),
                    strategy="native",
                    language=language,
                    original_query=original_query,
                    char_count=char_count,
                )
            ]
        chunks: List[Chunk] = []
        i = 0
        while i < len(sentences):
            group = sentences[i : i + 3]
            chunk_text = " ".join(group)
            chunks.append(
                Chunk(
                    chunk_id=f"{passage_id}_{i}",
                    passage_id=passage_id,
                    text=chunk_text,
                    strategy="sentence",
                    language=language,
                    original_query=original_query,
                    char_count=len(chunk_text),
                )
            )
            i += 2  # 1-sentence overlap
        return chunks if chunks else [
            Chunk(
                chunk_id=f"{passage_id}_0",
                passage_id=passage_id,
                text=text.strip(),
                strategy="native",
                language=language,
                original_query=original_query,
                char_count=char_count,
            )
        ]

    # Strategy C: Fixed 150-word window with 30-word overlap
    elif strategy == "fixed_overlap":
        words = text.split()
        chunk_size, overlap = 150, 30
        chunks: List[Chunk] = []
        for i, start in enumerate(range(0, len(words), chunk_size - overlap)):
            chunk_words = words[start : start + chunk_size]
            if len(chunk_words) < 10:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append(
                Chunk(
                    chunk_id=f"{passage_id}_{i}",
                    passage_id=passage_id,
                    text=chunk_text,
                    strategy="fixed_overlap",
                    language=language,
                    original_query=original_query,
                    char_count=len(chunk_text),
                )
            )
        return chunks if chunks else [
            Chunk(
                chunk_id=f"{passage_id}_0",
                passage_id=passage_id,
                text=text.strip(),
                strategy="native",
                language=language,
                original_query=original_query,
                char_count=char_count,
            )
        ]

    # Strategy D: Semantic Merge based on sentence embedding topic shift
    else:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 15]
        if len(sentences) < 4:
            return [
                Chunk(
                    chunk_id=f"{passage_id}_0",
                    passage_id=passage_id,
                    text=text.strip(),
                    strategy="native",
                    language=language,
                    original_query=original_query,
                    char_count=char_count,
                )
            ]

        from pipeline.embedder import get_model
        model = get_model()
        embs = model.encode(sentences, normalize_embeddings=True)
        sims = [
            cosine_similarity([embs[i]], [embs[i + 1]])[0][0]
            for i in range(len(embs) - 1)
        ]
        threshold = float(np.percentile(sims, 25)) if sims else 0.5

        chunks_text: List[str] = []
        group: List[str] = [sentences[0]]
        for i, sim in enumerate(sims):
            if sim < threshold and len(group) >= 2:
                chunks_text.append(" ".join(group))
                group = []
            group.append(sentences[i + 1])
        if group:
            chunks_text.append(" ".join(group))

        return [
            Chunk(
                chunk_id=f"{passage_id}_{i}",
                passage_id=passage_id,
                text=c,
                strategy="semantic_merge",
                language=language,
                original_query=original_query,
                char_count=len(c),
            )
            for i, c in enumerate(chunks_text)
        ]


class MultiStrategyChunker:
    """Wrapper class for pipeline indexing."""
    def chunk_record(self, passage_text: str, passage_id: str, original_query: str = "", language: str = "en") -> List[Chunk]:
        return chunk_passage(passage_id, passage_text, language=language, original_query=original_query)
