"""
Request-scoped state. One instance per request. Never shared.
All pipeline stages write to this. Never use module globals.
"""
from dataclasses import dataclass, field
import uuid

@dataclass
class RequestState:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    transcript: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    query_embedding: list[float] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    final_answer: dict = field(default_factory=dict)
    # Latency per stage (ms)
    stt_ms: float = 0.0
    guardrail_ms: float = 0.0
    expand_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0

    def latency_breakdown(self) -> dict:
        self.total_ms = (
            self.guardrail_ms + self.expand_ms + self.embed_ms +
            self.retrieve_ms + self.rerank_ms + self.generate_ms
        )
        return {
            "stt_ms": self.stt_ms,
            "guardrail_ms": self.guardrail_ms,
            "query_expand_ms": self.expand_ms,
            "embed_ms": self.embed_ms,
            "retrieve_ms": self.retrieve_ms,
            "rerank_ms": self.rerank_ms,
            "generate_ms": self.generate_ms,
            "total_ms": round(self.total_ms, 2)
        }
