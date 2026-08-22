"""
pipeline/latency.py
Stage-level latency tracking, profiling, and P50/P70/P100 percentile calculation.
"""

import time
import statistics
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Any, List, Optional


@dataclass
class PipelineTrace:
    query_id: str
    stt_ms: float = 0.0
    guardrail_input_ms: float = 0.0
    query_expand_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    guardrail_output_ms: float = 0.0
    total_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "stt_ms": self.stt_ms,
            "guardrail_input_ms": self.guardrail_input_ms,
            "query_expand_ms": self.query_expand_ms,
            "embed_ms": self.embed_ms,
            "retrieve_ms": self.retrieve_ms,
            "rerank_ms": self.rerank_ms,
            "generate_ms": self.generate_ms,
            "guardrail_output_ms": self.guardrail_output_ms,
            "total_ms": self.total_ms,
            "extra": self.extra,
        }


class LatencyTracker:
    def __init__(self, window: int = 500):
        self._traces: deque[PipelineTrace] = deque(maxlen=window)

    def record(self, trace: PipelineTrace):
        self._traces.append(trace)

    def record_trace(self, trace: PipelineTrace):
        self.record(trace)

    def percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_v = sorted(values)
        idx = int(len(sorted_v) * p / 100.0)
        return round(sorted_v[min(idx, len(sorted_v) - 1)], 2)

    def report(self) -> Dict[str, Any]:
        if not self._traces:
            return {
                "total_queries": 0,
                "overall": {"P50": 0.0, "P70": 0.0, "P100": 0.0, "mean": 0.0},
                "per_stage": {},
            }
        totals = [t.total_ms for t in self._traces]
        stages = {
            "stt": [t.stt_ms for t in self._traces if t.stt_ms > 0],
            "guardrail_input": [t.guardrail_input_ms for t in self._traces],
            "query_expansion": [t.query_expand_ms for t in self._traces],
            "embedding": [t.embed_ms for t in self._traces],
            "retrieval": [t.retrieve_ms for t in self._traces],
            "generation": [t.generate_ms for t in self._traces],
            "guardrail_output": [t.guardrail_output_ms for t in self._traces],
        }
        return {
            "total_queries": len(self._traces),
            "overall": {
                "P50": self.percentile(totals, 50),
                "P70": self.percentile(totals, 70),
                "P100": max(totals) if totals else 0.0,
                "mean": round(statistics.mean(totals), 2) if totals else 0.0,
            },
            "per_stage": {
                stage: {
                    "P50": self.percentile(vals, 50),
                    "P70": self.percentile(vals, 70),
                    "P100": max(vals) if vals else 0.0,
                }
                for stage, vals in stages.items()
                if vals
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        return self.report()


tracker = LatencyTracker()
latency_tracker = tracker


class Timer:
    def __init__(self):
        self._start = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
