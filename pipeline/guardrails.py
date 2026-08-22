"""
pipeline/guardrails.py
Fast, low-latency guardrails for input sanitization and post-generation grounding.
"""

import re
from typing import Optional, Dict, Any, List

QUESTION_PATTERNS = re.compile(
    r'\b(what|who|where|when|why|how|which|is|are|was|were|does|did|'
    r'can|could|would|tell me|explain|define|describe|list|name|meaning|concept)\b',
    re.IGNORECASE,
)

UNSAFE_PATTERNS = re.compile(
    r'\b(kill|harm|hurt|bomb|weapon|suicide|abuse|illegal|hack|exploit)\b',
    re.IGNORECASE,
)


def check_input(transcript: str) -> Optional[Dict[str, Any]]:
    """
    Fast pre-retrieval guardrail (< 1ms).
    Returns error dict if blocked, None if OK to proceed.
    """
    if not transcript or len(transcript.strip()) < 3:
        return {
            "status": "error",
            "code": "empty_transcript",
            "message": "Could not understand audio. Please try again.",
        }

    clean_text = transcript.strip()

    if UNSAFE_PATTERNS.search(clean_text):
        return {
            "status": "blocked",
            "code": "unsafe_input",
            "message": "This query cannot be processed due to safety policies.",
        }

    if not QUESTION_PATTERNS.search(clean_text) and len(clean_text.split()) <= 2:
        return {
            "status": "blocked",
            "code": "off_topic",
            "message": "Please ask a factual question about the dataset.",
        }

    return None


def check_output(answer, chunks: List[Any]) -> Optional[Dict[str, Any]]:
    """
    Post-generation guardrail (< 2ms).
    Enforces grounding and confidence threshold.
    """
    if getattr(answer, "no_answer", False) or not getattr(answer, "grounded", True):
        return {
            "status": "no_answer",
            "answer": "The dataset does not contain a reliable answer to your question.",
            "source_passages": [c.text for c in chunks[:3]] if chunks else [],
            "confidence": 0.0,
        }

    confidence = getattr(answer, "confidence", 1.0)
    if confidence < 0.35:
        return {
            "status": "low_confidence",
            "answer": f"[Low confidence] {getattr(answer, 'answer', '')}",
            "source_passage": getattr(answer, "source_passage", ""),
            "confidence": confidence,
            "warning": "This answer has low confidence. Review the source passage.",
        }

    return None


class PipelineGuardrails:
    """Wrapper class for pipeline compatibility."""
    def check_pre_retrieval(self, query: str):
        err = check_input(query)
        return (err is None, err or {}, 0.01)

    def check_post_retrieval(self, answer, chunks):
        err = check_output(answer, chunks)
        return (err is None, err or {}, 0.01)
