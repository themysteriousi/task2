import re
import logging
from typing import List, Tuple

logger = logging.getLogger("GuardrailEngine")

class PipelineGuardrails:
    def __init__(self):
        # Basic pattern checks for prompt injection or system exploitation
        self.blocked_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"drop database",
            r"script.*src="
        ]

    def validate_input(self, query: str) -> Tuple[bool, str]:
        """
        Validates query for unsafe input patterns and length constraints.
        """
        if not query or len(query.strip()) < 3:
            return False, "Query is too short or empty."

        for pattern in self.blocked_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Blocked malicious or off-topic pattern in query: {query}")
                return False, "Query violated safety guidelines."

        return True, "Valid"

    def verify_grounding(self, response_text: str, retrieved_contexts: List[str]) -> bool:
        """
        Lightweight lexical verification pass to ensure output contains key terms from retrieved context.
        """
        if not retrieved_contexts or "do not know" in response_text.lower():
            return True

        combined_context = " ".join(retrieved_contexts).lower()
        response_words = set(re.findall(r'\w+', response_text.lower()))
        
        # Filter common stop words
        stop_words = {"the", "is", "at", "which", "and", "a", "an", "to", "in", "for", "of", "on", "with"}
        keywords = response_words - stop_words

        if not keywords:
            return True

        # Check overlap percentage
        matched_words = [word for word in keywords if word in combined_context]
        overlap_ratio = len(matched_words) / len(keywords)

        logger.info(f"Context grounding overlap ratio: {overlap_ratio:.2f}")
        return overlap_ratio >= 0.25