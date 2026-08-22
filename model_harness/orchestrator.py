import time
import logging
from typing import Dict, Any, AsyncGenerator
from pydantic import BaseModel

from vector_store.qdrant_engine import QdrantVectorEngine
from model_harness.nemotron_client import NemotronClient
from model_harness.guardrails import PipelineGuardrails

logger = logging.getLogger("PipelineOrchestrator")

class RAGResponseSchema(BaseModel):
    query: str
    transcription: str
    answer: str
    retrieved_contexts: list
    latency_ms: float
    grounded: bool

class StatefulRAGOrchestrator:
    def __init__(self, vector_store: QdrantVectorEngine, embed_model):
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.nemotron_client = NemotronClient()
        self.guardrails = PipelineGuardrails()

    async def execute_rag_pipeline(
        self, 
        query: str, 
        top_k: int = 3, 
        max_retries: int = 2
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.perf_counter()
        
        # 1. Input Guardrail Verification
        is_valid, message = self.guardrails.validate_input(query)
        if not is_valid:
            yield {
                "type": "error",
                "content": message,
                "latency_ms": (time.perf_counter() - start_time) * 1000
            }
            return

        # 2. Vector Store Retrieval
        retrieval_start = time.perf_counter()
        query_vector = self.embed_model.encode(query).tolist()
        retrieved_docs = self.vector_store.search(query_vector=query_vector, top_k=top_k)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000
        logger.info(f"Retrieved {len(retrieved_docs)} contexts in {retrieval_time:.2f}ms")

        contexts = [doc["parent_context"] for doc in retrieved_docs]

        # 3. Model Generation with Retry Logic
        attempts = 0
        generated_answer = ""
        
        while attempts < max_retries:
            try:
                async for token in self.nemotron_client.generate_stream(query, contexts):
                    generated_answer += token
                    yield {
                        "type": "token",
                        "content": token
                    }
                break
            except Exception as e:
                attempts += 1
                logger.warning(f"Generation attempt {attempts} failed: {str(e)}")
                if attempts >= max_retries:
                    yield {
                        "type": "error",
                        "content": "Model service temporarily unavailable.",
                        "latency_ms": (time.perf_counter() - start_time) * 1000
                    }
                    return

        # 4. Grounding Check & Final Metadata
        is_grounded = self.guardrails.verify_grounding(generated_answer, contexts)
        total_latency = (time.perf_counter() - start_time) * 1000

        yield {
            "type": "final",
            "metadata": RAGResponseSchema(
                query=query,
                transcription=query,
                answer=generated_answer,
                retrieved_contexts=contexts,
                latency_ms=total_latency,
                grounded=is_grounded
            ).model_dump()
        }