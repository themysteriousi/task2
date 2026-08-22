import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


class QdrantVectorEngine:
    def __init__(self, collection_name: str = "msmarco_indic_rag"):
        self.client = QdrantClient(":memory:")  # High-performance local RAM instance
        self.collection_name = collection_name
        self.vector_size = 384  # Standard size for bge-small / MiniLM

    def init_collection(self, vector_size: int = 384):
        self.vector_size = vector_size
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        print(f"[VectorDB] Initialized collection '{self.collection_name}' in RAM.")

    def index_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "parent_id": chunk["parent_id"],
                        "child_text": chunk["child_text"],
                        "parent_text": chunk["parent_text"],
                        **chunk["metadata"]
                    }
                )
            )
            
        # Batch upload
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[VectorDB] Indexed {len(points)} vectors successfully.")

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        
        retrieved_contexts = []
        for res in results.points:
            payload = res.payload or {}
            retrieved_contexts.append({
                "score": res.score,
                "child_text": payload.get("child_text"),
                "parent_context": payload.get("parent_text"),
                "query_id": payload.get("query_id")
            })
        return retrieved_contexts