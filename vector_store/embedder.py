import time
from typing import List
from sentence_transformers import SentenceTransformer


class LocalFastEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        print(f"[Embedder] Initializing {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings in batch mode."""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding for a single incoming voice query."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        return embedding.tolist()