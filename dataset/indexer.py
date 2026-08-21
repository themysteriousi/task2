import os
import sys
import logging
from typing import List, Optional

# Dynamically add the project root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dataset.loader import MSMARCOLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorIndexer")

INDEX_DIR = "vectorstore"
EMBED_MODEL_NAME = "BAAI/bge-m3"

class VectorIndexer:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        logger.info(f"Initializing embedding model '{EMBED_MODEL_NAME}'...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store: Optional[FAISS] = None

    def build_index(self, lang: str = "hi", max_records: int = 1000, batch_size: int = 256):
        """Streams dataset records, creates embeddings, and builds a FAISS index."""
        loader = MSMARCOLoader(langs=lang)
        documents: List[Document] = []
        
        logger.info(f"Extracting up to {max_records} passage records for language '{lang}'...")
        
        for record in loader.stream_records(max_records=max_records):
            doc = Document(
                page_content=record["passage_text"],
                metadata={
                    "query_id": record["query_id"],
                    "query": record["query"],
                    "eng_text": record["eng_passage_text"],
                    "is_selected": record["is_selected"],
                    "lang": record["lang"],
                    "passage_id": record["passage_id"]
                }
            )
            documents.append(doc)

            if len(documents) >= batch_size:
                self._add_documents(documents)
                documents = []

        if documents:
            self._add_documents(documents)

        # Save index locally
        os.makedirs(self.index_dir, exist_ok=True)
        self.vector_store.save_local(self.index_dir)
        logger.info(f"FAISS vector store successfully saved to '{self.index_dir}'")

    def _add_documents(self, docs: List[Document]):
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)

    def load_index(self) -> FAISS:
        """Loads an existing FAISS index from disk."""
        if not os.path.exists(self.index_dir):
            raise FileNotFoundError(f"Vector store directory '{self.index_dir}' does not exist. Build the index first.")
        
        logger.info(f"Loading FAISS vector store from '{self.index_dir}'...")
        self.vector_store = FAISS.load_local(
            self.index_dir, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        return self.vector_store

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """Performs semantic similarity search against indexed passages."""
        if self.vector_store is None:
            self.load_index()
        return self.vector_store.similarity_search(query, k=top_k)


if __name__ == "__main__":
    indexer = VectorIndexer()
    indexer.build_index(lang="hi", max_records=1000)