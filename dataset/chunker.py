import uuid
from typing import List, Dict, Any


class HierarchicalSemanticChunker:
    def __init__(self, child_chunk_size: int = 150, overlap: int = 30):
        self.child_chunk_size = child_chunk_size
        self.overlap = overlap

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Splits text on sentence boundaries while respecting character window limits."""
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.child_chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                # Overlap retention
                current_chunk = current_chunk[-1:] if self.overlap > 0 else []
                current_len = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_len += sentence_len

        if current_chunk:
            chunks.append(". ".join(current_chunk))

        return chunks

    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates child vector nodes linked to full parent passage contexts.
        """
        processed_chunks = []

        for record in records:
            query_id = record.get("query_id", "unknown")
            if "passage_text" in record:
                passages = [record["passage_text"]]
                is_selected = [record.get("is_selected", 0)]
            else:
                passages = record.get("translated_passages", [])
                is_selected = record.get("is_selected", [])

            for passage_idx, parent_passage in enumerate(passages):
                if not parent_passage or not str(parent_passage).strip():
                    continue

                parent_passage = str(parent_passage)
                passage_id = record.get("passage_id", f"{query_id}_{passage_idx}")
                parent_id = f"parent_{passage_id}"
                selected_flag = bool(is_selected[passage_idx]) if passage_idx < len(is_selected) else False

                # Generate child chunks for high-resolution vector lookup
                child_texts = self.split_text_into_chunks(parent_passage)

                for child_idx, child_text in enumerate(child_texts):
                    processed_chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "parent_id": parent_id,
                        "child_text": child_text,
                        "parent_text": parent_passage,
                        "metadata": {
                            "query_id": query_id,
                            "passage_index": passage_idx,
                            "child_index": child_idx,
                            "is_selected": selected_flag,
                            "query_type": record.get("query_type"),
                            "target_lang": record.get("lang") or record.get("target_lang")
                        }
                    })

        print(f"[Chunker] Processed {len(records)} records into {len(processed_chunks)} hierarchical chunks.")
        return processed_chunks