import logging
from typing import Dict, Any, Generator, List, Union, Optional
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetLoader")

# Mapping short language codes to Hugging Face Parquet file paths
LANG_FILES = {
    "hi": "train/hintrain.parquet",
    "bn": "train/bentrain.parquet",
    "gu": "train/gujtrain.parquet",
    "mr": "train/martrain.parquet",
    "ta": "train/tamtrain.parquet",
    "te": "train/teltrain.parquet",
    "kn": "train/kantrain.parquet",
    "ml": "train/maltrain.parquet",
    "pa": "train/pantrain.parquet",
    "or": "train/oritrain.parquet",
    "as": "train/asmtrain.parquet",
    "ne": "train/neptrain.parquet",
    "sa": "train/santrain.parquet",
    "ur": "train/urdtrain.parquet",
    "en": "train/hintrain.parquet",  # English is embedded in every Indic file
}

class MSMARCOLoader:
    def __init__(self, langs: Union[str, List[str]] = "hi", lang: Optional[str] = None):
        """
        :param langs: Single lang code 'hi', list of codes ['hi', 'bn'], or 'all'.
        :param lang: Alternative single lang parameter for backwards compatibility.
        """
        chosen = lang if lang is not None else langs
        if chosen == "all":
            self.selected_langs = list(LANG_FILES.keys())
        elif isinstance(chosen, str):
            self.selected_langs = [chosen]
        else:
            self.selected_langs = chosen

    def stream_records(self, max_records: int = 1000) -> Generator[Dict[str, Any], None, None]:
        import os
        count = 0

        for lang in self.selected_langs:
            if count >= max_records:
                break

            filename = LANG_FILES.get(lang)
            if not filename:
                logger.warning(f"Language code '{lang}' not recognized. Skipping.")
                continue

            # Check if file exists locally in ./data/ first
            local_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", filename)
            if os.path.exists(local_data_path):
                local_path = local_data_path
                logger.info(f"Loading '{lang}' from local project directory: {local_path}")
            else:
                logger.info(f"Loading '{lang}' via Hugging Face local cache ({filename})...")
                local_path = hf_hub_download(
                    repo_id="ai4bharat/MSMARCO-XI",
                    filename=filename,
                    repo_type="dataset"
                )
                logger.info(f"Successfully loaded '{lang}' from: {local_path}")
            parquet_file = pq.ParquetFile(local_path)

            for batch in parquet_file.iter_batches(batch_size=128):
                if count >= max_records:
                    break

                batch_dict = batch.to_pydict()
                num_rows = len(batch_dict.get("query_id", []))

                for i in range(num_rows):
                    if count >= max_records:
                        break

                    passages_data = batch_dict["passages"][i] if "passages" in batch_dict else {}
                    translated_passages = passages_data.get("Translated_passages", []) if passages_data else []
                    english_passages = passages_data.get("English_passages", []) if passages_data else []
                    is_selected_flags = passages_data.get("is_selected", []) if passages_data else []

                    for idx, passage in enumerate(translated_passages):
                        yield {
                            "query_id": batch_dict["query_id"][i],
                            "query_type": batch_dict.get("query_type", ["general"] * num_rows)[i],
                            "query": batch_dict.get("query", [""] * num_rows)[i],
                            "passage_id": f"{batch_dict['query_id'][i]}_{idx}",
                            "passage_text": passage,
                            "eng_passage_text": english_passages[idx] if idx < len(english_passages) else "",
                            "is_selected": is_selected_flags[idx] if idx < len(is_selected_flags) else 0,
                            "lang": lang
                        }
                    count += 1