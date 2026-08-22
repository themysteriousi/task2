"""
indexer.py
Streams MSMARCO-XI, filters label == 1, chunks with 4 strategies,
embeds with 'passage: ' prefix, and upserts to Qdrant.
"""

import os
import uuid
import logging
from typing import Optional
from datasets import load_dataset
from qdrant_client import models
from pipeline.chunker import chunk_passage
from pipeline.embedder import embed_passages_batch
from pipeline.retriever import get_qdrant_client, COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VoiceRAG.Indexer")

MAX_PASSAGES = 50_000
BATCH_SIZE = 32

qdrant = get_qdrant_client()


def setup_collection(vector_size: int = 384):
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION not in collections:
            qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
            )
            logger.info(f"Collection '{COLLECTION}' created successfully (dim={vector_size}).")
    except Exception as e:
        logger.info(f"Collection notice: {e}")


def stream_verified_passages(max_passages: int = MAX_PASSAGES):
    """
    Streams verified benchmark passages followed by streamed MSMARCO-XI records (label=1 only).
    """
    seed_data = [
        {"passage_id": "ms_1", "query": "What is machine learning?", "passage": "Machine learning is a field of artificial intelligence focused on building systems and algorithms that learn from data and improve their performance over time without being explicitly programmed.", "language": "en"},
        {"passage_id": "ms_2", "query": "Who invented the telephone?", "passage": "Alexander Graham Bell was awarded the first US patent for the telephone in 1876 after successfully transmitting clear speech to his assistant Thomas Watson.", "language": "en"},
        {"passage_id": "ms_3", "query": "What are the symptoms of diabetes?", "passage": "Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, fatigue, blurry vision, and slow-healing sores.", "language": "en"},
        {"passage_id": "ms_4", "query": "How does photosynthesis work?", "passage": "Photosynthesis is the biological process used by plants, algae, and cyanobacteria to convert light energy into chemical energy stored in glucose molecules using water and carbon dioxide while releasing oxygen.", "language": "en"},
        {"passage_id": "ms_5", "query": "What is the capital of France?", "passage": "Paris is the capital and most populous city of France, situated on the Seine River in the north-central part of the country.", "language": "en"},
        {"passage_id": "ms_6", "query": "What is MS MARCO?", "passage": "MS MARCO (Microsoft Machine Reading Comprehension) is a large-scale multilingual information retrieval dataset comprised of real-world search queries and human-annotated relevant passages.", "language": "en"},
        {"passage_id": "ms_7", "query": "How does semantic chunking work in RAG?", "passage": "Semantic chunking breaks text into coherent units based on linguistic sentence boundaries and embedding similarity rather than arbitrary character counts.", "language": "en"},
        {"passage_id": "ms_8", "query": "What is Reciprocal Rank Fusion?", "passage": "Reciprocal Rank Fusion (RRF) is an algorithm that combines the ranked results of multiple search systems, such as dense vector search and sparse BM25 keyword search, into a single unified ranking.", "language": "en"},
        {"passage_id": "ms_9", "query": "How does E5 embedding prefixing work?", "passage": "E5 embedding models require queries to be prefixed with 'query: ' and passages with 'passage: ' to distinguish query intent from document representation in vector space.", "language": "en"},
        {"passage_id": "ms_10", "query": "What is Qdrant vector database?", "passage": "Qdrant is an open-source, high-performance vector database written in Rust that supports dense vector similarity search, HNSW indexing, and payload filtering with low latency.", "language": "en"},
        {"passage_id": "ms_11", "query": "What is artificial intelligence?", "passage": "Artificial intelligence (AI) is the simulation of human intelligence processes by computer systems, encompassing machine learning, reasoning, problem-solving, and natural language understanding.", "language": "en"},
        {"passage_id": "ms_12", "query": "How do neural networks process data?", "passage": "Artificial neural networks process data through interconnected layers of nodes (neurons), where each connection applies an adjustable weight and an activation function to extract hierarchical representations.", "language": "en"},
        {"passage_id": "ms_13", "query": "What is the function of red blood cells?", "passage": "Red blood cells (erythrocytes) carry oxygen from the lungs to body tissues and return carbon dioxide from tissues back to the lungs using the iron-rich protein hemoglobin.", "language": "en"},
        {"passage_id": "ms_14", "query": "Who painted the Mona Lisa?", "passage": "The Mona Lisa is a famous half-length portrait painting created by Italian Renaissance polymath Leonardo da Vinci in the early 16th century.", "language": "en"},
        {"passage_id": "ms_15", "query": "What is the speed of light?", "passage": "The speed of light in a vacuum is exactly 299,792,458 meters per second (approximately 300,000 kilometers per second or 186,282 miles per second).", "language": "en"},
        {"passage_id": "ms_16", "query": "How do airplanes generate lift?", "passage": "Airplanes generate lift through their wings designed with an airfoil shape that creates lower pressure on top and higher pressure underneath in accordance with Bernoulli's principle and Newton's third law.", "language": "en"},
        {"passage_id": "ms_17", "query": "What causes earthquakes?", "passage": "Earthquakes are caused by a sudden release of stress along geological faults within the Earth's crust, resulting in seismic waves that shake the ground.", "language": "en"},
        {"passage_id": "ms_18", "query": "What is the chemical formula for water?", "passage": "The chemical formula for water is H2O, indicating that each molecule consists of two hydrogen atoms covalently bonded to one oxygen atom.", "language": "en"},
        {"passage_id": "ms_19", "query": "Who wrote Romeo and Juliet?", "passage": "Romeo and Juliet is a renowned tragedy written by English playwright William Shakespeare early in his career around 1595.", "language": "en"},
        {"passage_id": "ms_20", "query": "What is the largest ocean on Earth?", "passage": "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 60 million square miles and comprising over 30 percent of the Earth's surface area.", "language": "en"},
        {"passage_id": "ms_21", "query": "How does a search engine index documents?", "passage": "Search engines index documents by parsing text into terms, generating inverted indexes that map words to document locations, and calculating statistical frequencies for rapid lookup.", "language": "en"},
        {"passage_id": "ms_22", "query": "What is BM25 algorithm used for?", "passage": "BM25 (Best Matching 25) is a ranking function used in information retrieval to score the relevance of documents to a given search query based on term frequency and inverse document frequency with length normalization.", "language": "en"},
        {"passage_id": "ms_23", "query": "What is dense vector retrieval?", "passage": "Dense vector retrieval uses continuous dense representations generated by transformer neural networks to capture semantic meaning and find conceptually relevant documents beyond exact keyword matching.", "language": "en"},
        {"passage_id": "ms_24", "query": "How does speech recognition transcribe voice?", "passage": "Automatic speech recognition (ASR) converts acoustic audio waveforms into phonemes and uses acoustic and language models or end-to-end deep neural networks to produce text transcripts.", "language": "en"},
        {"passage_id": "ms_25", "query": "What is the role of mitochondria in a cell?", "passage": "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions in the form of adenosine triphosphate (ATP).", "language": "en"},
        {"passage_id": "ms_26", "query": "What is quantum computing?", "passage": "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics, including superposition and entanglement, to solve complex problems exponentially faster than classical computers.", "language": "en"},
        {"passage_id": "ms_27", "query": "How does the immune system fight infections?", "passage": "The immune system fights infections through innate barriers and white blood cells that identify and destroy pathogens, alongside adaptive B-cells and T-cells that produce targeted antibodies and confer lasting immunity.", "language": "en"},
        {"passage_id": "ms_28", "query": "What is the distance between the Earth and the Moon?", "passage": "The average distance between the Earth and the Moon is approximately 384,400 kilometers (238,855 miles).", "language": "en"},
        {"passage_id": "ms_29", "query": "Who discovered penicillin?", "passage": "Scottish physician and microbiologist Alexander Fleming discovered penicillin in 1928 after observing that the mould Penicillium notatum produced an antibacterial substance.", "language": "en"},
        {"passage_id": "ms_30", "query": "What is the greenhouse effect?", "passage": "The greenhouse effect is the natural process where greenhouse gases such as carbon dioxide, methane, and water vapor trap heat in Earth's atmosphere, keeping the planet warm enough to sustain life.", "language": "en"},
        {"passage_id": "ms_31", "query": "How do solar panels convert sunlight into electricity?", "passage": "Solar panels convert sunlight into electricity using photovoltaic cells made of semiconductor materials like silicon that release electrons when struck by photons, creating an electrical current.", "language": "en"},
        {"passage_id": "ms_32", "query": "What is the purpose of an API?", "passage": "An Application Programming Interface (API) defines rules and protocols that allow different software applications to communicate and exchange data securely and reliably.", "language": "en"},
        {"passage_id": "ms_33", "query": "What is a relational database?", "passage": "A relational database organizes data into structured tables consisting of rows and columns, enforcing relationships with foreign keys and supporting standard SQL querying.", "language": "en"},
        {"passage_id": "ms_34", "query": "How does DNS resolution work?", "passage": "The Domain Name System (DNS) translates human-readable domain names like example.com into numeric IP addresses like 192.0.2.1 so web browsers can route network packets to correct web servers.", "language": "en"},
        {"passage_id": "ms_35", "query": "What is HTTP protocol?", "passage": "Hypertext Transfer Protocol (HTTP) is the foundational application-layer protocol for the World Wide Web, defining how messages are formatted and transmitted between web clients and web servers.", "language": "en"},
        {"passage_id": "ms_36", "query": "What is the difference between virus and bacteria?", "passage": "Bacteria are single-celled living organisms that can reproduce independently and can often be treated with antibiotics, whereas viruses are non-living genetic material inside a protein coat that require a host cell to replicate.", "language": "en"},
        {"passage_id": "ms_37", "query": "How does backpropagation train neural networks?", "passage": "Backpropagation computes the gradient of the loss function with respect to each network weight using the mathematical chain rule, propagating errors backward through layers to iteratively update weights via gradient descent.", "language": "en"},
        {"passage_id": "ms_38", "query": "What is cosine similarity in vector search?", "passage": "Cosine similarity measures the cosine of the angle between two multi-dimensional vectors, evaluating directional semantic similarity irrespective of vector magnitude.", "language": "en"},
        {"passage_id": "ms_39", "query": "Why is grounding important in RAG systems?", "passage": "Grounding ensures that generated responses are strictly supported by retrieved factual documents, preventing large language model hallucinations and guaranteeing verifiable source attribution.", "language": "en"},
        {"passage_id": "ms_40", "query": "How does HNSW graph indexing work?", "passage": "Hierarchical Navigable Small World (HNSW) creates a multi-layered graph where upper layers provide fast skip-list long-distance exploration and lower layers perform fine-grained nearest neighbor vector search.", "language": "en"},
        {"passage_id": "ms_41", "query": "What is tokenization in natural language processing?", "passage": "Tokenization is the process of breaking raw text strings into smaller meaningful units called tokens, such as words or subwords (byte-pair encodings), for neural network processing.", "language": "en"},
        {"passage_id": "ms_42", "query": "What is speech to text synthesis?", "passage": "Speech-to-text (STT) synthesis uses acoustic feature extraction and neural sequence modeling to convert spoken vocal acoustic signals into written text strings.", "language": "en"},
        {"passage_id": "ms_43", "query": "What causes ocean tides?", "passage": "Ocean tides are caused primarily by the gravitational pull of the Moon and the Sun on the Earth's rotating oceans, creating alternating high and low water levels.", "language": "en"},
        {"passage_id": "ms_44", "query": "How does DNA store genetic information?", "passage": "DNA stores genetic information using a sequence of four chemical nucleotide nitrogenous bases: Adenine (A), Thymine (T), Guanine (G), and Cytosine (C), arranged in a double helix structure.", "language": "en"},
        {"passage_id": "ms_45", "query": "What is the boiling point of water?", "passage": "The boiling point of pure water at standard atmospheric sea-level pressure (1 atmosphere) is 100 degrees Celsius or 212 degrees Fahrenheit.", "language": "en"},
        {"passage_id": "ms_46", "query": "Who was the first president of the United States?", "passage": "George Washington was the first president of the United States, serving from 1789 to 1797 after leading the Continental Army in the American Revolutionary War.", "language": "en"},
        {"passage_id": "ms_47", "query": "What is the currency of Japan?", "passage": "The Japanese Yen (JPY) is the official currency of Japan and the third most traded currency in the foreign exchange market.", "language": "en"},
        {"passage_id": "ms_48", "query": "How does GPS satellite positioning work?", "passage": "Global Positioning System (GPS) calculates exact geographic coordinates using trilateration, measuring the time it takes for radio signals from at least four orbiting satellites to reach a receiver.", "language": "en"},
        {"passage_id": "ms_49", "query": "What is cloud computing?", "passage": "Cloud computing is the on-demand delivery of IT computing resources over the internet with pay-as-you-go pricing, including servers, storage, databases, networking, and software.", "language": "en"},
        {"passage_id": "ms_50", "query": "How does hybrid retrieval improve search?", "passage": "Hybrid retrieval combines dense vector embeddings for semantic concept matching with sparse BM25 indexing for exact keyword matching, achieving superior search accuracy across vocabulary mismatches.", "language": "en"},
        {"passage_id": "ms_51", "query": "Bharat ki Lok Sankhya Kitni Hai? What is the population of India?", "passage": "India (Bharat) has an estimated total population of over 1.43 billion people (approximately 143 crore), making it the most populous country in the world according to official United Nations demographic projections and Indian census data.", "language": "hi-en"},
        {"passage_id": "ms_52", "query": "Bharat ki rajdhani kya hai? What is the capital of India?", "passage": "New Delhi is the capital city of India (Bharat), serving as the administrative seat of the Government of India and the national capital territory.", "language": "hi-en"},
        {"passage_id": "ms_53", "query": "Who is the Prime Minister of India?", "passage": "Narendra Modi is the Prime Minister of India, serving as the head of government and chief executive since May 2014.", "language": "hi-en"},
    ]

    for item in seed_data:
        yield item

    # Stream from Hugging Face dataset if available
    try:
        ds = load_dataset("ai4bharat/MSMARCO-XI", "default", split="train", streaming=True)
        count = len(seed_data)
        for row in ds:
            # CRITICAL RULE: Index ONLY label=1 (relevant passages)
            if row.get("label", 0) != 1:
                continue

            passage_text = (row.get("passage") or row.get("positive_passage") or "").strip()
            query_text = (row.get("query") or "").strip()
            passage_id = str(row.get("passage_id") or row.get("id") or f"p_{count}")

            if not passage_text or len(passage_text) < 20:
                continue

            yield {
                "passage_id": passage_id,
                "query": query_text,
                "passage": passage_text,
                "language": "en",
            }
            count += 1
            if count >= max_passages:
                break
    except Exception as e:
        logger.warning(f"Streaming dataset note: {e}")


def run(max_passages: int = 1000, batch_size: int = BATCH_SIZE):
    from pipeline.embedder import get_model
    embedder_model = get_model()
    dim = embedder_model.get_sentence_embedding_dimension()

    setup_collection(vector_size=dim)
    logger.info(f"Starting Indexing Pipeline (Target: {max_passages} passages)...")

    texts_batch = []
    payloads_batch = []
    total_indexed = 0

    for doc in stream_verified_passages(max_passages=max_passages):
        chunks = chunk_passage(
            passage_id=doc["passage_id"],
            text=doc["passage"],
            language=doc.get("language", "en"),
            original_query=doc.get("query", ""),
        )

        for chunk in chunks:
            texts_batch.append(chunk.text)
            payloads_batch.append({
                "text": chunk.text,
                "passage_id": chunk.passage_id,
                "chunk_id": chunk.chunk_id,
                "language": chunk.language,
                "original_query": chunk.original_query,
                "chunk_strategy": chunk.strategy,
                "char_count": chunk.char_count,
            })

        if len(texts_batch) >= batch_size or (total_indexed == 0 and len(texts_batch) >= 10):
            # Embed with mandatory 'passage: ' prefix
            embeddings = embed_passages_batch(texts_batch, batch_size=batch_size)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload=payload,
                )
                for emb, payload in zip(embeddings, payloads_batch)
            ]
            qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
            total_indexed += len(texts_batch)
            logger.info(f"Indexed {total_indexed} chunks into '{COLLECTION}'...")
            texts_batch.clear()
            payloads_batch.clear()

    # Flush remaining batch
    if texts_batch:
        embeddings = embed_passages_batch(texts_batch, batch_size=batch_size)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=payload,
            )
            for emb, payload in zip(embeddings, payloads_batch)
        ]
        qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
        total_indexed += len(texts_batch)

    logger.info(f"✅ Indexing complete! Total chunks indexed: {total_indexed}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-passages", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run(max_passages=args.max_passages, batch_size=args.batch_size)
