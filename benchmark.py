"""
benchmark.py
50-Query Latency & Grounding Benchmark Harness for Voice-Powered Search Engine.
Runs 50 factual test queries against SSE /query/stream and outputs P50/P70/P100 latency reports.
"""

import asyncio
import httpx
import json
import statistics
import time

TEST_QUERIES = [
    "What is machine learning?",
    "Who invented the telephone?",
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What are the symptoms of diabetes?",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "What causes earthquakes?",
    "How does the immune system fight infections?",
    "What is quantum computing?",
    "What is the boiling point of water?",
    "Who was the first president of the United States?",
    "What is the chemical formula for water?",
    "How do solar panels convert sunlight into electricity?",
    "What is artificial intelligence?",
    "Who discovered penicillin?",
    "How does DNA store genetic information?",
    "What is the difference between virus and bacteria?",
    "What is MS MARCO?",
    "How does semantic chunking work in RAG?",
    "What is Reciprocal Rank Fusion?",
    "What is Qdrant vector database?",
    "How does E5 embedding prefixing work?",
    "What is BM25 algorithm used for?",
    "What is dense vector retrieval?",
    "How does speech recognition transcribe voice?",
    "What is the role of mitochondria in a cell?",
    "What is the distance between the Earth and the Moon?",
    "What is the greenhouse effect?",
    "What is the purpose of an API?",
    "What is a relational database?",
    "How does DNS resolution work?",
    "What is HTTP protocol?",
    "How does backpropagation train neural networks?",
    "What is cosine similarity in vector search?",
    "Why is grounding important in RAG systems?",
    "How does HNSW graph indexing work?",
    "What is tokenization in natural language processing?",
    "What is speech to text synthesis?",
    "What causes ocean tides?",
    "What is the currency of Japan?",
    "How does GPS satellite positioning work?",
    "What is cloud computing?",
    "How does hybrid retrieval improve search?",
    "How do airplanes generate lift?",
    "Who painted the Mona Lisa?",
    "What is the largest ocean on Earth?",
    "How do neural networks process data?",
    "What is the function of red blood cells?",
    "How does a search engine index documents?"
]


async def run_one(client: httpx.AsyncClient, query: str) -> float:
    start = time.perf_counter()
    total_ms = None
    async with client.stream(
        "POST",
        "http://127.0.0.1:8000/query/stream",
        json={"text": query},
        timeout=30.0,
    ) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("stage") == "pipeline" and "result" in data:
                        lb = data["result"].get("latency_breakdown", {})
                        total_ms = lb.get("total_ms")
                except Exception:
                    pass

    return total_ms or ((time.perf_counter() - start) * 1000)


async def main():
    print(f"🚀 Running {len(TEST_QUERIES)} benchmark queries against http://127.0.0.1:8000/query/stream...\n")
    latencies = []

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(TEST_QUERIES):
            ms = await run_one(client, q)
            latencies.append(ms)
            print(f"[{i+1:02d}/{len(TEST_QUERIES)}] {q[:45]:<45} → {ms:.1f}ms")

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p70 = latencies[int(n * 0.70)]
    p100 = latencies[-1]
    mean = statistics.mean(latencies)

    print(f"""
╔══════════════════════════════════════╗
║     LATENCY REPORT ({n} queries)      ║
╠══════════════════════════════════════╣
║  P50  (median):  {p50:>8.1f} ms        ║
║  P70:            {p70:>8.1f} ms        ║
║  P100 (max):     {p100:>8.1f} ms        ║
║  Mean:           {mean:>8.1f} ms        ║
╚══════════════════════════════════════╝
    """)

    # Print server-side tracker percentiles
    async with httpx.AsyncClient() as client:
        r = await client.get("http://127.0.0.1:8000/latency/report")
        print("Server-side detailed report:")
        print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
