import sys
import os
import asyncio
import time
import numpy as np
from rich.console import Console
from rich.table import Table

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.loader import MSMARCOLoader
from dataset.chunker import AdvancedChunker
from vector_store.qdrant_engine import QdrantVectorStore
from model_harness.orchestrator import StatefulRAGOrchestrator

console = Console()

async def run_latency_benchmark(num_samples: int = 100):
    console.print("[bold green]Starting Benchmark Execution across 100 Queries...[/bold green]")

    # 1. Ingestion
    loader = MSMARCOLoader(lang="hi")
    chunker = AdvancedChunker()
    vector_store = QdrantVectorStore()

    records = list(loader.stream_records(max_records=num_samples))
    all_chunks = []
    test_queries = []

    for r in records:
        all_chunks.extend(chunker.create_parent_child_chunks(r))
        if r["query"]:
            test_queries.append(r["query"])

    vector_store.upsert_chunks(all_chunks)
    orchestrator = StatefulRAGOrchestrator(vector_store)

    test_queries = test_queries[:num_samples]
    latencies_ms = []

    console.print(f"[bold yellow]Executing RAG queries for {len(test_queries)} samples...[/bold yellow]")

    # 2. Benchmark Loop
    for idx, query in enumerate(test_queries):
        start_time = time.perf_counter()
        
        async for packet in orchestrator.execute_rag_pipeline(query=query):
            pass  # Consume full stream

        total_time_ms = (time.perf_counter() - start_time) * 1000
        latencies_ms.append(total_time_ms)
        
        if (idx + 1) % 20 == 0:
            console.print(f"Processed {idx + 1}/{len(test_queries)} queries...")

    # 3. Analytics Calculation
    p50 = float(np.percentile(latencies_ms, 50))
    p70 = float(np.percentile(latencies_ms, 70))
    p100 = float(np.max(latencies_ms))
    avg_latency = float(np.mean(latencies_ms))

    # Display Metrics Table
    table = Table(title="Voice-RAG Latency Analytics Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Target (ms)", style="magenta")
    table.add_column("Measured Latency (ms)", style="green")

    table.add_row("P50 Latency (Median)", "< 200ms", f"{p50:.2f} ms")
    table.add_row("P70 Latency", "< 200ms", f"{p70:.2f} ms")
    table.add_row("P100 Latency (Max)", "< 200ms", f"{p100:.2f} ms")
    table.add_row("Average Latency", "-", f"{avg_latency:.2f} ms")

    console.print(table)

if __name__ == "__main__":
    asyncio.run(run_latency_benchmark(100))