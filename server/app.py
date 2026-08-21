import os
import time
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dataset.loader import MSMARCOLoader
from dataset.chunker import AdvancedChunker
from vector_store.qdrant_engine import QdrantVectorStore
from stt.elevenlabs_stream import ElevenLabsSTT
from model_harness.orchestrator import StatefulRAGOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceRAGServer")

app = FastAPI(title="Voice-Enabled RAG Pipeline with Nemotron")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core State
vector_store = None
orchestrator = None
stt_engine = None

@app.on_event("startup")
async def startup_event():
    global vector_store, orchestrator, stt_engine
    logger.info("Initializing vector store and loading MSMARCO dataset...")
    
    vector_store = QdrantVectorStore()
    loader = MSMARCOLoader(lang="hi")
    chunker = AdvancedChunker()

    # Index first 100 records into memory
    all_chunks = []
    for record in loader.stream_records(max_records=100):
        chunks = chunker.create_parent_child_chunks(record)
        all_chunks.extend(chunks)

    vector_store.upsert_chunks(all_chunks)
    orchestrator = StatefulRAGOrchestrator(vector_store)
    
    if os.getenv("ELEVENLABS_API_KEY"):
        stt_engine = ElevenLabsSTT()
    logger.info("Pipeline setup complete.")

class TextQueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/api/rag/text")
async def text_rag_endpoint(request: TextQueryRequest):
    start_time = time.perf_counter()
    tokens = []
    final_metadata = None

    async for packet in orchestrator.execute_rag_pipeline(request.query, top_k=request.top_k):
        if packet["type"] == "token":
            tokens.append(packet["content"])
        elif packet["type"] == "final":
            final_metadata = packet["metadata"]
        elif packet["type"] == "error":
            raise HTTPException(status_code=400, detail=packet["content"])

    return {
        "transcription": request.query,
        "answer": "".join(tokens),
        "metadata": final_metadata,
        "server_latency_ms": (time.perf_counter() - start_time) * 1000
    }

@app.websocket("/ws/voice-rag")
async def websocket_voice_rag(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            start_time = time.perf_counter()

            if not stt_engine:
                await websocket.send_json({"type": "error", "message": "ElevenLabs API key not configured."})
                continue

            # 1. ElevenLabs STT
            transcription = await stt_engine.transcribe_audio_bytes(audio_bytes)
            await websocket.send_json({"type": "transcription", "text": transcription})

            # 2. RAG Streaming Generation
            async for packet in orchestrator.execute_rag_pipeline(transcription):
                if packet["type"] == "token":
                    await websocket.send_json({"type": "token", "token": packet["content"]})
                elif packet["type"] == "final":
                    packet["metadata"]["total_voice_latency_ms"] = (time.perf_counter() - start_time) * 1000
                    await websocket.send_json({"type": "final", "data": packet["metadata"]})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket execution error: {str(e)}")
        await websocket.close()