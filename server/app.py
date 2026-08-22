import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dataset.loader import MSMARCOLoader
from dataset.chunker import HierarchicalSemanticChunker
from vector_store.qdrant_engine import QdrantVectorEngine
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

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    vector_store = QdrantVectorEngine()
    vector_store.init_collection(vector_size=384)

    loader = MSMARCOLoader()
    chunker = HierarchicalSemanticChunker()

    # Index first 100 records into memory
    records = list(loader.stream_records(max_records=100))
    all_chunks = chunker.process_records(records)

    if all_chunks:
        texts = [c["child_text"] for c in all_chunks]
        embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()
        vector_store.index_chunks(all_chunks, embeddings)

    orchestrator = StatefulRAGOrchestrator(vector_store, embed_model)

    if os.getenv("ELEVENLABS_API_KEY"):
        stt_engine = ElevenLabsSTT()
    logger.info("Pipeline setup complete.")

class TextQueryRequest(BaseModel):
    query: Optional[str] = None
    text: Optional[str] = None
    top_k: int = 3

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "vector_store_ready": vector_store is not None,
        "orchestrator_ready": orchestrator is not None,
        "stt_ready": stt_engine is not None,
    }

@app.post("/api/rag/text")
@app.post("/query/text")
async def text_rag_endpoint(request: TextQueryRequest):
    user_query = request.text or request.query
    if not user_query:
        raise HTTPException(status_code=400, detail="Query or text field is required.")

    start_time = time.perf_counter()
    tokens = []
    final_metadata = None

    async for packet in orchestrator.execute_rag_pipeline(user_query, top_k=request.top_k):
        if packet["type"] == "token":
            tokens.append(packet["content"])
        elif packet["type"] == "final":
            final_metadata = packet["metadata"]
        elif packet["type"] == "error":
            raise HTTPException(status_code=400, detail=packet["content"])

    return {
        "transcription": user_query,
        "answer": "".join(tokens),
        "metadata": final_metadata,
        "server_latency_ms": (time.perf_counter() - start_time) * 1000
    }

import json

@app.websocket("/ws/voice-rag")
async def websocket_voice_rag(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                start_time = time.perf_counter()

                if not stt_engine:
                    await websocket.send_json({"type": "error", "message": "ElevenLabs STT is not configured on server."})
                    continue

                try:
                    transcription = await stt_engine.transcribe_audio_bytes(audio_bytes)
                except Exception as e:
                    logger.error(f"STT Error: {e}")
                    await websocket.send_json({"type": "error", "message": f"Transcription error: {str(e)}"})
                    continue

                if not transcription:
                    await websocket.send_json({"type": "transcription", "text": ""})
                    await websocket.send_json({
                        "type": "final",
                        "data": {
                            "query": "",
                            "transcription": "",
                            "answer": "No speech detected.",
                            "retrieved_contexts": [],
                            "latency_ms": (time.perf_counter() - start_time) * 1000,
                            "grounded": False
                        }
                    })
                    continue

                await websocket.send_json({"type": "transcription", "text": transcription})

                # Stream RAG generation
                async for packet in orchestrator.execute_rag_pipeline(transcription):
                    if packet["type"] == "token":
                        await websocket.send_json({"type": "token", "token": packet["content"]})
                    elif packet["type"] == "final":
                        packet["metadata"]["total_voice_latency_ms"] = (time.perf_counter() - start_time) * 1000
                        await websocket.send_json({"type": "final", "data": packet["metadata"]})
                    elif packet["type"] == "error":
                        await websocket.send_json({"type": "error", "message": packet["content"]})

            elif "text" in message and message["text"]:
                user_text = message["text"].strip()
                try:
                    data = json.loads(user_text)
                    user_query = data.get("query") or data.get("text") or data.get("transcription")
                except Exception:
                    user_query = user_text

                if not user_query:
                    continue

                start_time = time.perf_counter()
                logger.info(f"Processing text query via WebSocket: '{user_query}'")
                await websocket.send_json({"type": "transcription", "text": user_query})

                async for packet in orchestrator.execute_rag_pipeline(user_query):
                    if packet["type"] == "token":
                        await websocket.send_json({"type": "token", "token": packet["content"]})
                    elif packet["type"] == "final":
                        packet["metadata"]["total_voice_latency_ms"] = (time.perf_counter() - start_time) * 1000
                        await websocket.send_json({"type": "final", "data": packet["metadata"]})
                    elif packet["type"] == "error":
                        await websocket.send_json({"type": "error", "message": packet["content"]})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket execution error: {str(e)}")
        try:
            await websocket.close()
        except Exception:
            pass