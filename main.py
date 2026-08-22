"""
Every request creates a fresh RequestState().
No shared mutable state across requests.
"""
import asyncio, json, uuid, time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.state import RequestState
from pipeline.stt import transcribe
from pipeline.guardrails import check_input, check_output
from pipeline.query_expander import expand_query
from pipeline.embedder import embed_query
from pipeline.retriever import retrieve
from pipeline.agent import generate_answer
from pipeline.latency import tracker, Timer, PipelineTrace

app = FastAPI(title="Voice RAG — Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class TextQuery(BaseModel):
    text: str
    language: str = "en-IN"

def sse(stage: str, status: str, data: dict, request_id: str) -> str:
    payload = {"stage": stage, "status": status, "request_id": request_id, **data}
    return f"data: {json.dumps(payload)}\n\n"

async def run_pipeline_stream(transcript: str, request_id: str):
    """
    Creates a brand-new RequestState for each call.
    Zero shared mutable state. Every variable is local or in `state`.
    """
    # ── Fresh state for THIS request ───────────────────────────────
    state = RequestState(request_id=request_id)
    state.transcript = transcript

    # ── Stage 0: Transcript confirmed ──────────────────────────────
    yield sse("transcript", "done", {"transcript": transcript}, request_id)

    # ── Stage 1: Input guardrail ────────────────────────────────────
    yield sse("guardrail_input", "running", {}, request_id)
    with Timer() as t:
        block = check_input(transcript)
    state.guardrail_ms = t.elapsed_ms

    if block:
        yield sse("guardrail_input", "blocked",
                  {"result": block, "ms": t.elapsed_ms}, request_id)
        yield sse("pipeline", "complete",
                  {"blocked": True, "result": block, "request_id": request_id},
                  request_id)
        return  # State goes out of scope here — no leakage

    yield sse("guardrail_input", "passed", {"ms": t.elapsed_ms}, request_id)

    # ── Stage 2: Query expansion ────────────────────────────────────
    yield sse("query_expansion", "running", {"query": transcript}, request_id)
    with Timer() as t:
        expanded = await expand_query(transcript)
    state.expand_ms = t.elapsed_ms
    state.expanded_queries = expanded  # stored in THIS request's state only

    yield sse("query_expansion", "done", {
        "variants": expanded,
        "count": len(expanded),
        "ms": t.elapsed_ms
    }, request_id)

    # ── Stage 3: Embedding ──────────────────────────────────────────
    yield sse("embedding", "running", {"model": "multilingual-e5-large"}, request_id)
    loop = asyncio.get_event_loop()
    with Timer() as t:
        emb = await loop.run_in_executor(None, embed_query, transcript)
    state.embed_ms = t.elapsed_ms
    state.query_embedding = emb.tolist()

    yield sse("embedding", "done", {
        "dims": len(emb),
        "ms": t.elapsed_ms
    }, request_id)

    # ── Stage 4: Live Internet Web Retrieval + Neural Reranking ───────
    yield sse("retrieval", "running", {"strategy": "live internet search + cross-encoder"}, request_id)
    with Timer() as t:
        chunks = await retrieve(state, top_k=5)  # reads/writes state
    state.retrieve_ms = t.elapsed_ms

    if not chunks:
        no_result = {
            "answer": "No relevant information found on the web for this query.",
            "source_passage_text": "", "confidence": 0.0, "grounded": False
        }
        yield sse("retrieval", "empty", {"ms": t.elapsed_ms}, request_id)
        yield sse("pipeline", "complete",
                  {"result": no_result, "latency": state.latency_breakdown()},
                  request_id)
        return

    # Convert logits to probabilities for display
    top_probs = [c.get("rerank_prob", 0.0) for c in chunks[:3]]
    yield sse("retrieval", "done", {
        "count": len(chunks),
        "top_relevance_scores": top_probs,   # 0-1 probabilities, not raw logits
        "top_passages_preview": [c["text"][:120] + "..." for c in chunks[:3]],
        "strategies_used": list({c["strategy"] for c in chunks}),
        "sources": [c.get("title", "Web Source") for c in chunks[:3]],
        "ms": t.elapsed_ms
    }, request_id)

    # ── Stage 5: Answer generation ───────────────────────────────────
    yield sse("generation", "running", {
        "model": "nvidia/nemotron-mini-4b-instruct",
        "context_chunks": len(chunks)
    }, request_id)
    with Timer() as t:
        answer = await generate_answer(state)   # reads state, no globals
    state.generate_ms = t.elapsed_ms

    yield sse("generation", "done", {
        "confidence": answer.confidence,
        "grounded": answer.grounded,
        "ms": t.elapsed_ms
    }, request_id)

    # ── Stage 6: Output guardrail ────────────────────────────────────
    yield sse("guardrail_output", "running", {}, request_id)
    with Timer() as t:
        override = check_output(answer, chunks)
    state.guardrail_ms += t.elapsed_ms

    if override:
        yield sse("guardrail_output", "override",
                  {"result": override, "ms": t.elapsed_ms}, request_id)
        yield sse("pipeline", "complete",
                  {"result": override, "latency": state.latency_breakdown()},
                  request_id)
        return

    yield sse("guardrail_output", "passed", {"ms": t.elapsed_ms}, request_id)

    # ── Final result ──────────────────────────────────────────────────
    latency = state.latency_breakdown()

    # Record to tracker (append-only, thread-safe)
    trace = PipelineTrace(
        query_id=request_id,
        guardrail_input_ms=state.guardrail_ms,
        query_expand_ms=state.expand_ms,
        embed_ms=state.embed_ms,
        retrieve_ms=state.retrieve_ms,
        generate_ms=state.generate_ms,
        total_ms=latency["total_ms"]
    )
    tracker.record(trace)

    final_result = {
        "answer": answer.answer,
        "source_passage_text": answer.source_passage_text,
        "source_passage_id": answer.source_passage_id,
        "confidence": answer.confidence,
        "grounded": answer.grounded,
        "top_passages": [
            {
                "title": c.get("title", "Web Source"),
                "url": c.get("url", ""),
                "text": c["text"],
                "passage_id": c["passage_id"],
                "relevance": c.get("rerank_prob", 0.0)
            }
            for c in chunks[:3]
        ],
        "latency_breakdown": latency,
        "request_id": request_id
    }

    yield sse("pipeline", "complete", {"result": final_result}, request_id)
    # state goes out of scope here — garbage collected — zero leakage


@app.post("/query/stream")
async def query_stream_endpoint(body: TextQuery):
    request_id = str(uuid.uuid4())[:8]
    return StreamingResponse(
        run_pipeline_stream(body.text, request_id),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )

@app.post("/query/voice/stream")
async def query_voice_stream_endpoint(
    file: UploadFile = File(...),
    language: str = "en-IN"
):
    audio_bytes = await file.read()
    request_id = str(uuid.uuid4())[:8]

    async def stream():
        with Timer() as t:
            transcript = await transcribe(audio_bytes, language)
        stt_ms = t.elapsed_ms

        if not transcript:
            yield f"data: {json.dumps({'stage':'stt','status':'failed','ms':stt_ms,'request_id':request_id})}\n\n"
            return

        yield f"data: {json.dumps({'stage':'stt','status':'done','transcript':transcript,'ms':stt_ms,'request_id':request_id})}\n\n"
        # Now run through the full pipeline with the fresh transcript
        async for event in run_pipeline_stream(transcript, request_id):
            yield event

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )

@app.get("/latency/report")
async def latency_report():
    return tracker.report()

@app.get("/health")
async def health():
    from pipeline.retriever import get_qdrant_client, COLLECTION
    try:
        qc = get_qdrant_client()
        info = qc.get_collection(COLLECTION)
        count = info.points_count
    except Exception:
        count = -1
    return {
        "status": "ok",
        "index_size": count,
        "queries_served": len(tracker._traces)
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html") as f:
        return f.read()

class LegacyQueryRequest(BaseModel):
    query: Optional[str] = None
    text: Optional[str] = None
    top_k: int = 3

@app.post("/api/rag/text")
@app.post("/query/text")
async def legacy_text_rag_endpoint(request: LegacyQueryRequest):
    user_query = request.text or request.query or ""
    if not user_query:
        return {
            "answer": "Please provide a query.",
            "transcription": "",
            "retrieved_contexts": [],
            "latency_ms": 0,
            "grounded": False
        }

    req_id = str(uuid.uuid4())[:8]
    final_res = None
    async for event in run_pipeline_stream(user_query, req_id):
        if event.startswith("data: "):
            try:
                d = json.loads(event[6:].strip())
                if d.get("stage") == "pipeline" and "result" in d:
                    final_res = d["result"]
            except Exception:
                pass

    if not final_res:
        return {
            "query": user_query,
            "transcription": user_query,
            "answer": "No answer could be generated.",
            "retrieved_contexts": [],
            "latency_ms": 0,
            "grounded": False
        }

    return {
        "query": user_query,
        "transcription": user_query,
        "answer": final_res.get("answer", ""),
        "retrieved_contexts": [p.get("text", "") for p in final_res.get("top_passages", [])],
        "latency_ms": final_res.get("latency_breakdown", {}).get("total_ms", 0),
        "grounded": final_res.get("grounded", False),
        "total_voice_latency_ms": final_res.get("latency_breakdown", {}).get("total_ms", 0),
    }

@app.websocket("/ws/voice-rag")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    data = {"query": msg["text"]}
                query = data.get("query", "")
                if query:
                    await ws.send_json({"type": "transcription", "text": query})
                    req_id = str(uuid.uuid4())[:8]
                    async for event in run_pipeline_stream(query, req_id):
                        if event.startswith("data: "):
                            try:
                                d = json.loads(event[6:].strip())
                                if d.get("stage") == "pipeline" and "result" in d:
                                    r = d["result"]
                                    await ws.send_json({
                                        "type": "final",
                                        "data": {
                                            "query": query,
                                            "transcription": query,
                                            "answer": r.get("answer", ""),
                                            "retrieved_contexts": [p.get("text", "") for p in r.get("top_passages", [])],
                                            "latency_ms": r.get("latency_breakdown", {}).get("total_ms", 0),
                                            "grounded": r.get("grounded", False),
                                            "total_voice_latency_ms": r.get("latency_breakdown", {}).get("total_ms", 0),
                                        }
                                    })
                            except Exception:
                                pass
            elif "bytes" in msg and msg["bytes"]:
                audio_bytes = msg["bytes"]
                req_id = str(uuid.uuid4())[:8]
                transcript = await transcribe(audio_bytes)
                await ws.send_json({"type": "transcription", "text": transcript})
                if transcript:
                    async for event in run_pipeline_stream(transcript, req_id):
                        if event.startswith("data: "):
                            try:
                                d = json.loads(event[6:].strip())
                                if d.get("stage") == "pipeline" and "result" in d:
                                    r = d["result"]
                                    await ws.send_json({
                                        "type": "final",
                                        "data": {
                                            "query": transcript,
                                            "transcription": transcript,
                                            "answer": r.get("answer", ""),
                                            "retrieved_contexts": [p.get("text", "") for p in r.get("top_passages", [])],
                                            "latency_ms": r.get("latency_breakdown", {}).get("total_ms", 0),
                                            "grounded": r.get("grounded", False),
                                            "total_voice_latency_ms": r.get("latency_breakdown", {}).get("total_ms", 0),
                                        }
                                    })
                            except Exception:
                                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

