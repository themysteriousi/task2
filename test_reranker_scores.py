# test_reranker_scores.py
import asyncio, httpx, json

async def test_scores():
    async with httpx.AsyncClient() as client:
        answer_data = None
        async with client.stream("POST", "http://localhost:8000/query/stream",
                                  json={"text": "What is machine learning?"},
                                  timeout=30.0) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    d = json.loads(line[6:])
                    if d.get("stage") == "retrieval" and d.get("status") == "done":
                        scores = d.get("top_relevance_scores", [])
                        for s in scores:
                            assert 0.0 <= s <= 1.0, f"Score {s} is not in [0,1]! Logit not converted."
                        print(f"✅ Scores are valid probabilities: {scores}")

asyncio.run(test_scores())
