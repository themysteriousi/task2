# test_isolation.py
import asyncio, httpx, json

async def test_isolation():
    """
    Send 3 different queries rapidly. Each answer MUST match its query.
    If any answer bleeds into another query's result, Bug #1 is not fixed.
    """
    queries = [
        {"text": "Who is the Prime Minister of India?",    "keyword": "Modi"},
        {"text": "What is the population of India?",       "keyword": "billion"},
        {"text": "What is the capital city of India?",     "keyword": "Delhi"},
    ]

    async with httpx.AsyncClient() as client:
        # Fire all 3 concurrently — maximum stress test for isolation
        tasks = [
            run_single(client, q["text"], q["keyword"])
            for q in queries
        ]
        results = await asyncio.gather(*tasks)

    for q, ok in zip(queries, results):
        status = "✅ PASS" if ok else "❌ FAIL — stale state bug still present"
        print(f"{status}: '{q['text'][:40]}' → keyword '{q['keyword']}'")

async def run_single(client, query, keyword):
    answer = ""
    async with client.stream("POST", "http://localhost:8000/query/stream",
                              json={"text": query}, timeout=30.0) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                d = json.loads(line[6:])
                if d.get("stage") == "pipeline":
                    answer = d.get("result", {}).get("answer", "")
    return keyword.lower() in answer.lower()

asyncio.run(test_isolation())
