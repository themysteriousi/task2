"""
test_accuracy.py
Verifies factual accuracy and keyword grounding against ground truth facts.
"""

import asyncio
import httpx
import json

GROUND_TRUTH = [
    {"query": "What is machine learning?", "expected_keyword": "artificial intelligence"},
    {"query": "Who invented the telephone?", "expected_keyword": "Bell"},
    {"query": "What is the capital of France?", "expected_keyword": "Paris"},
    {"query": "How does photosynthesis work?", "expected_keyword": "light"},
    {"query": "What are the symptoms of diabetes?", "expected_keyword": "thirst"},
    {"query": "What is Reciprocal Rank Fusion?", "expected_keyword": "ranking"},
]


async def test():
    print("🧪 Running Ground Truth Accuracy Validation...\n")
    passed = 0
    async with httpx.AsyncClient() as client:
        for case in GROUND_TRUTH:
            answer_text = ""
            async with client.stream(
                "POST",
                "http://127.0.0.1:8000/query/stream",
                json={"text": case["query"]},
                timeout=30.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            d = json.loads(line[6:])
                            if d.get("stage") == "pipeline":
                                answer_text = d.get("result", {}).get("answer", "")
                        except Exception:
                            pass

            kw = case["expected_keyword"]
            found = kw.lower() in answer_text.lower()
            if found:
                passed += 1
            print(f"{'✅' if found else '❌'} '{case['query'][:35]:<35}' → keyword '{kw}' {'FOUND' if found else 'MISSING'}")
            print(f"   Answer: {answer_text[:110]}...\n")

    print(f"Accuracy: {passed}/{len(GROUND_TRUTH)} ({passed/len(GROUND_TRUTH)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(test())
