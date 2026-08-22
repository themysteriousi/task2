# test_expansion.py
import asyncio, httpx, json

async def test_expansion():
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "http://localhost:8000/query/stream",
                                  json={"text": "Abhi ke Bharat ke Pradhan Mantri kaun hai?"},
                                  timeout=30.0) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    d = json.loads(line[6:])
                    if d.get("stage") == "query_expansion" and d.get("status") == "done":
                        variants = d.get("variants", [])
                        print(f"Expansion variants: {variants}")
                        # All variants must relate to Prime Minister, not population
                        bad = [v for v in variants
                               if any(w in v.lower() for w in
                                      ["population", "census", "billion", "crore", "demographic"])]
                        if bad:
                            print(f"❌ FAIL — Off-topic variants: {bad}")
                        else:
                            print("✅ PASS — All variants stay on-topic")

asyncio.run(test_expansion())
