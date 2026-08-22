import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
from openai import AsyncOpenAI

# FIXED prompt: prevents topic drift in expansions
EXPANSION_PROMPT = """Rephrase the following search query in 3 different ways.

STRICT RULES:
1. Every rephrasing must ask for EXACTLY the same information as the original
2. Do NOT add new topics, related concepts, or contextual information
3. If the query is in Hindi or another language, provide English translations
4. Keep each rephrasing concise — under 15 words
5. Output ONLY the 3 rephrased queries, one per line, no numbering, no explanation

Original query: {query}"""

async def expand_query(query: str) -> list[str]:
    """
    Returns [original] + [3 focused variants].
    If expansion fails or drifts, falls back to just [original].
    """
    providers = [
        {
            "base_url": os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            "api_key": os.getenv("NVIDIA_NIM_API_KEY", ""),
            "model": os.getenv("NVIDIA_NIM_MODEL", "nvidia/nemotron-mini-4b-instruct"),
            "timeout": 3.0
        },
        {
            "base_url": os.getenv("LLM_BASE_URL", "http://localhost:20128/v1"),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "model": os.getenv("LLM_MODEL", "antigravity/claude-sonnet-4-6-low"),
            "timeout": 2.0
        }
    ]

    for p in providers:
        if not p["api_key"]:
            continue
        try:
            client = AsyncOpenAI(
                api_key=p["api_key"],
                base_url=p["base_url"],
                timeout=p["timeout"]
            )
            resp = await client.chat.completions.create(
                model=p["model"],
                max_tokens=150,
                messages=[{"role": "user", "content": EXPANSION_PROMPT.format(query=query)}]
            )
            raw = resp.choices[0].message.content.strip()
            lines = [l.strip() for l in raw.split("\n") if l.strip()][:3]

            # Sanity check: each variant must share >= 1 key word with original
            query_words = set(query.lower().split())
            valid = []
            for line in lines:
                line_words = set(line.lower().split())
                stopwords = {"the", "a", "an", "is", "of", "in", "what", "who",
                             "how", "are", "was", "were", "ke", "hai", "ka", "ki"}
                q_content = query_words - stopwords
                l_content = line_words - stopwords
                if q_content & l_content or len(valid) == 0:
                    valid.append(line)

            return [query] + valid[:3]
        except Exception:
            continue

    return [query]  # Fallback: just use the original
