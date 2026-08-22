import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from pipeline.state import RequestState

class SearchResult(BaseModel):
    answer: str = Field(description="The final answer to the user's query.")
    source_passage_text: str = Field(description="The exact text or snippet of the source passage used to form the answer.")
    source_passage_id: str = Field(description="The URL or ID of the passage used.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 that the answer is correct.")
    grounded: bool = Field(description="True if the answer is grounded in the retrieved passages, False otherwise.")

SYSTEM_PROMPT = """You are a high-speed precision search engine assistant.
Your task is to provide accurate, concise, factual, and direct answers to user queries using the provided live internet search results.

RULES:
1. Provide a direct, factual answer to the query based on the search passages.
2. Keep the answer concise, natural, and informative (1-3 sentences).
3. Set source_passage_text to the most relevant snippet or sentence from the web results.
4. Set source_passage_id to the URL or ID of the source web page.
5. Set confidence between 0.85 and 1.0 for direct factual answers, or lower if ambiguous.
6. Set grounded to true when the answer is supported by the web search results.

Always respond ONLY with a valid JSON object matching this schema:
{
  "answer": "...",
  "source_passage_text": "...",
  "source_passage_id": "...",
  "confidence": 0.95,
  "grounded": true
}
"""

async def generate_answer(state: RequestState) -> SearchResult:
    """
    Generates the final answer.
    Takes `state`, reads retrieved web passages, returns a SearchResult.
    Guarantees no state leakage across calls.
    """
    if not state.retrieved_chunks:
        return SearchResult(
            answer="No relevant information found on the web for this query.",
            source_passage_text="",
            source_passage_id="",
            confidence=0.0,
            grounded=False
        )

    # Build the prompt payload
    chunks_text = "\n\n".join(
        f"Source: {c.get('title', 'Web Result')} ({c.get('passage_id', c.get('url', ''))})\nSnippet: {c.get('text', '')}"
        for c in state.retrieved_chunks[:4]
    )

    user_prompt = f"""Query: {state.transcript}

Live Internet Search Results:
{chunks_text}

Provide the concise, factual answer to the query in JSON format."""

    # Primary high-speed NIM endpoint with local proxy support
    providers = [
        {
            "base_url": os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            "api_key": os.getenv("NVIDIA_NIM_API_KEY", "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"),
            "model": os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-mini-4b-instruct"),
            "timeout": 4.0
        },
        {
            "base_url": os.getenv("LLM_BASE_URL", "http://localhost:20128/v1"),
            "api_key": os.getenv("LLM_API_KEY", "sk-13445f2000b0373e-915022-933bda8c"),
            "model": os.getenv("LLM_MODEL", "antigravity/claude-sonnet-4-6-low"),
            "timeout": 2.5
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
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            content = resp.choices[0].message.content.strip()

            # Extract JSON if wrapped in markdown code blocks or has extra text
            if "```" in content:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                if match:
                    content = match.group(1)
            elif "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]

            data = json.loads(content)
            return SearchResult(**data)
        except Exception:
            # Continue to next provider or fallback
            continue

    # Fallback to direct top passage if LLM call fails
    top = state.retrieved_chunks[0] if state.retrieved_chunks else {}
    top_text = top.get("text", "")
    return SearchResult(
        answer=top_text or "No answer could be retrieved from the internet.",
        source_passage_text=top_text,
        source_passage_id=str(top.get("passage_id", top.get("url", ""))),
        confidence=0.85 if top else 0.0,
        grounded=True if top else False
    )
