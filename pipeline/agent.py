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
    source_passage_text: str = Field(description="The exact text of the source passage used to form the answer.")
    source_passage_id: str = Field(description="The UUID of the passage used.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 that the answer is correct.")
    grounded: bool = Field(description="True if the answer is completely grounded in the provided passages, False otherwise.")

SYSTEM_PROMPT = """You are a precision search engine that extracts answers from passages.

STRICT RULES — follow exactly:
1. ONLY answer using the provided context chunks.
2. If the answer is NOT in the chunks, say exactly: "I cannot answer this based on the retrieved documents." and set grounded=false, confidence=0.0.
3. NEVER make up information or use outside knowledge.
4. Set source_passage_text to the exact substring from the context you used.
5. Set source_passage_id to the ID of the chunk you used.
6. confidence=1.0 means you found the exact answer explicitly stated.
7. confidence=0.3 means the passage is tangentially related only.

Always respond ONLY with a valid JSON object matching this schema:
{
  "answer": "...",
  "source_passage_text": "...",
  "source_passage_id": "...",
  "confidence": 1.0,
  "grounded": true
}
"""

async def generate_answer(state: RequestState) -> SearchResult:
    """
    Generates the final answer.
    Takes `state`, reads chunks, returns a SearchResult.
    Guarantees no state leakage across calls.
    """
    if not state.retrieved_chunks:
        return SearchResult(
            answer="No relevant passages found.",
            source_passage_text="",
            source_passage_id="",
            confidence=0.0,
            grounded=False
        )

    # Build the prompt payload
    chunks_text = "\n\n".join(
        f"ID: {c.get('passage_id', '')}\nText: {c.get('text', '')}"
        for c in state.retrieved_chunks[:3]
    )

    user_prompt = f"""Query: {state.transcript}

Retrieved Passages:
{chunks_text}

Provide the answer based ONLY on the passages above in JSON format."""

    # Primary high-speed NIM endpoint with local proxy support
    providers = [
        {
            "base_url": os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            "api_key": os.getenv("NVIDIA_NIM_API_KEY", ""),
            "model": os.getenv("NVIDIA_NIM_MODEL", "nvidia/nemotron-mini-4b-instruct"),
            "timeout": 4.0
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
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=250
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

    # Fallback to direct top passage
    top = state.retrieved_chunks[0] if state.retrieved_chunks else {}
    return SearchResult(
        answer=top.get("text", "No answer could be generated."),
        source_passage_text=top.get("text", ""),
        source_passage_id=str(top.get("passage_id", "")),
        confidence=0.7 if top else 0.0,
        grounded=True if top else False
    )
