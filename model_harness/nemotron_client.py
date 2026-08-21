import os
import logging
from typing import AsyncGenerator, List, Dict, Any
from openai import AsyncOpenAI

logger = logging.getLogger("NemotronClient")

class NemotronClient:
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_NIM_API_KEY")
        self.base_url = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-4-340b-instruct")

        if not self.api_key:
            raise ValueError("NVIDIA_NIM_API_KEY environment variable is missing.")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def generate_stream(
        self, 
        query: str, 
        contexts: List[str], 
        system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        if not system_prompt:
            system_prompt = (
                "You are an accurate AI assistant. Answer the query strictly based on the provided context. "
                "If the context does not contain enough information, state clearly that you do not know."
            )

        context_str = "\n\n".join([f"Context [{i+1}]: {ctx}" for i, ctx in enumerate(contexts)])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context Information:\n{context_str}\n\nQuery: {query}"}
        ]

        try:
            logger.info(f"Sending prompt to Nemotron model: {self.model}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
                stream=True
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Nemotron API streaming error: {str(e)}")
            raise RuntimeError(f"Nemotron Generation Failed: {str(e)}")