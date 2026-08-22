"""
pipeline/stt.py
Speech-to-Text integration with Sarvam AI API.
"""

import os
import io
import time
import logging
from typing import Optional, Tuple
import httpx

logger = logging.getLogger("VoiceRAG.STT")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


async def transcribe(audio_bytes: bytes, language_code: str = "en-IN", model: str = "saarika:v2.5") -> Optional[str]:
    """
    Send audio to Sarvam STT. Returns transcript string or None on failure.
    Accepts wav, mp3, webm audio formats.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        logger.error("SARVAM_API_KEY is missing from environment.")
        return None

    model_name = os.getenv("SARVAM_STT_MODEL", model)

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            files = {
                "file": ("audio.wav", audio_bytes, "audio/wav")
            }
            data = {
                "language_code": language_code,
                "model": model_name,
            }
            resp = await client.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": api_key},
                files=files,
                data=data,
            )
            if resp.status_code != 200:
                logger.error(f"Sarvam STT failed ({resp.status_code}): {resp.text}")
                return None

            res_json = resp.json()
            return res_json.get("transcript", "").strip() or None
        except Exception as e:
            logger.error(f"Sarvam STT exception: {e}")
            return None


class SarvamSTT:
    """Wrapper class for pipeline compatibility."""
    def __init__(self, api_key: Optional[str] = None, model: str = "saarika:v2.5"):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        self.model = os.getenv("SARVAM_STT_MODEL", model)

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav", language_code: str = "en-IN") -> Tuple[str, float]:
        start = time.perf_counter()
        t = await transcribe(audio_bytes, language_code=language_code, model=self.model)
        latency_ms = (time.perf_counter() - start) * 1000
        if not t:
            raise RuntimeError("Transcription failed or returned empty text.")
        return t, latency_ms
