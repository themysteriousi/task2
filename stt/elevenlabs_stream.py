import os
import logging
import httpx
from typing import Optional
from elevenlabs.client import ElevenLabs

logger = logging.getLogger("ElevenLabsSTT")

class ElevenLabsSTT:
    def __init__(self, api_key: Optional[str] = None, model_id: str = "scribe_v2"):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set in environment variables.")
        
        self.model_id = model_id
        self.client = ElevenLabs(api_key=self.api_key)

    async def transcribe_audio_bytes(self, audio_data: bytes, language_code: str = "eng") -> str:
        """
        Transcribe incoming audio byte buffer using ElevenLabs Scribe v2 model.
        """
        try:
            logger.info("Sending audio stream to ElevenLabs Scribe API...")
            
            # Call ElevenLabs Speech-to-Text endpoint
            transcription = self.client.speech_to_text.convert(
                file=audio_data,
                model_id=self.model_id,
                tag_audio_events=False,
                language_code=language_code
            )
            
            transcribed_text = transcription.text.strip()
            logger.info(f"Transcription completed: '{transcribed_text}'")
            return transcribed_text

        except Exception as e:
            logger.error(f"ElevenLabs STT error: {str(e)}")
            raise RuntimeError(f"ElevenLabs STT Processing Failed: {str(e)}")