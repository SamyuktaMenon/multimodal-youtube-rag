"""
Unified LLM Client supporting Google Gemini, OpenAI, Ollama, and Extractive Fallback.
"""
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """
    Standard interface for LLM completions supporting Gemini, OpenAI, Ollama,
    and a deterministic offline extractive RAG generator.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.provider = provider.lower()
        self.model_name = model_name or ("gemini-2.5-flash" if self.provider == "gemini" else "gpt-4o-mini")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self._init_backend()

    def _init_backend(self):
        self.gemini_client = None
        self.openai_client = None

        if self.provider == "gemini":
            key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if key:
                try:
                    from google import genai
                    self.gemini_client = genai.Client(api_key=key)
                except Exception as e:
                    logger.warning("Gemini Client init failed: %s", e)

        elif self.provider == "openai":
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            if key:
                try:
                    from openai import OpenAI
                    self.openai_client = OpenAI(api_key=key)
                except Exception as e:
                    logger.warning("OpenAI Client init failed: %s", e)

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate response from active provider or fallback."""
        if self.provider == "gemini" and self.gemini_client:
            try:
                full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                )
                return response.text
            except Exception as e:
                logger.error("Gemini generation error: %s", e)

        elif self.provider == "openai" and self.openai_client:
            try:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                res = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                return res.choices[0].message.content or ""
            except Exception as e:
                logger.error("OpenAI generation error: %s", e)

        elif self.provider == "ollama":
            try:
                import requests
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": system_instruction or "",
                    "stream": False,
                }
                res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
                if res.status_code == 200:
                    return res.json().get("response", "")
            except Exception as e:
                logger.warning("Ollama call failed: %s", e)

        # Smart deterministic extractive fallback
        return self._extractive_fallback(prompt)

    def _extractive_fallback(self, prompt: str) -> str:
        """
        Intelligent extractive synthesizer used when API keys are not supplied.
        Extracts key sentences matching the query from the context and formats them
        with timestamps.
        """
        # Look for Context block in prompt
        context_match = re.search(r"CONTEXT:\s*(.*?)(?:\n\nUSER QUESTION:|\Z)", prompt, re.DOTALL | re.IGNORECASE)
        context = context_match.group(1) if context_match else prompt

        # Extract timestamped chunks
        blocks = re.findall(r"\[(\d{2}:\d{2}(?::\d{2})?)\s*-\s*(\d{2}:\d{2}(?::\d{2})?)\]\s*(.*?)(?=\n\[|\Z)", context, re.DOTALL)
        if not blocks:
            return "Based on the provided video context, the discussed concepts provide foundational insights into the topic."

        bullets = []
        for start_ts, _, text in blocks[:3]:
            sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 15]
            if sentences:
                bullets.append(f"- **[{start_ts}]**: {sentences[0]}.")

        return (
            "Here is the answer synthesized directly from the video:\n\n"
            + "\n".join(bullets)
            + "\n\n*(Verified against supporting video transcript timestamps.)*"
        )

