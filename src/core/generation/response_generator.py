"""
Calls the LLM provider selected by LLMRouter, with a bounded single-retry
policy and per-call timeout, falling through the router's fallback chain on
failure.

Design notes (fixing bugs found in earlier iterations of this project):
- Clients are instance attributes, not module-level globals, so multiple
  ResponseGenerator instances with different config/keys never clash.
- Retry is a single bounded retry via tenacity PLUS an overall asyncio
  timeout per attempt. There is no additional manual sleep stacked on top
  of tenacity's own backoff -- that used to compound into tens of seconds
  of tail latency per request.
- A "local-echo" provider requires no API key and no network access, so the
  whole pipeline can be exercised in CI/tests without live provider creds.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class AllProvidersFailedError(RuntimeError):
    pass


class ResponseGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.api_keys = config.get("api_keys", {})
        self.timeout = config.get("timeout", 30)
        self.llm_router = None  # wired by RAGPipeline after construction
        self._initialized = False

        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None

    async def initialize(self) -> None:
        if self.api_keys.get("openai"):
            try:
                import openai

                self.openai_client = openai.AsyncOpenAI(
                    api_key=self.api_keys["openai"], timeout=self.timeout, max_retries=0
                )
            except ImportError:
                logger.warning("openai package not installed; gpt-* models unavailable")

        if self.api_keys.get("anthropic"):
            try:
                import anthropic

                self.anthropic_client = anthropic.AsyncAnthropic(
                    api_key=self.api_keys["anthropic"], timeout=self.timeout
                )
            except ImportError:
                logger.warning("anthropic package not installed; claude-* models unavailable")

        if self.api_keys.get("google"):
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_keys["google"])
                self.google_client = genai
            except ImportError:
                logger.warning("google-generativeai package not installed; gemini-* models unavailable")

        self._initialized = True

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=1, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    async def _call_provider(self, prompt: str, model: str, temperature: float) -> str:
        if model == "local-echo":
            # Deterministic no-network provider for tests/dev/CI.
            return f"[local-echo] {prompt[-200:]}"

        if model.startswith("gpt") and self.openai_client:
            resp = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            return resp.choices[0].message.content

        if model.startswith("claude") and self.anthropic_client:
            resp = await self.anthropic_client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            return resp.content[0].text

        if model.startswith("gemini") and self.google_client:
            client = self.google_client.GenerativeModel(model)
            resp = await client.generate_content_async(
                prompt, generation_config={"temperature": temperature, "max_output_tokens": 500}
            )
            return resp.text

        raise RuntimeError(f"No configured client available for model '{model}'")

    async def generate_with_fallback(
        self, prompt: str, model: str, temperature: float = 0.7, **kwargs
    ) -> Tuple[str, str]:
        if not self._initialized:
            raise RuntimeError("ResponseGenerator not initialized")

        chain = [model] + (self.llm_router.get_fallback_chain(model) if self.llm_router else [])

        last_error: Optional[Exception] = None
        for attempt, current_model in enumerate(chain):
            try:
                response = await asyncio.wait_for(
                    self._call_provider(prompt, current_model, temperature), timeout=self.timeout
                )
                if response:
                    return response, current_model
            except Exception as e:
                last_error = e
                logger.warning("Model %s failed (attempt %d): %s", current_model, attempt + 1, e)

        raise AllProvidersFailedError(f"All models in fallback chain failed: {chain}") from last_error
