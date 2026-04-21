# app/summarizer.py
import logging
from typing import Optional
from openai import OpenAI, APIConnectionError, APIStatusError, BadRequestError

logger = logging.getLogger("uvicorn.error")

PROMPTS = {
    "v1": (
        "Generate a concise, factual summary of the document for retrieval."
        "Identify the topic, key entities, processes, rules, and important facts."
        "Write concisely and without fluff."
    )
}


class DocumentSummarizer:
    def __init__(
        self,
        enabled: bool,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 300,
        prompt_version: str = "v1",
        fallback_chars: int = 4000,
    ):
        self.enabled = enabled
        self.fallback_chars = fallback_chars
        self.prompt = PROMPTS.get(prompt_version, PROMPTS["v1"])
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if enabled and api_key:
            try:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info("OpenAI client initialized for summarization")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Summarization disabled.")
                self.client = None
                self.enabled = False
        else:
            self.client = None
            if enabled:
                logger.warning("Summarization enabled but no API key provided. Using fallback.")

    def summarize(self, full_text: str) -> str:
        """Generate summary or return fallback (first N chars)."""
        if not self.enabled or self.client is None:
            logger.debug("Using fallback summarization (first chars)")
            return full_text[: self.fallback_chars]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": full_text[: self.fallback_chars]},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            summary = response.choices[0].message.content.strip()
            if summary:
                logger.debug(f"LLM summary generated ({len(summary)} chars)")
                return summary
            else:
                logger.warning("LLM returned empty summary, using fallback")
                return full_text[: self.fallback_chars]

        except (APIConnectionError, APIStatusError, BadRequestError) as e:
            logger.warning(f"OpenAI API error: {e}. Using fallback.")
            return full_text[: self.fallback_chars]
        except Exception as e:
            logger.error(f"Unexpected error in summarization: {e}. Using fallback.")
            return full_text[: self.fallback_chars]