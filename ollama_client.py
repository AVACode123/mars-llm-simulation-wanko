import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"  # または利用可能なモデル名
DEFAULT_MAX_TOKENS = 200

class OllamaClient:
    """OpenAI API client, keeping the old class name for compatibility."""

    def __init__(
        self,
        base_url=None,
        model=DEFAULT_MODEL,
        temperature=0.7,
        max_tokens=DEFAULT_MAX_TOKENS,
        repeat_penalty=None,
        repeat_last_n=None,
        min_p=None
    ):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, temperature=None, max_tokens=None) -> str:
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""
            
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            return response.output_text.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""

    def check_connection(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def list_models(self):
        return [self.model]

    def check_model_exists(self) -> bool:
        return True