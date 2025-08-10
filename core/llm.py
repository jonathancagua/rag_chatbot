from __future__ import annotations
from typing import List, Dict
from groq import Groq


class GroqLLM:
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192", temperature: float = 0.2, max_tokens: int = 800):
        self.client = Groq(api_key=api_key)
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.9,
        )
        return resp.choices[0].message.content
