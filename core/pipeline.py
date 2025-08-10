from __future__ import annotations
from typing import List, Dict, Any, Tuple

from .store import PineconeStore
from .embedder import Embedder
from .llm import GroqLLM
from .memory import MemoryWindow
from .prompt import RAGPromptBuilder


class RAGPipeline:
    def __init__(self, store: PineconeStore, embedder: Embedder, llm: GroqLLM, top_k: int = 4, memory: MemoryWindow | None = None):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.memory = memory or MemoryWindow(6)

    def answer(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        docs = self.store.query(query, self.embedder, top_k=self.top_k)
        messages = RAGPromptBuilder.build_messages(query, docs)
        mem_msgs = self.memory.get()
        full_messages = mem_msgs + messages if mem_msgs else messages
        answer = self.llm.chat(full_messages)
        self.memory.add("user", query)
        self.memory.add("assistant", answer)
        return answer, docs
