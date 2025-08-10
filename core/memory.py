from __future__ import annotations
from typing import List, Dict


class MemoryWindow:
    def __init__(self, k: int = 6):
        self.k = k
        self.buffer: List[Dict[str, str]] = []

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        if len(self.buffer) > 2 * self.k:  # user+assistant por turno
            self.buffer = self.buffer[-2 * self.k :]

    def get(self) -> List[Dict[str, str]]:
        return list(self.buffer)
