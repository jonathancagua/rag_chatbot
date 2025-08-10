from __future__ import annotations
from typing import List


class TextChunker:
    @staticmethod
    def split(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks
