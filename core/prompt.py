from __future__ import annotations
from typing import List, Dict, Any


class RAGPromptBuilder:
    @staticmethod
    def build_system() -> str:
        return (
            "Eres un asistente que responde solo con la información del contexto proporcionado. "
            "Si no encuentras la respuesta en el contexto, di 'No lo encuentro en el CV'. "
            "Cita brevemente las secciones relevantes cuando corresponda."
        )

    @staticmethod
    def contexts_to_str(contexts: List[Dict[str, Any]]) -> str:
        parts = []
        for i, c in enumerate(contexts, 1):
            meta = c.get("metadata", {})
            src = meta.get("source", meta.get("doc_id", "CV"))
            parts.append(f"[Fragmento {i} | {src} | score={c.get('score', 0):.3f}]\n{c.get('texto','')}\n")
        return "\n---\n".join(parts)

    @staticmethod
    def build_messages(user_query: str, contexts: List[Dict[str, Any]]):
        system = RAGPromptBuilder.build_system()
        context_str = RAGPromptBuilder.contexts_to_str(contexts)
        content = (
            f"Contexto:\n{context_str}\n\n"
            f"Pregunta del usuario: {user_query}\n"
            f"Responde de forma concisa, en español."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
