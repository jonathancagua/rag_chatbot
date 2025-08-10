from __future__ import annotations
import os
import streamlit as st

from core.config import load_config
from core.embedder import Embedder
from core.store import PineconeStore
from core.llm import GroqLLM
from core.memory import MemoryWindow
from core.pipeline import RAGPipeline
from core.loader import PDFLoader
from core.chunker import TextChunker


def _load_cv_text(uploaded) -> str:
    if uploaded is None:
        return ""
    data = uploaded.read()
    name = (uploaded.name or "").lower()
    if name.endswith(".pdf"):
        return PDFLoader.load_pdf(data)
    return data.decode("utf-8", errors="ignore")


def main():
    st.set_page_config(page_title="TP1 – RAG CV", page_icon="🗂️", layout="wide")
    st.title("🗂️ TP1 – Chatbot RAG que consulta tu CV")
    st.caption("Arquitectura con clases + configuración por .env (Pinecone Serverless)")

    cfg = load_config()

    # Sidebar – parámetros de consulta (los secretos vienen del .env)
    with st.sidebar:
        st.header("⚙️ Parámetros")
        top_k = st.slider("Top-K", 1, 10, cfg.top_k)
        temperature = st.slider("Temperature", 0.0, 1.0, cfg.temperature, 0.05)
        memory_k = st.slider("Memoria (turnos)", 1, 10, cfg.memory_k)
        max_tokens = st.slider("max_tokens", 128, 2048, cfg.max_tokens, 64)
        chunk_size = st.number_input("chunk_size", 200, 2000, cfg.chunk_size, 50)
        overlap = st.number_input("overlap", 0, 400, cfg.overlap, 10)

    # Validación de credenciales
    if not cfg.groq_api_key:
        st.error("Falta GROQ_API_KEY en .env")
        st.stop()
    if not cfg.pinecone_api_key:
        st.error("Falta PINECONE_API_KEY en .env")
        st.stop()

    # Componentes
    embedder = Embedder(cfg.embed_model)
    store = PineconeStore(cfg.pinecone_api_key, cfg.pinecone_cloud, cfg.pinecone_region, cfg.pinecone_index)
    store.ensure_index(dimension=embedder.dim, metric="cosine")
    llm = GroqLLM(cfg.groq_api_key, cfg.model_name, temperature, max_tokens)

    # Estado: memoria
    if "_mem_k" not in st.session_state or st.session_state.get("_mem_k") != memory_k:
        st.session_state["memory"] = MemoryWindow(memory_k)
        st.session_state["_mem_k"] = memory_k
    memory: MemoryWindow = st.session_state["memory"]

    pipeline = RAGPipeline(store, embedder, llm, top_k, memory)

    # Ingesta del CV
    st.subheader("1) Carga tu CV")
    uploaded = st.file_uploader("Sube tu CV en PDF o .txt", type=["pdf", "txt"])

    if uploaded is None:
        st.info("Sube un archivo para continuar.")
        return

    cv_text = _load_cv_text(uploaded)
    st.success(f"CV cargado. Longitud: {len(cv_text)} caracteres")
    st.text_area("Vista previa (solo lectura)", cv_text[:4000], height=200)

    if st.button("📥 Indexar en Pinecone", type="primary"):
        with st.spinner("Creando chunks y subiendo a Pinecone..."):
            chunks = TextChunker.split(cv_text, int(chunk_size), int(overlap))
            base_meta = {"doc_id": uploaded.name, "source": uploaded.name, "tipo": "CV"}
            inserted = store.upsert_chunks(chunks, base_meta, embedder)
        st.success(f"Se indexaron {inserted} fragmentos en '{cfg.pinecone_index}'.")

    # Consulta
    st.subheader("2) Pregunta sobre tu CV")
    query = st.text_input("Escribe tu pregunta", placeholder="¿En qué empresas tiene experiencia? ¿Qué skills domina?")

    col1, col2 = st.columns([1,1])
    with col1:
        ask = st.button("🔎 Consultar", type="primary")
    with col2:
        clear = st.button("🧹 Limpiar memoria")

    if clear:
        st.session_state["memory"] = MemoryWindow(memory_k)
        st.success("Memoria conversacional reiniciada")

    if ask and query.strip():
        with st.spinner("Generando respuesta con RAG..."):
            answer, sources = pipeline.answer(query)
        st.markdown("### 🤖 Respuesta")
        st.write(answer)
        with st.expander("📎 Fragmentos relevantes"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**{i}.** *score* {src['score']:.3f}")
                st.code(src["texto"][:1200])
        with st.expander("🧠 Memoria (últimos turnos)"):
            for m in memory.get():
                who = "👤" if m["role"] == "user" else "🤖"
                st.write(f"{who} {m['content']}")


if __name__ == "__main__":
    main()
