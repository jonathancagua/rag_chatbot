# RAG Chatbot (Streamlit + Pinecone Serverless + Groq)

Chatbot tipo RAG que indexa tu **CV** (PDF/TXT) en Pinecone y responde preguntas usando contexto recuperado. UI en Streamlit, embeddings con Sentence-Transformers y LLM de Groq.

## Stack

* **Frontend**: Streamlit
* **RAG**: Pinecone (serverless v3) + Sentence-Transformers
* **LLM**: Groq (`llama3-8b-8192` por defecto)
* **Gestión**: Poetry, `.env`

## Requisitos

* Python **3.10–3.12**
* Claves: `GROQ_API_KEY` y `PINECONE_API_KEY`

## Instalación (Poetry)

```bash
# en la raíz del proyecto
cp .env.example .env
poetry env use 3.11     # o tu versión en el rango 3.10–3.12
poetry install
poetry run streamlit run app.py
```

## Variables de entorno (.env)

```env
GROQ_API_KEY=tu_clave
MODEL_NAME=llama3-8b-8192

PINECONE_API_KEY=tu_clave
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_INDEX_NAME=cv-index

EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=4
TEMPERATURE=0.2
MAX_TOKENS=800
MEMORY_K=6
CHUNK_SIZE=800
OVERLAP=100
```

## Uso

1. **Sube tu CV** (PDF o TXT) desde la app.
2. Presiona **“Indexar en Pinecone”** para crear fragmentos y subirlos.
3. Pregunta en lenguaje natural (ej.: “¿Qué skills domina?”).
4. Mira la respuesta, los fragmentos usados y la memoria de la conversación.

## Estructura

```
TP1-RAG/
├─ app.py
├─ .env.example
├─ core/
│  ├─ config.py      # carga .env
│  ├─ loader.py      # lee PDF/TXT
│  ├─ chunker.py     # split con overlap
│  ├─ embedder.py    # sentence-transformers
│  ├─ store.py       # Pinecone v3 serverless
│  ├─ prompt.py      # plantilla RAG
│  ├─ llm.py         # cliente Groq
│  ├─ memory.py      # ventana de memoria
│  └─ pipeline.py    # orquestación RAG
└─ pyproject.toml / requirements.txt
```
## 📹 Demo
<video src="video.webm" controls style="max-width:100%; width:720px;">
  Tu navegador no soporta video HTML5. Mira el archivo aquí:
  <a href="video.webm">video.webm</a>.
</video>

## Notas importantes

* **Pinecone serverless (v3)**: en planes Starter, usa `aws` + `us-east-1`. Ajusta en `.env`.
* Evita el cliente viejo `pinecone-client<3`; este proyecto ya usa `pinecone>=3`.
* Si cambias el modelo/embeddings, reindexa para alinear dimensiones.


