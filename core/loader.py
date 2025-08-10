from __future__ import annotations
import io
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


class PDFLoader:
    @staticmethod
    def load_pdf(file_bytes: bytes) -> str:
        if PdfReader is None:
            raise RuntimeError("pypdf no está instalado. Añádelo a requirements.txt")
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n\n".join(texts)
