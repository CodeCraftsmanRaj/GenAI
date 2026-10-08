from pathlib import Path
from typing import Iterable
import json
from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".html", ".htm", ".pdf", ".csv", ".json"}

def extract_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(f"[Page {i+1}]\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages))
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text("\n", strip=True)
    if suffix == ".json":
        try:
            return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")
    return path.read_text(encoding="utf-8", errors="ignore")

def load_documents(root: Path) -> list[dict]:
    docs = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            text = extract_file(path).strip()
            if text:
                docs.append({"source": path.name, "source_path": str(path.relative_to(root)), "text": text})
    return docs

def chunk_documents(documents: Iterable[dict], chunk_size_words: int = 450, overlap_words: int = 60) -> list[dict]:
    if chunk_size_words <= 0 or overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("Require chunk_size_words > overlap_words >= 0")
    chunks = []
    for doc in documents:
        words = doc["text"].split()
        step = chunk_size_words - overlap_words
        for start in range(0, len(words), step):
            part = words[start:start + chunk_size_words]
            if not part:
                continue
            chunks.append({
                "chunk_id": f"chunk_{len(chunks):05d}",
                "source": doc["source"],
                "source_path": doc.get("source_path", doc["source"]),
                "page_section": f"words {start}-{start + len(part)}",
                "text": " ".join(part),
            })
    return chunks
