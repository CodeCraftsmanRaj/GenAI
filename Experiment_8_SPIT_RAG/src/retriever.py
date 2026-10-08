from pathlib import Path
import json

import faiss
import numpy as np
import requests


class VectorRetriever:
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.index = None
        self.chunks = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings using Ollama."""
        vectors = []

        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model_name,
                    "input": text,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("embeddings"):
                raise RuntimeError(
                    f"Ollama returned no embeddings: {data}"
                )

            vectors.append(data["embeddings"][0])

        vectors = np.asarray(vectors, dtype="float32")

        # Normalize vectors so inner product corresponds to cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)

        return vectors

    def build(self, chunks: list[dict]):
        if not chunks:
            raise ValueError(
                "No chunks found. Add documents to knowledge_base first."
            )

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        print(f"Generating Ollama embeddings for {len(texts)} chunks...")
        vectors = self._embed(texts)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index is None:
            raise RuntimeError(
                "Index is not built/loaded. Run: uv run main.py ingest"
            )

        if not self.chunks:
            return []

        query_vector = self._embed([query])
        k = min(top_k, len(self.chunks))
        scores, ids = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = dict(self.chunks[int(idx)])
                item["score"] = float(score)
                results.append(item)

        return results

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "vectors.faiss"),
        )

        (directory / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "embedding_provider": "ollama",
                    "embedding_model": self.model_name,
                    "chunk_count": len(self.chunks),
                    "embedding_dimension": self.index.d,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, directory: Path):
        self.index = faiss.read_index(
            str(directory / "vectors.faiss")
        )

        self.chunks = json.loads(
            (directory / "chunks.json").read_text(encoding="utf-8")
        )

        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            saved_model = metadata.get("embedding_model")
            if saved_model and saved_model != self.model_name:
                raise RuntimeError(
                    f"Index uses embedding model {saved_model!r}, "
                    f"but config specifies {self.model_name!r}. "
                    "Run ingestion again."
                )

        if self.index.d != len(self._embed(["embedding dimension check"])[0]):
            raise RuntimeError(
                "The saved FAISS index dimension does not match "
                "the configured Ollama embedding model. Run ingestion again."
            )
