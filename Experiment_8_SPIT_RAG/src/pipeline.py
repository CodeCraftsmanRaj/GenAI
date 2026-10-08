from pathlib import Path
import json
import time

from .config import load_config, project_path
from .documents import load_documents, chunk_documents
from .retriever import VectorRetriever
from .generator import generate_answer


class RAGPipeline:
    def __init__(self, config=None):
        self.config = config or load_config()

        self.kb_root = project_path(
            self.config["knowledge_base"]["root"],
            self.config,
        )
        self.index_dir = project_path(
            self.config["retrieval"]["index_dir"],
            self.config,
        )

        retrieval_config = self.config["retrieval"]
        llm_config = self.config["llm"]

        self.retriever = VectorRetriever(
            model_name=retrieval_config["embedding_model"],
            base_url=retrieval_config.get(
                "embedding_url",
                llm_config.get(
                    "base_url",
                    "http://localhost:11434",
                ),
            ),
        )

    def ingest(self):
        docs = load_documents(self.kb_root)

        if not docs:
            raise RuntimeError(
                f"No supported documents found under {self.kb_root}. "
                "Add PDF, TXT, MD, HTML, CSV or JSON files."
            )

        knowledge_config = self.config["knowledge_base"]

        chunks = chunk_documents(
            docs,
            knowledge_config["chunk_size_words"],
            knowledge_config["chunk_overlap_words"],
        )

        self.retriever.build(chunks)
        self.retriever.save(self.index_dir)

        output_dir = project_path(
            self.config["project"]["output_dir"],
            self.config,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        metadata = {
            "documents": len(docs),
            "chunks": len(chunks),
            "sources": [doc["source_path"] for doc in docs],
            "embedding_provider": "ollama",
            "embedding_model": retrieval_config_value(
                self.config, "embedding_model"
            ),
        }

        (output_dir / "corpus_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"Ingested {len(docs)} documents into {len(chunks)} chunks."
        )
        print(f"Vector index: {self.index_dir}")
        print(f"Metadata: {output_dir / 'corpus_metadata.json'}")

    def load_index(self):
        index_path = self.index_dir / "vectors.faiss"

        if not index_path.exists():
            raise RuntimeError(
                "Vector index missing. Run: uv run main.py ingest"
            )

        self.retriever.load(self.index_dir)

    def retrieve(self, question, top_k=None):
        if self.retriever.index is None:
            self.load_index()

        k = (
            top_k
            if top_k is not None
            else self.config["retrieval"]["top_k"]
        )

        return self.retriever.search(question, k)

    def ask(self, question, top_k=None, generate=True):
        started = time.perf_counter()
        chunks = self.retrieve(question, top_k)

        threshold = float(
            self.config["retrieval"].get("similarity_threshold", 0.0)
        )

        if not chunks or chunks[0]["score"] < threshold:
            return {
                "question": question,
                "answer": (
                    "I could not find sufficiently relevant evidence "
                    "in the available knowledge base."
                ),
                "retrieved_chunks": chunks,
                "latency_seconds": time.perf_counter() - started,
                "refused": True,
            }

        if generate:
            answer = generate_answer(
                question,
                chunks,
                self.config,
            )
        else:
            answer = (
                "Generation disabled; inspect the retrieved evidence below."
            )

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks,
            "latency_seconds": time.perf_counter() - started,
            "refused": False,
        }


def retrieval_config_value(config, key):
    return config["retrieval"].get(key)
