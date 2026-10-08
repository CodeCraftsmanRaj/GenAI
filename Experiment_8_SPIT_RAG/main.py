import argparse
import json
from pathlib import Path
from src.pipeline import RAGPipeline
from src.config import load_config, project_path

def cmd_ingest():
    RAGPipeline().ingest()

def cmd_ask(question: str, top_k: int | None):
    result = RAGPipeline().ask(question, top_k)
    print("\nANSWER\n" + result["answer"])
    print("\nRETRIEVED CHUNKS")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"\n[{i}] {chunk['source']} | {chunk.get('page_section')} | similarity={chunk['score']:.4f}\n{chunk['text'][:1200]}")
    print(f"\nLatency: {result['latency_seconds']:.3f}s")

def cmd_retrieve(question: str, top_k: int | None):
    chunks = RAGPipeline().retrieve(question, top_k)
    print(json.dumps(chunks, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description="SPIT-domain RAG chatbot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Extract, chunk, embed and index knowledge-base files")
    ask = sub.add_parser("ask", help="Ask a question and generate a grounded answer")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    retrieve = sub.add_parser("retrieve", help="Show retrieved chunks and scores without generation")
    retrieve.add_argument("question")
    retrieve.add_argument("--top-k", type=int, default=None)
    sub.add_parser("evaluate", help="Run top-k retrieval evaluation on evaluation/qa.jsonl")
    args = parser.parse_args()
    if args.command == "ingest": cmd_ingest()
    elif args.command == "ask": cmd_ask(args.question, args.top_k)
    elif args.command == "retrieve": cmd_retrieve(args.question, args.top_k)
    elif args.command == "evaluate":
        from src.evaluate import run_evaluation
        run_evaluation()

if __name__ == "__main__": main()
