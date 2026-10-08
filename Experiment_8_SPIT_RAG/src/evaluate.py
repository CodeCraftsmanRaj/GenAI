import json, time
import pandas as pd
from pathlib import Path
from .pipeline import RAGPipeline
from .config import load_config, project_path

def load_qa(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows

def run_evaluation():
    cfg = load_config()
    qa_path = project_path(cfg["evaluation"]["qa_file"], cfg)
    rows = load_qa(qa_path)
    pipe = RAGPipeline(cfg); pipe.load_index()
    output = project_path(cfg["project"]["output_dir"], cfg); output.mkdir(parents=True, exist_ok=True)
    details, summary = [], []
    for k in cfg["evaluation"]["top_k_values"]:
        hits = 0; latencies = []
        for item in rows:
            t0 = time.perf_counter(); retrieved = pipe.retrieve(item["question"], k); elapsed = time.perf_counter()-t0
            expected = item.get("expected_source", "")
            hit = any(expected.lower() in c["source"].lower() or expected.lower() in c.get("source_path", "").lower() for c in retrieved) if expected else None
            if hit is True: hits += 1
            latencies.append(elapsed)
            details.append({"k": k, "question": item["question"], "expected_source": expected, "hit": hit, "retrieved_sources": "; ".join(c["source"] for c in retrieved), "top_score": retrieved[0]["score"] if retrieved else None, "retrieval_latency_seconds": elapsed})
        known = sum(1 for x in details if x["k"] == k and x["hit"] is not None)
        summary.append({"k": k, "questions": len(rows), "hit_rate": hits/known if known else None, "mean_retrieval_latency_seconds": sum(latencies)/len(latencies) if latencies else None})
    pd.DataFrame(details).to_csv(output / "topk_evaluation_details.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "topk_evaluation_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved evaluation CSVs to {output}")
