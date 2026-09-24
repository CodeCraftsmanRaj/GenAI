from pathlib import Path
import csv
import yaml

from .config import load_config
from .domain import load_domain_information
from .llm import BankingChatbot

ROOT = Path(__file__).resolve().parents[1]

def load_questions():
    with open(ROOT / "data" / "evaluation_questions.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]

def run_evaluation():
    config = load_config()
    domain = load_domain_information()
    questions = load_questions()
    bot = BankingChatbot(config)

    rows = []
    for item in questions:
        print(f'[{item["id"]}/10] {item["question"]}')
        response = bot.answer(item["question"], domain, [])
        print(response)
        print("-" * 70)
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "information_available": item["information_available"],
            "response": response,
            "correct": "",
            "hallucination_or_unsupported_claim": "",
        })

    output = ROOT / "evaluation_results.csv"
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evaluation results to: {output}")

if __name__ == "__main__":
    run_evaluation()
