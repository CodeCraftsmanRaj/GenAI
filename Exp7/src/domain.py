from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_domain_information():
    with open(ROOT / "data" / "domain_information.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["domain_information"]
