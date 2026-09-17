from pathlib import Path
import json, random
import numpy as np
import torch, yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def result_path(name):
    p = ROOT / "results"; p.mkdir(exist_ok=True); return p / name

def model_path(name):
    p = ROOT / "models" / name; p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj, name):
    with open(result_path(name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
