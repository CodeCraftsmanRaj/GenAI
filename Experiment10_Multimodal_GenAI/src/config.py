from pathlib import Path
import yaml
def load_config(path="config.yaml"):
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f: cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict): raise ValueError("config.yaml must contain a YAML mapping")
    return cfg
