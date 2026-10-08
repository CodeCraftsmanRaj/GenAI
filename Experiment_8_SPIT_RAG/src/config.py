from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_root"] = str(ROOT)
    return config

def project_path(value: str, config: dict) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(config["_root"]) / p
