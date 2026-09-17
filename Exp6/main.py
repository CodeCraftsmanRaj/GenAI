import argparse
from pathlib import Path
from src.pipeline import run_all


def main():
    parser = argparse.ArgumentParser(description="EXP6: BERT sentence classification lab")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Use a small subset and fewer epochs for a smoke test.")
    args = parser.parse_args()

    run_all(Path(args.config), quick=args.quick)


if __name__ == "__main__":
    main()