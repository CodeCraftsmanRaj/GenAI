import subprocess
import sys


def run(*args):
    print("\n>>>", " ".join(args))
    subprocess.run(
        [sys.executable, *args],
        check=True,
    )


def main():

    # 1. Dataset preparation
    run("src/prepare_data.py")

    # 2. Tokenization analysis
    run("src/tokenization.py")

    # 3. Baseline evaluation
    run("src/baseline.py")

    # 4. LoRA rank experiments
    for rank in (4, 8):
        run(
            "src/train_lora.py",
            "--rank",
            str(rank),
        )

        run(
            "src/evaluate.py",
            "--rank",
            str(rank),
        )

    # 5. Compare LoRA ranks
    run("src/rank_comparison.py")

    print("\n========================================")
    print(" EXPERIMENT 5 COMPLETED")
    print("========================================")
    print("\nCheck the results/ directory.")


if __name__ == "__main__":
    main()