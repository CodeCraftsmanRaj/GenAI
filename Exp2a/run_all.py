import subprocess
import yaml


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    for name in cfg["experiment"]["models"]:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        subprocess.run(["python", "train.py", "--model", name], check=True)


if __name__ == "__main__":
    main()
