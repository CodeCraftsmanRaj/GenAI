import argparse, json, logging, random, time
from pathlib import Path
import numpy as np
import torch
from src.config import load_config
from src.image_gen import generate_image
from src.audio_gen import generate_audio
from src.video_compose import compose_video, write_srt
from src.evaluation import write_evaluation_checklist

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--skip-image", action="store_true")
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["project"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(out/"run.log", mode="w", encoding="utf-8"), logging.StreamHandler()])
    seed = int(cfg["project"].get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    meta = {"project": cfg["project"]["name"],
      "student": {k: cfg["project"].get(k) for k in ["student_name","uid","batch"]},
      "seed": seed, "prompts": cfg["prompts"],
      "models": {"image": cfg["models"]["image"]["name"], "audio": cfg["models"]["audio"]["name"]},
      "stages": {}}
    image = out/"generated_image.png"; audio = out/"generated_music.wav"; video = out/"final_awareness_video.mp4"
    t0 = time.perf_counter()
    if not args.skip_image:
        st = time.perf_counter(); meta["stages"]["image"] = {**generate_image(cfg, image), "generation_seconds": round(time.perf_counter()-st, 3)}
    else: meta["stages"]["image"] = {"skipped": True, "path": str(image)}
    if not args.skip_audio:
        st = time.perf_counter(); meta["stages"]["audio"] = {**generate_audio(cfg, audio), "generation_seconds": round(time.perf_counter()-st, 3)}
    else: meta["stages"]["audio"] = {"skipped": True, "path": str(audio)}
    if not args.skip_video:
        st = time.perf_counter(); meta["stages"]["video"] = {**compose_video(cfg, image, audio, video), "generation_seconds": round(time.perf_counter()-st, 3)}
        write_srt(cfg, out/"captions.srt")
    else: meta["stages"]["video"] = {"skipped": True}
    meta["total_runtime_seconds"] = round(time.perf_counter()-t0, 3)
    meta["device_info"] = {"cuda_available": torch.cuda.is_available(),
      "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "torch_version": torch.__version__}
    (out/"generation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_evaluation_checklist(out/"evaluation_checklist.md", meta)
    logging.info("All artifacts and metadata saved under %s", out.resolve())

if __name__ == "__main__": main()
