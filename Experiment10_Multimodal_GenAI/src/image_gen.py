import logging
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image

def choose_device(value):
    if value == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable.")
    return value

def generate_image(config, output_path: Path):
    c = config["models"]["image"]; device = choose_device(c.get("device","auto"))
    dtype = torch.float16 if device == "cuda" else torch.float32
    logging.info("Loading %s on %s", c["name"], device)
    kwargs = {"torch_dtype": dtype}
    if device == "cuda": kwargs["variant"] = "fp16"
    pipe = AutoPipelineForText2Image.from_pretrained(c["name"], **kwargs).to(device)
    if device == "cuda":
        try: pipe.enable_attention_slicing()
        except Exception: pass
    gen = torch.Generator(device=device).manual_seed(int(config["project"].get("seed",42)))
    prompt = config["prompts"]["image"]
    result = pipe(prompt=prompt, width=int(c["width"]), height=int(c["height"]),
      num_inference_steps=int(c["num_inference_steps"]), guidance_scale=float(c["guidance_scale"]), generator=gen)
    img = result.images[0]; output_path.parent.mkdir(parents=True, exist_ok=True); img.save(output_path)
    info = {"model": c["name"], "prompt": prompt, "path": str(output_path),
      "width": img.width, "height": img.height, "device": device,
      "num_inference_steps": int(c["num_inference_steps"])}
    logging.info("Saved image: %s (%sx%s)", output_path, img.width, img.height)
    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return info
