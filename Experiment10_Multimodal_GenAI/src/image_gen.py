import logging
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image


def choose_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return value


def generate_image(config, output_path: Path):
    """Generate the awareness artwork with the configured original model."""
    c = config["models"]["image"]
    device = choose_device(c.get("device", "auto"))
    dtype = torch.float16 if device == "cuda" else torch.float32

    logging.info("Loading image model %s on %s", c["name"], device)
    kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["variant"] = "fp16"

    pipe = AutoPipelineForText2Image.from_pretrained(c["name"], **kwargs).to(device)
    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            logging.debug("Attention slicing unavailable; continuing.")

    # Stable Diffusion Turbo is distilled for very few inference steps and uses
    # guidance_scale=0 by design. Keep these model-specific settings unchanged.
    generator = torch.Generator(device=device).manual_seed(
        int(config["project"].get("seed", 42))
    )
    prompt = config["prompts"]["image"]

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                width=int(c["width"]),
                height=int(c["height"]),
                num_inference_steps=int(c.get("num_inference_steps", 4)),
                guidance_scale=float(c.get("guidance_scale", 0.0)),
                generator=generator,
            )
        image = result.images[0]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)

        info = {
            "model": c["name"],
            "prompt": prompt,
            "path": str(output_path),
            "width": image.width,
            "height": image.height,
            "device": device,
            "num_inference_steps": int(c.get("num_inference_steps", 4)),
        }
        logging.info("Saved image: %s (%sx%s)", output_path, image.width, image.height)
        return info
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
