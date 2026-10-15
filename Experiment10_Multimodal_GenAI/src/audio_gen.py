import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from src.image_gen import choose_device

def generate_audio(config, output_path: Path):
    c = config["models"]["audio"]; device = choose_device(c.get("device","auto"))
    dtype = torch.float16 if device == "cuda" else torch.float32
    logging.info("Loading %s on %s", c["name"], device)
    processor = AutoProcessor.from_pretrained(c["name"])
    model = MusicgenForConditionalGeneration.from_pretrained(c["name"], torch_dtype=dtype).to(device).eval()
    prompt = config["prompts"]["music"]
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k:v.to(device) for k,v in inputs.items()}
    max_tokens = int(c.get("max_new_tokens", 512))
    with torch.inference_mode(): values = model.generate(**inputs, max_new_tokens=max_tokens)
    audio = values[0,0].detach().float().cpu().numpy()
    if not audio.size: raise RuntimeError("MusicGen returned empty audio.")
    audio = np.nan_to_num(audio)
    sr = int(model.config.audio_encoder.sampling_rate)
    output_path.parent.mkdir(parents=True, exist_ok=True); sf.write(output_path, audio, sr)
    info = {"model": c["name"], "prompt": prompt, "path": str(output_path),
      "sample_rate": sr, "channels": 1, "duration_seconds": round(len(audio)/sr,3),
      "device": device, "max_new_tokens": max_tokens}
    logging.info("Saved audio: %s (%.2f sec)", output_path, len(audio)/sr)
    del model, processor
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return info
