import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from src.image_gen import choose_device


def generate_audio(config, output_path: Path):
    """Generate, peak-normalize, and save the instrumental soundtrack."""
    c = config["models"]["audio"]
    device = choose_device(c.get("device", "auto"))
    dtype = torch.float16 if device == "cuda" else torch.float32

    logging.info("Loading audio model %s on %s", c["name"], device)
    processor = AutoProcessor.from_pretrained(c["name"])
    model = MusicgenForConditionalGeneration.from_pretrained(
        c["name"], torch_dtype=dtype
    ).to(device).eval()

    prompt = config["prompts"]["music"]
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    max_tokens = int(c.get("max_new_tokens", 1024))

    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_tokens)

        audio = generated[0, 0].detach().float().cpu().numpy()
        if audio.size == 0:
            raise RuntimeError("MusicGen returned empty audio.")

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-8:
            # Keep headroom and avoid saving a near-silent WAV.
            audio = audio * (0.90 / peak)
        else:
            raise RuntimeError("MusicGen returned an effectively silent waveform.")

        sample_rate = int(model.config.audio_encoder.sampling_rate)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")

        info = {
            "model": c["name"],
            "prompt": prompt,
            "path": str(output_path),
            "sample_rate": sample_rate,
            "channels": 1,
            "duration_seconds": round(len(audio) / sample_rate, 3),
            "device": device,
            "max_new_tokens": max_tokens,
            "peak_amplitude_after_normalization": round(
                float(np.max(np.abs(audio))), 4
            ),
        }
        logging.info(
            "Saved audio: %s (%.2f sec; peak %.3f)",
            output_path,
            len(audio) / sample_rate,
            float(np.max(np.abs(audio))),
        )
        return info
    finally:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
