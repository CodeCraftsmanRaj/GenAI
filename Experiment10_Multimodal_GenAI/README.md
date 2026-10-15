# Experiment 10 — Multimodal Content Generation Using Generative AI

**Student:** Raj Mathuria | **UID:** 2023300139 | **Batch:** PE-C

## Aim
Generate an image and instrumental music from a shared theme, then combine them into a 15–30 second educational awareness video with captions, titles, and subtle motion.

## Recommended models (from the lab manual)
- Image: `stabilityai/sd-turbo`
- Music: `facebook/musicgen-small`
- Video composition: MoviePy / FFmpeg
- Inference backend: PyTorch (no TensorFlow required)

## Setup
Use a GPU runtime such as Google Colab when available. The first run downloads model weights and requires internet access. MusicGen Small can require substantial memory.

```bash
uv sync
```
Ensure FFmpeg is installed. On Ubuntu/Colab:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Run
```bash
uv run main.py
```
Optional:
```bash
uv run main.py --skip-image
uv run main.py --skip-audio
uv run main.py --skip-video
```
Skipped stages require the corresponding existing files in `results/`.

## Outputs
All generated artifacts are saved under `results/`:
- `generated_image.png`
- `generated_music.wav`
- `final_awareness_video.mp4`
- `captions.srt`
- `generation_metadata.json` (models, prompts, dimensions, durations, timing)
- `evaluation_checklist.md`
- `run.log`

A separate image-to-video model is not required by the lab manual. The video is composed from the generated image and music, with motion, title, captions, and audio synchronization.

## Ethical safeguards
The video states that AI can support research but cannot establish a genetic diagnosis from facial appearance alone. Do not use patient photos or present generated faces as real cases. Review outputs for representational bias, artefacts, model licensing, and responsible use.
