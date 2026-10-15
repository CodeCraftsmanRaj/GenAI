from pathlib import Path
def write_evaluation_checklist(path: Path, meta: dict):
    image=meta.get("stages",{}).get("image",{}); audio=meta.get("stages",{}).get("audio",{}); video=meta.get("stages",{}).get("video",{})
    text=f'''# Experiment 10 — Evaluation Checklist

Tick items only after inspecting/listening to the actual outputs.

## Image
- [ ] Image matches the prompt and shared theme.
- [ ] Composition works in a 16:9 video.
- [ ] Representation is respectful and inclusive.
- [ ] No facial appearance is presented as evidence of a genetic condition.
- [ ] No obvious unwanted artefacts.

## Music
- [ ] Instrumental mood is calm, empathetic, and hopeful.
- [ ] No unintended speech or vocals.
- [ ] No obvious clipping or severe audio artefacts.
- [ ] Music volume does not distract from the message.

## Video
- [ ] MP4 plays fully.
- [ ] Title and AI-generated disclosure are readable.
- [ ] Captions are readable and timed correctly.
- [ ] Motion is subtle.
- [ ] Audio and video are synchronized.
- [ ] Message says facial appearance alone cannot establish a genetic diagnosis.
- [ ] Synthetic content is not presented as genuine patient evidence.

## Recorded metadata
- Image model: {image.get('model','not recorded')}
- Image dimensions: {image.get('width','?')} x {image.get('height','?')}
- Music model: {audio.get('model','not recorded')}
- Music duration: {audio.get('duration_seconds','?')} seconds
- Video: {video.get('path','not recorded')}

## Observations to complete after inspection
- Image relevance and quality:
- Music mood and quality:
- Cross-modal consistency:
- Caption timing/readability:
- Artefacts or limitations:
- Ethical review:
'''
    path.write_text(text,encoding="utf-8")
