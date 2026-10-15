import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    concatenate_audioclips,
    afx,
)

logger = logging.getLogger(__name__)
_RESAMPLING = getattr(Image, "Resampling", Image)
_LANCZOS = getattr(_RESAMPLING, "LANCZOS", getattr(Image, "LANCZOS", 1))


def _font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except (OSError, ValueError):
                pass
    return ImageFont.load_default()


def _wrap(draw, text, font, max_width):
    words = str(text).split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _fit_font(draw, text, max_width, start_size, min_size=16):
    size = start_size
    while size > min_size:
        font = _font(size, True)
        bbox = draw.textbbox((0, 0), str(text), font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 2
    return _font(min_size, True)


def _title_overlay(title, disclosure, width, height):
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = max(28, int(width * 0.035))
    panel_top = max(20, int(height * 0.035))
    panel_height = max(112, int(height * 0.19))
    draw.rounded_rectangle(
        (margin, panel_top, width - margin, panel_top + panel_height),
        radius=22,
        fill=(7, 18, 42, 220),
        outline=(160, 205, 245, 90),
        width=2,
    )
    text_width = width - 2 * (margin + 24)
    title_font = _fit_font(draw, title, text_width, 44, 22)
    draw.text(
        (margin + 24, panel_top + 18),
        str(title),
        font=title_font,
        fill=(255, 255, 255, 255),
    )
    disclosure_font = _font(22, True)
    disclosure_lines = _wrap(draw, disclosure, disclosure_font, text_width)
    if len(disclosure_lines) > 1:
        disclosure_font = _font(18, True)
        disclosure_lines = _wrap(draw, disclosure, disclosure_font, text_width)
    draw.text(
        (margin + 26, panel_top + 76),
        " ".join(disclosure_lines),
        font=disclosure_font,
        fill=(190, 225, 255, 255),
    )
    return np.asarray(image)


def _caption_overlay(text, width, height):
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    side_margin = max(40, int(width * 0.055))
    max_text_width = width - 2 * side_margin - 44
    font_size = max(28, min(36, int(height * 0.048)))
    font = _font(font_size, True)
    lines = _wrap(draw, text, font, max_text_width)

    line_height = int(font_size * 1.35)
    pad_y, pad_x = 18, 24
    panel_height = len(lines) * line_height + 2 * pad_y
    panel_bottom = height - max(26, int(height * 0.045))
    panel_top = panel_bottom - panel_height

    draw.rounded_rectangle(
        (side_margin, panel_top, width - side_margin, panel_bottom),
        radius=20,
        fill=(3, 8, 20, 220),
        outline=(255, 255, 255, 60),
        width=1,
    )
    for index, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = (width - line_width) / 2
        y = panel_top + pad_y + index * line_height
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 200),
        )
    return np.asarray(image)


def _srt_time(seconds):
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def write_srt(config, path: Path):
    """Write the exact same caption intervals/text used in the video."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = float(config["video"]["duration_seconds"])
    lines = []
    number = 1
    for caption in config["video"].get("captions", []):
        start = max(0.0, float(caption["start"]))
        end = min(duration, float(caption["end"]))
        if end <= start:
            continue
        lines.extend([
            str(number),
            f"{_srt_time(start)} --> {_srt_time(end)}",
            str(caption["text"]),
            "",
        ])
        number += 1
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved subtitles: %s", path)


def _prepare_image(image_path, width, height):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    target_ratio = width / height
    source_ratio = image.width / image.height
    if source_ratio > target_ratio:
        new_width = max(1, int(round(image.height * target_ratio)))
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    elif source_ratio < target_ratio:
        new_height = max(1, int(round(image.width / target_ratio)))
        top = (image.height - new_height) // 2
        image = image.crop((0, top, image.width, top + new_height))
    return image.resize((width, height), _LANCZOS)


def _background_clip(image, duration, zoom_amount):
    """Make a gentle zoom by cropping/resizing with Pillow, not MoviePy.resize()."""
    width, height = image.size
    pixels = np.asarray(image.convert("RGB"))
    zoom_amount = max(0.0, float(zoom_amount))

    def make_frame(t):
        progress = min(1.0, max(0.0, float(t) / max(duration, 0.001)))
        scale = 1.0 + zoom_amount * progress
        crop_width = max(1, int(round(width / scale)))
        crop_height = max(1, int(round(height / scale)))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        crop = pixels[top:top + crop_height, left:left + crop_width]
        return np.asarray(Image.fromarray(crop).resize((width, height), _LANCZOS))

    return VideoClip(make_frame=make_frame, duration=duration)


def compose_video(config, image_path: Path, audio_path: Path, output_path: Path):
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Missing image: {image_path}; generate image first.")
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Missing audio: {audio_path}; generate audio first.")

    c = config["video"]
    duration = float(c["duration_seconds"])
    fps = int(c["fps"])
    width, height = int(c["width"]), int(c["height"])
    if duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Video duration, FPS, width, and height must be positive.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video = None
    source_audio = None
    repeated_audio = None
    trimmed_audio = None
    normalized_audio = None
    volume_audio = None
    faded_audio = None

    try:
        image = _prepare_image(image_path, width, height)
        zoom_amount = float(c.get("motion", {}).get("zoom_amount", c.get("zoom_amount", 0.025)))
        background = _background_clip(image, duration, zoom_amount)
        title = ImageClip(
            _title_overlay(c.get("title", ""), c.get("disclosure", ""), width, height),
            transparent=True,
        ).set_duration(duration).set_position((0, 0))
        layers = [background, title]

        for caption in c.get("captions", []):
            start = max(0.0, float(caption["start"]))
            end = min(duration, float(caption["end"]))
            if end <= start:
                logger.warning("Skipping invalid caption interval: %s", caption)
                continue
            overlay = ImageClip(
                _caption_overlay(str(caption["text"]), width, height),
                transparent=True,
            ).set_start(start).set_duration(end - start).set_position((0, 0))
            layers.append(overlay)

        video = CompositeVideoClip(layers, size=(width, height)).set_duration(duration)

        # Load audio and normalize its peak before applying the configured mix level.
        # The old 0.28 multiplier left this exact output peaking at only -23.4 dBFS.
        source_audio = AudioFileClip(str(audio_path))
        if source_audio.duration is None or source_audio.duration <= 0:
            raise ValueError(f"Audio file is empty or invalid: {audio_path}")

        if source_audio.duration < duration:
            repeat_count = int(math.ceil(duration / source_audio.duration))
            repeated_audio = concatenate_audioclips([source_audio] * repeat_count)
            trimmed_audio = repeated_audio.subclip(0, duration)
        else:
            trimmed_audio = source_audio.subclip(0, duration)

        try:
            normalized_audio = trimmed_audio.fx(afx.audio_normalize)
        except Exception:
            logger.warning("MoviePy audio normalization unavailable; using original audio.", exc_info=True)
            normalized_audio = trimmed_audio

        volume = max(0.0, min(1.0, float(c.get("audio", {}).get("volume", 0.82))))
        volume_audio = normalized_audio.volumex(volume)
        fade_seconds = max(0.0, min(duration, float(c.get("audio", {}).get("fade_out_seconds", 1.5))))
        if fade_seconds > 0:
            faded_audio = volume_audio.fx(afx.audio_fadeout, fade_seconds)
        else:
            faded_audio = volume_audio
        video = video.set_audio(faded_audio)

        logger.info("Rendering %s (%dx%d, %d FPS, %.2f sec)", output_path, width, height, fps, duration)
        video.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            bitrate="5000k",
            preset="medium",
            threads=4,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            logger="bar",
        )
    finally:
        if video is not None:
            try:
                video.close()
            except Exception:
                logger.debug("Error closing video clip", exc_info=True)
        seen = set()
        # Close derived clips before their underlying source.
        for clip in [faded_audio, volume_audio, normalized_audio, trimmed_audio, repeated_audio, source_audio]:
            if clip is None or id(clip) in seen:
                continue
            seen.add(id(clip))
            try:
                clip.close()
            except Exception:
                logger.debug("Error closing audio clip", exc_info=True)

    logger.info("Saved video: %s", output_path)
    return {
        "path": str(output_path),
        "duration_seconds": duration,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "captions": len(c.get("captions", [])),
        "image_path": str(image_path),
        "audio_path": str(audio_path),
        "note": "Composed generated image and instrumental soundtrack with synced captions, title/disclosure overlays, subtle zoom, normalized audio, and H.264/AAC output.",
    }
