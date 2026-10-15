import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_audioclips, vfx

def _font(size, bold=False):
    paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for p in paths:
        if Path(p).exists():
            try: return ImageFont.truetype(p, size)
            except Exception: pass
    return ImageFont.load_default()

def _title_overlay(title, disclosure, w, h):
    im = Image.new("RGBA",(w,h),(0,0,0,0)); d=ImageDraw.Draw(im)
    d.rounded_rectangle((28,24,w-28,130), radius=18, fill=(6,15,38,185))
    d.text((52,36), title, font=_font(42,True), fill="white")
    d.text((54,91), disclosure, font=_font(19,True), fill=(205,225,255,255))
    return np.array(im)

def _caption_overlay(text, w, h):
    im=Image.new("RGBA",(w,h),(0,0,0,0)); d=ImageDraw.Draw(im); font=_font(32,True)
    words=text.split(); lines=[]; cur=""
    for word in words:
        test=(cur+" "+word).strip()
        if d.textbbox((0,0),test,font=font)[2] <= w-100: cur=test
        else:
            if cur: lines.append(cur)
            cur=word
    if cur: lines.append(cur)
    lh=42; y0=h-(len(lines)*lh+62)
    d.rounded_rectangle((32,y0,w-32,h-20),radius=18,fill=(0,0,0,175))
    for i,line in enumerate(lines):
        box=d.textbbox((0,0),line,font=font); x=(w-(box[2]-box[0]))/2
        d.text((x,y0+14+i*lh),line,font=font,fill="white")
    return np.array(im)

def _srt_time(t):
    ms=int(round(float(t)*1000)); hh=ms//3600000; ms%=3600000
    mm=ms//60000; ms%=60000; ss=ms//1000; ms%=1000
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"

def write_srt(config, path: Path):
    lines=[]
    for i,c in enumerate(config["video"]["captions"],1):
        lines += [str(i), f"{_srt_time(c['start'])} --> {_srt_time(c['end'])}", c["text"], ""]
    path.write_text("\n".join(lines),encoding="utf-8")

def compose_video(config, image_path: Path, audio_path: Path, output_path: Path):
    if not image_path.exists(): raise FileNotFoundError(f"Missing {image_path}; generate image first.")
    if not audio_path.exists(): raise FileNotFoundError(f"Missing {audio_path}; generate audio first.")
    c=config["video"]; dur=float(c["duration_seconds"]); fps=int(c["fps"]); w=int(c["width"]); h=int(c["height"])
    im=Image.open(image_path).convert("RGB"); ratio=w/h
    if im.width/im.height>ratio:
        nw=int(im.height*ratio); x=(im.width-nw)//2; im=im.crop((x,0,x+nw,im.height))
    else:
        nh=int(im.width/ratio); y=(im.height-nh)//2; im=im.crop((0,y,im.width,y+nh))
    im=im.resize((w,h),Image.Resampling.LANCZOS)
    bg=ImageClip(np.array(im)).with_duration(dur)
    # A slow zoom is applied with MoviePy's resize effect; compose at fixed output dimensions.
    zoom=bg.with_effects([vfx.Resize(lambda t: 1.0+0.035*(t/max(dur,0.01)))])
    zoom=zoom.with_position("center")
    base=CompositeVideoClip([zoom],size=(w,h)).with_duration(dur)
    title=ImageClip(_title_overlay(c["title"],c["disclosure"],w,h),transparent=True).with_duration(dur)
    layers=[base,title]
    for cap in c["captions"]:
        start=float(cap["start"]); end=min(float(cap["end"]),dur)
        if end<=start: continue
        layers.append(ImageClip(_caption_overlay(cap["text"],w,h),transparent=True).with_start(start).with_duration(end-start))
    video=CompositeVideoClip(layers,size=(w,h)).with_duration(dur)
    audio=AudioFileClip(str(audio_path))
    if audio.duration < dur:
        audio=concatenate_audioclips([audio]*int(np.ceil(dur/audio.duration))).with_duration(dur)
    else: audio=audio.subclipped(0,dur)
    audio=audio.with_volume_scaled(float(c.get("audio",{}).get("volume",0.28)))
    fade=float(c.get("audio",{}).get("fade_out_seconds",1.5))
    if fade>0:
        try: audio=audio.with_effects([vfx.AudioFadeOut(fade)])
        except Exception: pass
    video=video.with_audio(audio)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    video.write_videofile(str(output_path),fps=fps,codec="libx264",audio_codec="aac",preset="medium",threads=4,logger="bar")
    video.close(); audio.close()
    logging.info("Saved video: %s",output_path)
    return {"path":str(output_path),"duration_seconds":dur,"resolution":f"{w}x{h}","fps":fps,
      "captions":len(c["captions"]),"image_path":str(image_path),"audio_path":str(audio_path),
      "note":"Composed from generated image and music with titles, captions, and subtle motion; no separate video model used."}
