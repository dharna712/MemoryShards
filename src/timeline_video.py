"""
Exports a reconstructed timeline as an animated MP4.

Input is the list that build_timeline.build_timeline() returns. For every event
the video shows its photos in turn with a slow Ken Burns zoom/pan, an overlay
with the event date, place and caption, and a small lat/lon scatter in the
corner whose current point is highlighted. There is no map: the scatter is just
the events' coordinates scaled to the bounding box of the whole timeline.

Captions are drawn exactly as the captioning model produced them; nothing is
corrected or rewritten here.

Usage:
    python src/timeline_video.py PHOTOS_DIR OUT.mp4 [--no-caption] [--seconds 75]
Needs Pillow, numpy and ffmpeg on PATH.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H = 1280, 720
FPS = 24
FADE_S = 0.5
ZOOM = 1.2
MIN_TOTAL_S, MAX_TOTAL_S = 60.0, 90.0
MIN_SHOT_S, MAX_SHOT_S = 2.0, 6.0

_FONT_CANDIDATES = ["arial.ttf", "DejaVuSans.ttf", "Arial.ttf"]


def _font(size):
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(draw, text, font, max_w):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines[:3]


def plan_shots(timeline, total_seconds=75.0):
    """One shot per photo, keeping chronological order. If there are too many
    photos for the 2 s minimum per shot, photos are dropped evenly but the first
    photo of every event is always kept. Returns (shots, seconds_per_shot)."""
    shots = [(i, p) for i, ev in enumerate(timeline) for p in ev["photos"]]
    total_seconds = min(max(total_seconds, MIN_TOTAL_S), MAX_TOTAL_S)
    max_shots = int(total_seconds // MIN_SHOT_S)
    if len(shots) > max_shots:
        firsts = {}
        for k, (i, _) in enumerate(shots):
            firsts.setdefault(i, k)
        keep = set(firsts.values())
        rest = [k for k in range(len(shots)) if k not in keep]
        room = max(max_shots - len(keep), 0)
        if room and rest:
            step = len(rest) / room
            keep |= {rest[int(j * step)] for j in range(room)}
        shots = [s for k, s in enumerate(shots) if k in keep]
    per = min(max(total_seconds / max(len(shots), 1), MIN_SHOT_S), MAX_SHOT_S)
    return shots, per


def _scatter_points(timeline):
    pts = [(ev["lat"], ev["lon"]) for ev in timeline if ev.get("lat") is not None]
    if not pts:
        return None
    lats, lons = zip(*pts)
    return min(lats), max(lats), min(lons), max(lons)


def _overlay(ev_index, ev, timeline, bounds):
    """Static RGBA layer for one event: caption bar plus the lat/lon scatter."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bar_h = 190
    d.rectangle([0, H - bar_h, W, H], fill=(10, 10, 14, 175))

    when = ev["start_time"].strftime("%d %b %Y, %I:%M %p") if ev["start_time"] else "Undated"
    d.text((40, H - bar_h + 18), when, font=_font(30), fill=(240, 190, 90, 255))
    d.text((40, H - bar_h + 58), ev["place"], font=_font(38), fill=(255, 255, 255, 255))
    if ev.get("caption"):
        cap_font = _font(26)
        for n, line in enumerate(_wrap(d, ev["caption"], cap_font, W - 80)):
            d.text((40, H - bar_h + 108 + n * 30), line, font=cap_font, fill=(220, 220, 225, 255))

    if bounds:
        bx, by, bw, bh = W - 220, 24, 190, 130
        d.rectangle([bx, by, bx + bw, by + bh], fill=(10, 10, 14, 150))
        la0, la1, lo0, lo1 = bounds
        for j, other in enumerate(timeline):
            if other.get("lat") is None:
                continue
            fx = 0.5 if lo1 == lo0 else (other["lon"] - lo0) / (lo1 - lo0)
            fy = 0.5 if la1 == la0 else (other["lat"] - la0) / (la1 - la0)
            x, y = bx + 15 + fx * (bw - 30), by + bh - 15 - fy * (bh - 30)
            if j == ev_index:
                d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(240, 190, 90, 255))
            else:
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(200, 200, 205, 200))
    return layer


def _ken_burns_frames(path, n_frames, flip):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    bw, bh = int(W * ZOOM), int(H * ZOOM)
    scale = max(bw / img.width, bh / img.height)
    img = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1), Image.BILINEAR)
    cx, cy = img.width / 2, img.height / 2
    for f in range(n_frames):
        t = f / max(n_frames - 1, 1)
        z = 1.0 + (ZOOM - 1.0) * t            # zoom in
        cw, ch = bw / z, bh / z
        # drift sideways a little, alternating direction per shot
        shift = (t - 0.5) * (bw - cw) * (1 if flip else -1)
        x0 = cx - cw / 2 + shift
        y0 = cy - ch / 2
        x0 = min(max(x0, 0), img.width - cw)
        y0 = min(max(y0, 0), img.height - ch)
        # frame is W x H with the same aspect as the crop
        crop = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
        yield crop.resize((W, H), Image.BILINEAR)


def render_timeline_video(timeline, out_path, total_seconds=75.0):
    """Renders the timeline to out_path (MP4, H.264). Returns a small summary dict."""
    if not timeline:
        raise ValueError("timeline is empty")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    shots, per = plan_shots(timeline, total_seconds)
    bounds = _scatter_points(timeline)
    n_frames = int(round(per * FPS))
    n_fade = int(FADE_S * FPS)
    overlays = {}

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    prev_last = None
    try:
        for k, (ev_i, photo) in enumerate(shots):
            if ev_i not in overlays:
                overlays[ev_i] = _overlay(ev_i, timeline[ev_i], timeline, bounds)
            ov = overlays[ev_i]
            frame = None
            for f, base in enumerate(_ken_burns_frames(photo, n_frames, k % 2 == 0)):
                frame = base.convert("RGBA")
                frame.alpha_composite(ov)
                frame = frame.convert("RGB")
                if prev_last is not None and f < n_fade:
                    frame = Image.blend(prev_last, frame, (f + 1) / n_fade)
                proc.stdin.write(np.asarray(frame).tobytes())
            prev_last = frame
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    except BaseException:
        proc.kill()
        raise
    return {"shots": len(shots), "seconds": round(len(shots) * n_frames / FPS, 1), "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser(description="Export a reconstructed timeline as an MP4.")
    ap.add_argument("photos_dir")
    ap.add_argument("out")
    ap.add_argument("--no-caption", action="store_true", help="skip BLIP captions (faster)")
    ap.add_argument("--seconds", type=float, default=75.0, help="target length, clamped to 60-90")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_timeline import build_timeline

    timeline, _ = build_timeline(args.photos_dir, caption=not args.no_caption)
    print(render_timeline_video(timeline, args.out, args.seconds))


if __name__ == "__main__":
    main()
