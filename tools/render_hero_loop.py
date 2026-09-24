"""
Renders the ambient hero background loop (web/media/hero-loop.mp4 + poster).

Original artwork, no third-party footage: slow-drifting translucent shards,
thin photo-frame outlines, dust, and amber light leaks on a warm near-black
field. Every moving element uses whole-number cycles over the loop length, so
the last frame flows straight into the first.

Usage:
    python tools/render_hero_loop.py
Needs numpy, Pillow and ffmpeg on PATH.
"""

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1280, 720
FPS = 30
SECONDS = 10
FRAMES = FPS * SECONDS
OUT_DIR = Path(__file__).resolve().parent.parent / "web" / "media"

BG = np.array([14, 11, 8], dtype=np.float32)
AMBER = (217, 163, 83)
CREAM = (240, 231, 214)
CRACK = (191, 228, 255)

rng = np.random.default_rng(7)
TAU = math.tau


def radial_blob(radius, color, strength):
    size = int(radius * 2)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt((x - radius) ** 2 + (y - radius) ** 2) / radius
    a = np.clip(1 - d, 0, 1) ** 2.2 * strength
    layer = np.zeros((size, size, 4), dtype=np.float32)
    layer[..., 0], layer[..., 1], layer[..., 2] = color
    layer[..., 3] = a * 255
    return Image.fromarray(layer.astype(np.uint8), "RGBA")


LEAKS = [
    # (blob image, centre x, centre y, orbit rx, orbit ry, phase, cycles)
    (radial_blob(520, AMBER, 0.16), 300, 220, 140, 80, 0.0, 1),
    (radial_blob(420, (255, 190, 120), 0.10), 1000, 520, 120, 100, 2.1, 1),
    (radial_blob(360, CRACK, 0.05), 1100, 160, 90, 60, 4.0, 2),
]

SHARDS = []
for i in range(16):
    cx, cy = rng.uniform(0, W), rng.uniform(0, H)
    r = rng.uniform(70, 260)
    ang = rng.uniform(0, TAU)
    pts = [(ang + k * TAU / 3 + rng.uniform(-0.5, 0.5), r * rng.uniform(0.6, 1.1)) for k in range(3)]
    SHARDS.append({
        "cx": cx, "cy": cy, "pts": pts, "depth": r / 260,
        "ax": rng.uniform(18, 60), "ay": rng.uniform(14, 44),
        "phase": rng.uniform(0, TAU), "cycles": int(rng.integers(1, 3)),
        "spin": rng.uniform(0.04, 0.14), "fill": rng.uniform(0.02, 0.06),
        "edge": AMBER if i % 3 else CRACK,
    })

FRAMES_OUTLINES = []
for i in range(6):
    w = rng.uniform(150, 300)
    FRAMES_OUTLINES.append({
        "cx": rng.uniform(80, W - 80), "cy": rng.uniform(60, H - 60), "w": w, "h": w * 0.68,
        "tilt": rng.uniform(-0.25, 0.25), "ax": rng.uniform(20, 50), "ay": rng.uniform(16, 36),
        "phase": rng.uniform(0, TAU), "cycles": int(rng.integers(1, 3)),
    })

DUST = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.8, 2.2), int(rng.integers(1, 3)),
         rng.uniform(0.15, 0.5), rng.uniform(0, TAU)) for _ in range(70)]

y_idx, x_idx = np.mgrid[0:H, 0:W].astype(np.float32)
vignette = 1 - 0.55 * np.clip(np.sqrt(((x_idx - W / 2) / (W / 2)) ** 2 + ((y_idx - H / 2) / (H / 2)) ** 2) - 0.35, 0, 1)
base = np.clip(BG[None, None, :] * vignette[..., None], 0, 255)


def draw_frame(f):
    t = f / FRAMES
    img = Image.fromarray(base.astype(np.uint8), "RGB").convert("RGBA")

    for blob, cx, cy, rx, ry, ph, cyc in LEAKS:
        px = cx + rx * math.cos(TAU * cyc * t + ph)
        py = cy + ry * math.sin(TAU * cyc * t + ph)
        img.alpha_composite(blob, (int(px - blob.width / 2), int(py - blob.height / 2)))

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for s in SHARDS:
        ox = s["ax"] * math.cos(TAU * s["cycles"] * t + s["phase"]) * (0.5 + s["depth"])
        oy = s["ay"] * math.sin(TAU * s["cycles"] * t + s["phase"]) * (0.5 + s["depth"])
        rot = s["spin"] * math.sin(TAU * t + s["phase"])
        poly = [(s["cx"] + ox + rad * math.cos(a + rot), s["cy"] + oy + rad * math.sin(a + rot)) for a, rad in s["pts"]]
        d.polygon(poly, fill=CREAM + (int(255 * s["fill"]),))
        d.line(poly + [poly[0]], fill=s["edge"] + (int(255 * (0.10 + 0.14 * s["depth"])),), width=1)

    for fr in FRAMES_OUTLINES:
        ox = fr["ax"] * math.cos(TAU * fr["cycles"] * t + fr["phase"])
        oy = fr["ay"] * math.sin(TAU * fr["cycles"] * t + fr["phase"])
        c, s_ = math.cos(fr["tilt"]), math.sin(fr["tilt"])
        corners = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            dx, dy = sx * fr["w"] / 2, sy * fr["h"] / 2
            corners.append((fr["cx"] + ox + dx * c - dy * s_, fr["cy"] + oy + dx * s_ + dy * c))
        d.polygon(corners, fill=CREAM + (7,))
        d.line(corners + [corners[0]], fill=CREAM + (34,), width=1)

    for x0, y0, r, laps, alpha, ph in DUST:
        y = (y0 - laps * H * t) % H
        x = x0 + 14 * math.sin(TAU * t + ph)
        a = int(255 * alpha * (0.6 + 0.4 * math.sin(TAU * 2 * t + ph)))
        d.ellipse([x - r, y - r, x + r, y + r], fill=AMBER + (a,))

    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(0.6)))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr += rng.normal(0, 1.4, arr.shape)  # dither, avoids banding in the dark gradients
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    try:
        for f in range(FRAMES):
            draw_frame(f).save(tmp / f"f{f:04d}.png")
        draw_frame(FRAMES // 4).save(OUT_DIR / "hero-poster.jpg", quality=82)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "29", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(OUT_DIR / "hero-loop.mp4"),
        ], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", OUT_DIR / "hero-loop.mp4")


if __name__ == "__main__":
    main()
