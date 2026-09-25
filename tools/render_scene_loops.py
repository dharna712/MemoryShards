"""
Renders the extra ambient background loops (original artwork, no footage):

    scan      a scanner sweeping across photo frames        -> try page
    converge  shards drifting together into one photograph  -> closing call to action
    frames    contact-sheet strips of frames scrolling      -> "No manual albums" scene

Each is 8 seconds and loops seamlessly (every motion uses whole cycles).
Shares its palette and helpers with render_hero_loop.py, including --light.

Usage:
    python tools/render_scene_loops.py [--light] [scan|converge|frames|all]
Needs numpy, Pillow and ffmpeg on PATH.
"""

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import render_hero_loop as hl  # reads --light from argv and defines the palette

W, H = hl.W, hl.H
FPS, SECONDS = 30, 8
FRAMES = FPS * SECONDS
TAU = math.tau
OUT_DIR = hl.OUT_DIR
AMBER, CREAM, CRACK = hl.AMBER, hl.CREAM, hl.CRACK
FILL, EDGE = hl.FILL_GAIN, hl.EDGE_GAIN


def new_canvas():
    return Image.fromarray(hl.base.astype(np.uint8), "RGB").convert("RGBA")


def finish(img, layer, rng):
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(0.5)))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) + rng.normal(0, 1.4, (H, W, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def rot_rect(cx, cy, w, h, tilt):
    c, s = math.cos(tilt), math.sin(tilt)
    return [(cx + dx * c - dy * s, cy + dx * s + dy * c)
            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


# ---------------------------------------------------------------- scan
def make_scan():
    rng = np.random.default_rng(21)
    frames = []
    for gx in range(5):
        for gy in range(3):
            landscape = rng.random() > 0.35
            w, h = (230, 155) if landscape else (150, 215)
            frames.append({
                "cx": 130 + gx * 250 + rng.uniform(-30, 30), "cy": 130 + gy * 230 + rng.uniform(-25, 25),
                "w": w, "h": h, "tilt": rng.uniform(-0.12, 0.12), "ax": rng.uniform(6, 16),
                "ay": rng.uniform(6, 14), "ph": rng.uniform(0, TAU),
            })
    xs = np.arange(W, dtype=np.float32)

    def draw(f):
        t = f / FRAMES
        rng2 = np.random.default_rng(f)
        img = new_canvas()
        sx = -200 + (W + 400) * t
        band = np.exp(-(((xs - sx) / 90.0) ** 2)) * 0.13 * hl.LEAK_GAIN
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        arr[..., 0], arr[..., 1], arr[..., 2] = AMBER
        arr[..., 3] = (band[None, :] * 255).astype(np.uint8)
        img.alpha_composite(Image.fromarray(arr, "RGBA"))
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for fr in frames:
            cx = fr["cx"] + fr["ax"] * math.cos(TAU * t + fr["ph"])
            cy = fr["cy"] + fr["ay"] * math.sin(TAU * t + fr["ph"])
            near = math.exp(-(((cx - sx) / 75.0) ** 2))
            pts = rot_rect(cx, cy, fr["w"], fr["h"], fr["tilt"])
            d.polygon(pts, fill=CREAM + (int((8 + 26 * near) * FILL),))
            d.line(pts + [pts[0]], fill=CREAM + (min(255, int((36 + 60 * near) * EDGE)),), width=1)
            if near > 0.15:  # corner brackets while the scanner passes
                for (x, y), (dx, dy) in zip(pts, ((1, 1), (-1, 1), (-1, -1), (1, -1))):
                    a = int(230 * near)
                    d.line([(x, y), (x + 14 * dx, y)], fill=AMBER + (a,), width=2)
                    d.line([(x, y), (x, y + 14 * dy)], fill=AMBER + (a,), width=2)
        d.line([(sx, 0), (sx, H)], fill=AMBER + (150,), width=2)
        return finish(img, layer, rng2)

    return draw, "scene-scan"


# ---------------------------------------------------------------- converge
def make_converge():
    rng = np.random.default_rng(33)
    rw, rh, cx0, cy0 = 520, 340, W / 2, H / 2
    hub = (0.58, 0.42)
    ring = [(0, 0), (0.35, 0), (1, 0), (1, 0.3), (1, 1), (0.65, 1), (0, 1), (0, 0.65)]
    left, top = cx0 - rw / 2, cy0 - rh / 2
    to_px = lambda p: (left + p[0] * rw, top + p[1] * rh)
    pieces = []
    for i in range(8):
        tri = [to_px(hub), to_px(ring[i]), to_px(ring[(i + 1) % 8])]
        mid = ((tri[0][0] + tri[1][0] + tri[2][0]) / 3, (tri[0][1] + tri[1][1] + tri[2][1]) / 3)
        ang = math.atan2(mid[1] - cy0, mid[0] - cx0) + rng.uniform(-0.4, 0.4)
        dist = rng.uniform(170, 330)
        pieces.append({"tri": tri, "mid": mid, "dx": math.cos(ang) * dist, "dy": math.sin(ang) * dist * 0.75,
                       "rot": rng.uniform(-0.55, 0.55), "shade": rng.uniform(0.7, 1.2)})
    dust = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.8, 2.0), rng.uniform(0, TAU)) for _ in range(60)]
    glow = hl.radial_blob(480, AMBER, 0.20 * hl.LEAK_GAIN)

    def draw(f):
        t = f / FRAMES
        rng2 = np.random.default_rng(f)
        c = 0.5 - 0.5 * math.cos(TAU * t)          # 0 scattered ... 1 assembled ... 0
        u = smoothstep((c - 0.12) / 0.7)
        img = new_canvas()
        glow_layer = glow.copy()
        glow_layer.putalpha(glow_layer.getchannel("A").point(lambda a: int(a * u)))
        img.alpha_composite(glow_layer, (int(cx0 - glow.width / 2), int(cy0 - glow.height / 2)))
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for p in pieces:
            ox, oy = p["dx"] * (1 - u), p["dy"] * (1 - u)
            rot = p["rot"] * (1 - u)
            cs, sn = math.cos(rot), math.sin(rot)
            pts = []
            for x, y in p["tri"]:
                rx, ry = x - p["mid"][0], y - p["mid"][1]
                pts.append((p["mid"][0] + ox + rx * cs - ry * sn, p["mid"][1] + oy + rx * sn + ry * cs))
            d.polygon(pts, fill=CREAM + (int((10 + 22 * u) * p["shade"] * FILL),))
            d.line(pts + [pts[0]], fill=AMBER + (min(255, int((60 + 120 * u) * EDGE)),), width=1)
        if u > 0.6:
            frame_pts = rot_rect(cx0, cy0, rw, rh, 0)
            d.line(frame_pts + [frame_pts[0]], fill=CREAM + (int(70 * (u - 0.6) / 0.4 * EDGE),), width=1)
        for x, y, r, ph in dust:
            yy = (y - H * t) % H
            a = int(255 * 0.35 * hl.DUST_GAIN * (0.6 + 0.4 * math.sin(TAU * 2 * t + ph)))
            d.ellipse([x - r, yy - r, x + r, yy + r], fill=AMBER + (a,))
        return finish(img, layer, rng2)

    return draw, "scene-converge"


# ---------------------------------------------------------------- frames
def make_frames():
    rng = np.random.default_rng(45)
    fw, fh, gap = 210, 140, 34
    step = fw + gap
    strips = [(120, 1), (360, -1), (600, 1)]
    count = W // step + 3
    tints = [[rng.uniform(0.6, 1.3) for _ in range(count)] for _ in strips]

    def draw(f):
        t = f / FRAMES
        rng2 = np.random.default_rng(f)
        img = new_canvas()
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for si, (y, direction) in enumerate(strips):
            shift = direction * step * t  # one frame of travel per loop
            d.rectangle([0, y - fh / 2 - 22, W, y + fh / 2 + 22], fill=CREAM + (int(9 * FILL),))
            for k in range(-1, count):
                x = k * step + shift + 30
                d.rectangle([x, y - fh / 2, x + fw, y + fh / 2], fill=CREAM + (int(12 * tints[si][k % count] * FILL),),
                            outline=CREAM + (int(58 * EDGE),))
                for hx in range(int(x) + 8, int(x + fw - 6), 22):  # film sprocket holes
                    d.rectangle([hx, y - fh / 2 - 16, hx + 9, y - fh / 2 - 8], fill=CREAM + (int(40 * EDGE),))
                    d.rectangle([hx, y + fh / 2 + 8, hx + 9, y + fh / 2 + 16], fill=CREAM + (int(40 * EDGE),))
        sweep = (t * (W + 500)) - 250
        d.rectangle([sweep - 2, 0, sweep + 2, H], fill=AMBER + (70,))
        return finish(img, layer, rng2)

    return draw, "scene-frames"


SCENES = {"scan": make_scan, "converge": make_converge, "frames": make_frames}


def render(name):
    draw, stem = SCENES[name]()
    stem += hl.SUFFIX
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    try:
        for f in range(FRAMES):
            draw(f).save(tmp / f"f{f:04d}.png")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(OUT_DIR / f"{stem}.mp4"),
        ], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", OUT_DIR / f"{stem}.mp4")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for scene in (list(SCENES) if not args or args[0] == "all" else args):
        render(scene)
