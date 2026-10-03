"""
End-to-end test of the live API against real fixtures: POSTs photos to
/api/timeline exactly like the web page does and checks the response.

Covers: bad input handling, EXIF+GPS photos, undated face photos grouped into
people (clean, across decades of age, blurred/low-res, children), and the
checkpoint fallback. Face fixtures come from held-out data the model never
trained on; nothing here uses personal photos.

Usage (API must be running on localhost:5000):
    python src/test_pipeline_e2e.py
"""

import io
import os
import random
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from PIL import Image, ImageFilter

from extract_metadata import extract_photo_metadata
from face_data import RAW, scan_agedb, split_agedb

API = os.environ.get("API", "http://localhost:5000")
rng = random.Random(21)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def post(files):
    """files: [(filename, bytes)] -> (status, json)"""
    t0 = time.time()
    r = requests.post(f"{API}/api/timeline",
                      files=[("photos", (n, b, "image/jpeg")) for n, b in files], timeout=900)
    try:
        body = r.json()
    except Exception:
        body = {}
    return r.status_code, body, time.time() - t0


def jpeg(path_or_img, degrade=False):
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img).convert("RGB")
    if degrade:
        w, h = img.size
        img = img.filter(ImageFilter.GaussianBlur(2.5)).resize((max(w // 3, 48), max(h // 3, 48))).resize((w, h))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=18 if degrade else 92)
    return buf.getvalue()


def photo_people(body):
    """filename -> frozenset(person ids) (each undated photo is its own event)"""
    out = {}
    for ev in body["events"]:
        for fn in ev["filenames"]:
            out[fn] = frozenset(ev["people"])
    return out


def pairwise(truth, pred):
    """precision/recall over photo pairs; pred maps filename -> set of person ids (single face each)."""
    tp = fp = fn = 0
    for a, b in combinations(truth, 2):
        same_truth = truth[a] == truth[b]
        same_pred = bool(pred.get(a, frozenset()) & pred.get(b, frozenset()))
        tp += same_truth and same_pred
        fp += (not same_truth) and same_pred
        fn += same_truth and not same_pred
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r


def face_test(name, items, min_precision, min_recall=0.0):
    """items: [(filename, jpeg_bytes, true_label)]"""
    truth = {fn: lab for fn, _, lab in items}
    status, body, secs = post([(fn, b) for fn, b, _ in items])
    if status != 200:
        check(name, False, f"HTTP {status}")
        return
    pred = photo_people(body)
    missing = [fn for fn in truth if fn not in pred]
    p, r = pairwise(truth, pred)
    n_true = len(set(truth.values()))
    detail = (f"{len(items)} photos, {n_true} true people -> {len(body['people'])} found | "
              f"precision {p:.3f} recall {r:.3f} | skipped {len(body['skipped'])} | {secs:.0f}s")
    check(name, not missing and p >= min_precision and r >= min_recall, detail)


def main():
    # ---- 1. health + bad input
    r = requests.get(f"{API}/api/health", timeout=10)
    check("health endpoint", r.status_code == 200 and r.json().get("status") == "ok")

    r = requests.post(f"{API}/api/timeline", timeout=30)
    check("no files -> 400 with message", r.status_code == 400 and "error" in r.json())

    junk = os.urandom(4000)
    tiny = jpeg(Image.new("RGB", (10, 10), (120, 120, 120)))
    notimg = b"this is not an image"
    status, body, _ = post([("corrupt.jpg", junk), ("notes.txt", notimg)])
    check("corrupt/non-image upload -> 200, no events, no crash",
          status == 200 and body.get("events") == [], f"status {status}, skipped {len(body.get('skipped', []))}")
    status, body, _ = post([("tiny.jpg", tiny)])
    check("10x10 image handled without crash", status == 200, f"events {len(body.get('events', []))}")

    # ---- 2. EXIF + GPS photos (Wikimedia), no faces expected to matter
    wiki_all = sorted((RAW / "wikimedia_geotagged").glob("*.jpg"))
    wiki = []
    for p in wiki_all:
        m = extract_photo_metadata(p)
        if m["timestamp"] and m["lat"] is not None:
            wiki.append(p)
        if len(wiki) == 12:
            break
    status, body, secs = post([(p.name, p.read_bytes()) for p in wiki])
    dated = [e for e in body.get("events", []) if e["start_time"]]
    placed = [e for e in body.get("events", []) if e["place"] and "Unknown" not in e["place"]]
    check("EXIF+GPS photos -> dated, placed events",
          status == 200 and len(dated) == len(body["events"]) and len(placed) == len(body["events"]),
          f"{len(wiki)} photos -> {len(body.get('events', []))} events, {len(dated)} dated, {len(placed)} placed, {secs:.0f}s")

    # ---- 3. face fixtures (held-out data only)
    ce = {}
    import csv
    with open(RAW / "celeba_holdout" / "manifest.csv") as f:
        for row in csv.DictReader(f):
            ce.setdefault(row["celeb_id"], []).append(RAW / "celeba_holdout" / row["path"])
    ids = rng.sample(sorted(ce), 5)
    items = [(f"c{i}_{j}.jpg", jpeg(p), i) for i in ids for j, p in enumerate(ce[i][:6])]
    face_test("faces, clean (5 people x 6 photos, no EXIF)", items, min_precision=0.95, min_recall=0.5)

    _, held = split_agedb(scan_agedb())
    pool = [i for i, v in held.items() if len(v) >= 6 and max(a for _, a in v) - min(a for _, a in v) >= 30]
    four = rng.sample(sorted(pool), 4)

    def age_items(degrade_half):
        out = []
        for ident in four:
            photos = sorted(held[ident], key=lambda x: x[1])
            pick = [photos[0], photos[len(photos) // 3], photos[2 * len(photos) // 3], photos[-1]]
            for k, (path, age) in enumerate(pick):
                out.append((f"{ident[:7]}_{ident[-6:]}_age{age}.jpg",
                            jpeg(path, degrade=degrade_half and k % 2 == 1), ident))
        return out

    face_test("faces across decades (4 people x 4 ages, 25-70y span)", age_items(False), 0.9)
    face_test("faces across decades + half blurred/low-res/JPEG-crushed", age_items(True), 0.9)

    kids = sorted((RAW / "utkface" / "kid").glob("*.jpg"), key=lambda p: int(p.stem.split("_")[0]))[-8:]
    items = []
    for k, p in enumerate(kids):
        items.append((f"kid{k}_clean.jpg", jpeg(p), k))
        items.append((f"kid{k}_blurry.jpg", jpeg(p, degrade=True), k))
    face_test("8 different children, clean + degraded (known weak spot: regression floor only)", items, 0.2)

    # ---- 4. mixed: dated landscapes + undated faces together; ordering / undated flag
    mixed = [(p.name, p.read_bytes()) for p in wiki[:4]] + [(f"u{j}.jpg", jpeg(ce[ids[0]][j])) for j in range(3)]
    status, body, _ = post(mixed)
    und = [e for e in body.get("events", []) if e["start_time"] is None]
    check("mixed dated + undated -> undated flagged, none dropped",
          status == 200 and len(und) == 3 and not body["skipped"] and
          sum(e["photo_count"] for e in body["events"]) == len(mixed),
          f"{len(body.get('events', []))} events, {len(und)} undated, skipped {len(body.get('skipped', []))}")

    # ---- summary
    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
