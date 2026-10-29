"""Per-stage latency of the pipeline on THIS machine (CPU-only PyTorch), per photo. Writes report/latency_profile.json."""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from extract_metadata import extract_photo_metadata
from places import offline_place
from cluster_events import cluster_events
import caption_events
import recognize_people as rp

ROOT = Path(__file__).resolve().parent.parent
photos = []
for p in sorted((ROOT / "data" / "raw" / "wikimedia_geotagged").glob("*.jpg")):
    m = extract_photo_metadata(p)
    if m["timestamp"] and m["lat"] is not None:
        photos.append((p, m))
    if len(photos) == 12:
        break
N = len(photos)

# warm-up (model loading is a one-off start-up cost, reported separately)
t = time.perf_counter(); caption_events.caption_image(photos[0][0]); cap_warm = time.perf_counter() - t
t = time.perf_counter(); rp.recognize_people([str(photos[0][0])]); face_warm = time.perf_counter() - t

t = time.perf_counter()
metas = [extract_photo_metadata(p) for p, _ in photos]
t_exif = time.perf_counter() - t
t = time.perf_counter()
for m in metas:
    offline_place(m["lat"], m["lon"])
t_geo = time.perf_counter() - t
t = time.perf_counter()
cluster_events([dict(r) for r in metas])
t_cluster = time.perf_counter() - t
t = time.perf_counter()
for p, _ in photos:
    caption_events.caption_image(p)
t_cap = time.perf_counter() - t
t = time.perf_counter()
rp.recognize_people([str(p) for p, _ in photos])
t_face = time.perf_counter() - t

res = dict(photos=N, device=str(torch.device("cpu")), per_photo_seconds=dict(
    exif=t_exif / N, geocode=t_geo / N, event_clustering=t_cluster / N, caption=t_cap / N, face_recognition=t_face / N),
    startup_seconds=dict(caption_model_first_call=cap_warm, face_models_first_call=face_warm))
res["per_photo_seconds"]["total"] = sum(res["per_photo_seconds"].values())
json.dump(res, open(ROOT / "report" / "latency_profile.json", "w"), indent=1)
print(json.dumps(res, indent=1))
