"""
Labelled evaluation of the event-clustering stage (EXIF time + GPS -> events).

There is no large labelled real-world event dataset, so we generate seeded SYNTHETIC trips with known ground-truth
events and score cluster_events() the way build_timeline uses it (a photo DBSCAN leaves unclustered becomes its own
event). Four scenarios, one of them deliberately hard:
  typical_day       4-6 events, 5-10 h and 5-30 km apart, plus 1-3 stray single photos
  same_place_days   events at the SAME coordinates but 12-48 h apart (must be split by time)
  same_time_places  events starting within 30 min of each other but 5-30 km apart (must be split by place)
  back_to_back_hard events only 1.5-3 h and 0.5-2 km apart (genuinely different, but close: expected to merge)
Metric: pairwise precision / recall / F1 over photo pairs ("same event?"), exact event-count rate.
Also a grid over geo_scale_km x time_scale_hours to check the hand-tuned defaults (1 km, 3 h).

Usage: python src/evaluate_events.py   -> prints tables, writes report/event_eval.json
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_events import cluster_events

OUT = Path(__file__).resolve().parent.parent / "report" / "event_eval.json"
KM_PER_DEG = 111.0


def photos(rng, truth_id, t0, lat, lon, k, minutes):
    return [dict(timestamp=t0 + timedelta(minutes=rng.uniform(0, minutes)),
                 lat=lat + rng.uniform(-0.0006, 0.0006), lon=lon + rng.uniform(-0.0006, 0.0006), truth=truth_id) for _ in range(k)]


def offset(rng, lat, lon, km_lo, km_hi):
    d = rng.uniform(km_lo, km_hi) / KM_PER_DEG; ang = rng.uniform(0, 2 * np.pi)
    return lat + d * np.sin(ang), lon + d * np.cos(ang)


def make(scn, rng):
    lat, lon = rng.uniform(12, 28), rng.uniform(72, 88)
    t = datetime(2026, 6, 1, 8, 0); recs = []; tid = 0
    if scn == "typical_day":
        for _ in range(rng.randint(4, 6)):
            recs += photos(rng, tid, t, lat, lon, rng.randint(3, 8), rng.uniform(5, 30)); tid += 1
            t += timedelta(hours=rng.uniform(5, 10)); lat, lon = offset(rng, lat, lon, 5, 30)
        for _ in range(rng.randint(1, 3)):  # strays: single photos far from everything
            la, lo = offset(rng, lat, lon, 60, 200); recs += photos(rng, tid, t, la, lo, 1, 1); tid += 1; t += timedelta(hours=rng.uniform(6, 20))
    elif scn == "same_place_days":
        for _ in range(rng.randint(3, 6)):
            recs += photos(rng, tid, t, lat, lon, rng.randint(3, 8), rng.uniform(5, 30)); tid += 1
            t += timedelta(hours=rng.uniform(12, 48))
    elif scn == "same_time_places":
        for _ in range(rng.randint(2, 3)):
            la, lo = offset(rng, lat, lon, 5, 30)
            recs += photos(rng, tid, t + timedelta(minutes=rng.uniform(0, 30)), la, lo, rng.randint(3, 8), rng.uniform(5, 30)); tid += 1
    elif scn == "long_event":  # people walking around: photos spread over 1.5-3 h and a few hundred metres
        for _ in range(rng.randint(2, 3)):
            k = rng.randint(6, 12); dur = rng.uniform(90, 180)
            for _ in range(k):
                recs.append(dict(timestamp=t + timedelta(minutes=rng.uniform(0, dur)),
                                 lat=lat + rng.gauss(0, 0.003), lon=lon + rng.gauss(0, 0.003), truth=tid))
            tid += 1; t += timedelta(hours=rng.uniform(8, 20)); lat, lon = offset(rng, lat, lon, 10, 40)
    elif scn == "back_to_back_hard":
        for _ in range(rng.randint(3, 5)):
            recs += photos(rng, tid, t, lat, lon, rng.randint(3, 8), rng.uniform(5, 20)); tid += 1
            t += timedelta(hours=rng.uniform(1.5, 3)); lat, lon = offset(rng, lat, lon, 0.5, 2)
    return recs, tid


def score(recs, truth_events, geo, hrs):
    recs = [dict(r) for r in recs]
    out = cluster_events(recs, geo_scale_km=geo, time_scale_hours=hrs)
    nxt = -2; lab = []
    for r in out:
        if r["cluster"] == -1:
            lab.append(nxt); nxt -= 1   # a standalone photo is its own event (as in build_timeline)
        else:
            lab.append(r["cluster"])
    tr = np.array([r["truth"] for r in out]); pr = np.array(lab)
    st = tr[:, None] == tr[None, :]; sp = pr[:, None] == pr[None, :]; iu = np.triu_indices(len(tr), 1)
    t, p = st[iu], sp[iu]; tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 1.0; rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, int(len(set(lab)) == truth_events)


SCN = ["typical_day", "same_place_days", "same_time_places", "back_to_back_hard", "long_event"]
N = 200
res = {"default": {}, "grid": {}}
data = {}
for s in SCN:
    rng = random.Random(hash(s) % 1000 + 7 if False else {"typical_day": 1, "same_place_days": 2, "same_time_places": 3, "back_to_back_hard": 4, "long_event": 5}[s])
    data[s] = [make(s, rng) for _ in range(N)]

print(f"Event clustering, defaults geo_scale_km=1.0 time_scale_hours=3.0, {N} seeded synthetic trips per scenario")
print(f"{'scenario':20s} precision  recall    F1   exact-event-count")
allf = []
for s in SCN:
    m = np.array([score(r, t, 1.0, 3.0) for r, t in data[s]])
    res["default"][s] = dict(precision=float(m[:, 0].mean()), recall=float(m[:, 1].mean()), f1=float(m[:, 2].mean()), exact=float(m[:, 3].mean()))
    allf.append(m[:, 2].mean())
    print(f"{s:20s} {m[:, 0].mean():.3f}     {m[:, 1].mean():.3f}   {m[:, 2].mean():.3f}   {m[:, 3].mean() * 100:.0f}%")
res["default"]["mean_f1_all"] = float(np.mean(allf)); res["default"]["mean_f1_first3"] = float(np.mean(allf[:3]))
print(f"mean F1 over all 5: {np.mean(allf):.3f}   over the 3 well-separated scenarios: {np.mean(allf[:3]):.3f}")

GEO = [0.25, 0.5, 1.0, 2.0, 5.0]; HRS = [1.0, 3.0, 6.0, 12.0]
print("\nParameter grid: mean F1 over the 3 realistic scenarios (rows geo km, columns time hours)")
print("geo\\time " + "  ".join(f"{h:5.0f}h" for h in HRS))
for g in GEO:
    row = []
    for h in HRS:
        f = np.mean([np.mean([score(r, t, g, h)[2] for r, t in data[s][:80]]) for s in SCN[:3]])
        res["grid"][f"{g}|{h}"] = float(f); row.append(f)
    print(f"{g:5.2f}km  " + "  ".join(f"{x:6.3f}" for x in row))
print("\nParameter grid on the two HARD scenarios (back_to_back_hard + long_event): mean F1")
print("geo\\time " + "  ".join(f"{h:5.0f}h" for h in HRS))
res["grid_hard"] = {}
best = (0, None)
for g in GEO:
    row = []
    for h in HRS:
        f = np.mean([np.mean([score(r, t, g, h)[2] for r, t in data[s][:80]]) for s in ("back_to_back_hard", "long_event")])
        res["grid_hard"][f"{g}|{h}"] = float(f); row.append(f)
        if f > best[0]: best = (f, (g, h))
    print(f"{g:5.2f}km  " + "  ".join(f"{x:6.3f}" for x in row))
res["best_hard"] = dict(f1=float(best[0]), geo=best[1][0], hours=best[1][1])
print(f"best on hard pair: geo {best[1][0]} km, time {best[1][1]} h -> F1 {best[0]:.3f}; defaults (1 km, 3 h) -> {res['grid_hard']['1.0|3.0']:.3f}")
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
