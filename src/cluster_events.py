"""
Day 3-4 — the Cluster stage: group photos into discrete "events" by
time + location proximity (DBSCAN over a joint time/geo feature space).

Two things get tested here, deliberately:

1. The real Wikimedia set (src/extract_metadata.py output) — 29 photos
   from unrelated photographers, decades, and continents. Nothing in it
   should cluster together. This is a negative-case test: it proves the
   distance metric and eps don't over-merge unrelated photos into a fake
   shared event.

2. A small synthetic set (jittered timestamps/GPS around a few real
   photos, clearly fabricated for this test only) — this is a positive-
   case test proving photos that genuinely are close in time and space
   DO get merged into one event. The real Wikimedia set can't test this
   half on its own, since none of its photos were actually taken near
   each other.

Both are needed — a clustering function that never merges anything would
pass test 1 by doing nothing.
"""

import random
from datetime import timedelta
from math import asin, cos, radians, sin, sqrt

import numpy as np
from sklearn.cluster import DBSCAN

from extract_metadata import extract_photo_metadata
import csv
from pathlib import Path


def haversine_km(coord1, coord2):
    lat1, lon1, lat2, lon2 = map(radians, [*coord1, *coord2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


def build_distance_matrix(records, geo_scale_km, time_scale_hours):
    """Combines geo and time distance into one unitless scale, so a
    single DBSCAN eps=1.0 means 'within geo_scale_km AND time_scale_hours
    of each other' (roughly — it's a joint radius, not a strict AND)."""
    n = len(records)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            geo_dist = haversine_km(
                (records[i]["lat"], records[i]["lon"]),
                (records[j]["lat"], records[j]["lon"]),
            )
            time_dist = abs((records[i]["timestamp"] - records[j]["timestamp"]).total_seconds()) / 3600
            combined = sqrt((geo_dist / geo_scale_km) ** 2 + (time_dist / time_scale_hours) ** 2)
            matrix[i, j] = matrix[j, i] = combined
    return matrix


def cluster_events(records, geo_scale_km=1.0, time_scale_hours=3.0, min_samples=2):
    """records: list of {'timestamp': datetime, 'lat': float, 'lon': float, ...}.
    Adds a 'cluster' key (-1 = not part of any multi-photo event)."""
    if len(records) < 2:
        for r in records:
            r["cluster"] = -1
        return records

    distance_matrix = build_distance_matrix(records, geo_scale_km, time_scale_hours)
    labels = DBSCAN(eps=1.0, min_samples=min_samples, metric="precomputed").fit_predict(distance_matrix)
    for record, label in zip(records, labels):
        record["cluster"] = int(label)
    return records


def print_events(records):
    events = {}
    for r in records:
        events.setdefault(r["cluster"], []).append(r)

    n_events = sum(1 for c in events if c != -1)
    n_singletons = len(events.get(-1, []))
    print(f"{n_events} multi-photo event(s), {n_singletons} standalone photo(s)\n")

    for cluster_id in sorted(events, key=lambda c: (c == -1, c)):
        items = events[cluster_id]
        label = "STANDALONE" if cluster_id == -1 else f"Event {cluster_id}"
        for r in sorted(items, key=lambda r: r["timestamp"]):
            print(f"  [{label}] {r['timestamp']}  {r.get('place', '?')}  {r.get('title', r.get('path'))}")


def real_data_test():
    print("=" * 70)
    print("TEST 1 — real Wikimedia photos (expect: no false merging)")
    print("=" * 70)
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "wikimedia_geotagged"
    with open(data_dir / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        meta = extract_photo_metadata(data_dir / row["path"])
        if meta["timestamp"] and meta["lat"] is not None:
            meta["title"] = row["title"]
            records.append(meta)

    records = cluster_events(records)
    print_events(records)
    n_events = len({r["cluster"] for r in records if r["cluster"] != -1})
    print(f"\nresult: {n_events} event(s) found among {len(records)} photos from unrelated searches.")
    print("Inspect these by hand before trusting them: this set is photos from different")
    print("photographers/topics, so any cluster should only form because two photos")
    print("genuinely share a photographer, place, and moment (e.g. a numbered photo series")
    print("from one outing) — not because the distance metric is too loose.")


def synthetic_positive_test():
    print("\n" + "=" * 70)
    print("TEST 2 — synthetic same-trip photos (expect: real merging)")
    print("=" * 70)
    random.seed(0)
    base_time = None
    base_lat, base_lon = 18.9220, 72.8347  # Gateway of India, for a familiar reference point

    # Simulates one real "event": 4 photos taken within ~20 min, ~50m of each other
    from datetime import datetime
    base_time = datetime(2026, 6, 10, 11, 15, 0)
    event_a = [
        {
            "timestamp": base_time + timedelta(minutes=random.uniform(0, 20)),
            "lat": base_lat + random.uniform(-0.0004, 0.0004),
            "lon": base_lon + random.uniform(-0.0004, 0.0004),
            "place": "Gateway of India area",
            "title": f"synthetic_event_a_{i}.jpg",
        }
        for i in range(4)
    ]

    # A second "event" 5 hours later and 3km away — same day, different event
    event_b_time = base_time + timedelta(hours=5)
    event_b = [
        {
            "timestamp": event_b_time + timedelta(minutes=random.uniform(0, 15)),
            "lat": base_lat + 0.03 + random.uniform(-0.0004, 0.0004),
            "lon": base_lon + 0.02 + random.uniform(-0.0004, 0.0004),
            "place": "Marine Drive area",
            "title": f"synthetic_event_b_{i}.jpg",
        }
        for i in range(3)
    ]

    records = cluster_events(event_a + event_b)
    print_events(records)

    n_events = len({r["cluster"] for r in records if r["cluster"] != -1})
    print(f"\nresult: {n_events} events found from 2 synthetic real events "
          f"({'PASS' if n_events == 2 else 'CHECK — expected exactly 2 events'})")


if __name__ == "__main__":
    real_data_test()
    synthetic_positive_test()
