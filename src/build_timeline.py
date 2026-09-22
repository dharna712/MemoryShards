"""
Day 10 — Fusion: run Extract -> Cluster -> Caption over a folder of
photos and merge the results into one ordered timeline. This is the
function the backend API (src/api.py) calls per request — everything
here is already-tested logic from Days 3, 4, and 9, just wired together.

Face recognition (Recognize stage) is deliberately NOT part of this
function yet — it clusters photos by *person*, which is a cross-event
signal ("who showed up more than once"), not a per-event one. It gets
folded in as an enrichment once the trained checkpoint exists (Day 8/11),
not blocking this stage.
"""

from pathlib import Path

from cluster_events import cluster_events
from caption_events import caption_event
from extract_metadata import extract_photo_metadata

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_photos(photos_dir):
    return sorted(
        p for p in Path(photos_dir).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_timeline(photos_dir, caption=True):
    """Returns (timeline, skipped) where:
    - timeline: list of events, sorted chronologically —
      [{'start_time': datetime, 'place': str, 'caption': str|None,
        'photo_count': int, 'photos': [str, ...]}, ...]
    - skipped: [{'filename': str, 'reason': str}, ...] — photos with no
      usable EXIF (no timestamp or no GPS) can't be placed on a timeline,
      same as a real system would face, but callers (the API, in
      particular) need to know WHICH photos and WHY rather than just a
      silent drop, so uploaders aren't left guessing why their count of
      events doesn't match their count of photos."""
    photo_paths = find_photos(photos_dir)

    records = []
    skipped = []
    for path in photo_paths:
        meta = extract_photo_metadata(path)
        has_time, has_gps = bool(meta["timestamp"]), meta["lat"] is not None
        if has_time and has_gps:
            records.append(meta)
        else:
            reason = (
                "no timestamp or GPS in EXIF" if not has_time and not has_gps
                else "no timestamp in EXIF" if not has_time
                else "no GPS in EXIF"
            )
            skipped.append({"filename": path.name, "reason": reason})

    if skipped:
        print(f"[fusion] skipped {len(skipped)}/{len(photo_paths)} photos (no usable EXIF timestamp/GPS)")

    records = cluster_events(records)

    # cluster == -1 means "not part of any multi-photo cluster" — every
    # standalone record is its OWN separate event, not one shared event.
    # Grouping them all under the literal key -1 was a real bug: it
    # merged every unrelated singleton photo into one fake mega-event.
    events_by_cluster = {}
    next_standalone_key = -2  # -1 is reserved by DBSCAN, count down from there
    for r in records:
        if r["cluster"] == -1:
            key = next_standalone_key
            next_standalone_key -= 1
        else:
            key = r["cluster"]
        events_by_cluster.setdefault(key, []).append(r)

    timeline = []
    for cluster_id, items in events_by_cluster.items():
        items = sorted(items, key=lambda r: r["timestamp"])
        event_caption = None
        if caption:
            event_caption = caption_event([r["path"] for r in items])
        timeline.append({
            "start_time": items[0]["timestamp"],
            "place": items[0]["place"],
            "caption": event_caption,
            "photo_count": len(items),
            "photos": [r["path"] for r in items],
            "standalone": len(items) == 1,
        })

    timeline.sort(key=lambda e: e["start_time"])
    return timeline, skipped


def print_timeline(timeline, skipped=None):
    for event in timeline:
        time_str = event["start_time"].strftime("%d %b, %I:%M %p")
        caption_str = f" — {event['caption']}" if event["caption"] else ""
        count_str = f" ({event['photo_count']} photos)" if event["photo_count"] > 1 else ""
        print(f"  {time_str}  {event['place']}{caption_str}{count_str}")
    for s in skipped or []:
        print(f"  [skipped] {s['filename']} — {s['reason']}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "wikimedia_geotagged"
    print(f"[fusion] building timeline from {data_dir}\n")
    timeline, skipped = build_timeline(data_dir, caption=False)  # caption=False for a fast structural test first
    print_timeline(timeline, skipped)
