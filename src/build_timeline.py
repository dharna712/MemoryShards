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
from places import place_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_photos(photos_dir):
    return sorted(
        p for p in Path(photos_dir).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_timeline(photos_dir, caption=True, online_places=False):
    """Returns (timeline, skipped) where:
    - timeline: list of events, sorted chronologically, with any undated
      events (see below) appended at the end —
      [{'start_time': datetime|None, 'place': str, 'caption': str|None,
        'photo_count': int, 'photos': [str, ...]}, ...]
    - skipped: [{'filename': str, 'reason': str}, ...] — photos that
      couldn't even be read (corrupt file, unsupported format). Photos
      that ARE readable but have no timestamp/GPS in EXIF (e.g. shared
      over WhatsApp, which strips it) are NOT skipped anymore — they
      still can't be placed chronologically or geographically, but they
      still have faces worth recognizing, so each becomes its own
      standalone "undated" event instead of being dropped silently."""
    photo_paths = find_photos(photos_dir)

    records = []
    undated = []
    skipped = []
    for path in photo_paths:
        try:
            meta = extract_photo_metadata(path)
        except Exception:
            skipped.append({"filename": path.name, "reason": "couldn't read photo (corrupt or unsupported file)"})
            continue
        has_time, has_gps = bool(meta["timestamp"]), meta["lat"] is not None
        if has_time and has_gps:
            records.append(meta)
        else:
            undated.append(meta)

    if undated:
        print(f"[fusion] {len(undated)}/{len(photo_paths)} photos had no usable EXIF timestamp/GPS — kept as undated events")
    if skipped:
        print(f"[fusion] skipped {len(skipped)}/{len(photo_paths)} unreadable photos")

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
        lat = sum(r["lat"] for r in items) / len(items)
        lon = sum(r["lon"] for r in items) / len(items)
        event_caption = None
        if caption:
            event_caption = caption_event([r["path"] for r in items])
        timeline.append({
            "start_time": items[0]["timestamp"],
            "place": place_label(lat, lon, online_places),
            "lat": lat,
            "lon": lon,
            "caption": event_caption,
            "photo_count": len(items),
            "photos": [r["path"] for r in items],
            "standalone": len(items) == 1,
        })

    timeline.sort(key=lambda e: e["start_time"])

    for meta in undated:
        event_caption = caption_event([meta["path"]]) if caption else None
        timeline.append({
            "start_time": None,
            "place": "Unknown location",
            "lat": None,
            "lon": None,
            "caption": event_caption,
            "photo_count": 1,
            "photos": [meta["path"]],
            "standalone": True,
        })

    return timeline, skipped


def print_timeline(timeline, skipped=None):
    for event in timeline:
        time_str = event["start_time"].strftime("%d %b, %I:%M %p") if event["start_time"] else "Undated"
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
