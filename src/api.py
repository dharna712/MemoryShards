"""
Backend for live UI integration: a small Flask API that wraps the
already-tested pipeline (Extract -> Cluster -> Caption, from
build_timeline.py) behind an HTTP endpoint the static site can call.

This is new surface area the static site didn't have before — it's the
piece that turns the site from "illustrative mockup" into "actually runs
the pipeline on whatever photos you upload."

Face recognition (Recognize) is not wired in here yet — see the note in
build_timeline.py. Can be added once the Colab-trained checkpoint exists,
without changing this endpoint's shape (just adds a 'people' field per
event).

Run locally:
    python src/api.py
Then POST photos (multipart, field name "photos") to
    http://localhost:5000/api/timeline
"""

import base64
import io
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

from build_timeline import build_timeline
from recognize_people import recognize_people

app = Flask(__name__)
# The page may be served from another origin (or a public https site calling this
# localhost API), so allow cross-origin and Chrome's private-network preflight.
CORS(app, allow_private_network=True)

THUMBNAIL_SIZE = (240, 240)
MAX_THUMBNAILS_PER_EVENT = 8  # caps payload size for events with many photos


def thumbnail_base64(photo_path):
    img = Image.open(photo_path).convert("RGB")
    img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def event_to_json(event, photo_to_people):
    thumbnails = [thumbnail_base64(p) for p in event["photos"][:MAX_THUMBNAILS_PER_EVENT]]
    return {
        "start_time": event["start_time"].isoformat(),
        # %-d avoids a leading zero on the day; year included — the
        # original format silently dropped it, which is genuinely
        # confusing once photos span more than one year (this dataset
        # spans 2009-2019)
        "display_time": event["start_time"].strftime("%d %b %Y, %I:%M %p"),
        "place": event["place"],
        "lat": event["lat"],
        "lon": event["lon"],
        "caption": event["caption"],
        "photo_count": event["photo_count"],
        "standalone": event["standalone"],
        "people": sorted({pid for p in event["photos"] for pid in photo_to_people.get(str(p), [])}),
        "filenames": [Path(p).name for p in event["photos"][:MAX_THUMBNAILS_PER_EVENT]],
        "thumbnail": thumbnails[0],
        "thumbnails": thumbnails,
        "more_photos_not_shown": max(0, event["photo_count"] - len(thumbnails)),
    }


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/timeline")
def timeline():
    files = request.files.getlist("photos")
    if not files:
        return jsonify({"error": "no photos uploaded (expected multipart field 'photos')"}), 400

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for f in files:
            if f.filename:
                f.save(tmp_path / f.filename)

        events, skipped = build_timeline(tmp_path, caption=True)

        if not events:
            return jsonify({
                "events": [],
                "skipped": skipped,
                "people": [],
                "message": "No usable photos — none had both a timestamp and GPS "
                           "location in their EXIF data.",
            })

        # Recognition is an enrichment: if it fails (missing checkpoint,
        # bad image) the timeline is still returned without people.
        try:
            people, photo_to_people = recognize_people([p for e in events for p in e["photos"]])
        except Exception as exc:
            print(f"[recognize] skipped: {exc}")
            people, photo_to_people = [], {}

        return jsonify({
            "events": [event_to_json(e, photo_to_people) for e in events],
            "skipped": skipped,
            "people": people,
        })


if __name__ == "__main__":
    # debug=True's file-watching reloader was falsely detecting changes in
    # torch/stdlib files mid-request on this machine and restarting the
    # server, killing in-flight requests — not needed for local testing,
    # and production (gunicorn) doesn't use this reloader at all
    app.run(debug=False, port=5000)
