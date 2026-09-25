# MemoryShards

AI memory reconstruction from fragmented digital data — takes scattered photos,
messages, notes, timestamps, and locations, and reconstructs them into a
chronological timeline of events.

See [`context/PROJECT.md`](context/PROJECT.md) for the full problem statement
and approach, and [`context/STATUS.md`](context/STATUS.md) for current progress.

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install --no-deps facenet-pytorch==2.6.0
```

Put the trained face model at `checkpoints/face_embedding_head.pt` (produced by
`notebooks/train_face_recognizer.ipynb`; gitignored). Without it the Recognize
stage is skipped and the rest of the pipeline still runs.

## Run

```bash
cd src && python api.py          # backend on http://localhost:5000
cd web && python -m http.server 8080   # then open http://localhost:8080
```

`web/try.html` talks to `http://localhost:5000` by default; `?api=<url>` points
it somewhere else and is remembered.

