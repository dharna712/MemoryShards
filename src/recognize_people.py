"""
Recognize stage: detect every face in the photos, embed each one with the
fine-tuned FaceNet checkpoint, and group faces into people.

Grouping uses average-linkage agglomerative clustering on cosine distance
(threshold 0.5). Chosen from src/evaluate_clustering.py on 230 faces of 29
identities the model never saw in training: 98.8% pairwise precision,
80.4% recall, ARI 0.883. DBSCAN chains different people together once the
threshold passes ~0.35 (precision falls to 0.28 at 0.45), so it was dropped.

Input crops are the raw 0-255 MTCNN output, no extra standardization: that
is exactly what training and evaluation fed the model.

Nothing here writes to disk. If the checkpoint is missing, the stage is
skipped (the pretrained-only backbone can't separate people, so its
output would be noise).
"""

import base64
import io
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "face_embedding_head.pt"
DISTANCE_THRESHOLD = 0.5
MIN_DETECTION_PROB = 0.97
MIN_FACE_PX = 48
AVATAR_PX = 96

_models = None


def _load_models():
    global _models
    if _models is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=False)
        model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _models = (mtcnn, model, device)
    return _models


def _avatar_base64(face_tensor):
    arr = face_tensor.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr).resize((AVATAR_PX, AVATAR_PX), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _detect_faces(photo_paths, mtcnn, model, device):
    faces = []
    for path in photo_paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        boxes, probs = mtcnn.detect(img)
        if boxes is None:
            continue
        keep = [
            i for i, (b, p) in enumerate(zip(boxes, probs))
            if p is not None and p >= MIN_DETECTION_PROB
            and min(b[2] - b[0], b[3] - b[1]) >= MIN_FACE_PX
        ]
        if not keep:
            continue
        crops = mtcnn.extract(img, boxes[keep], None)
        if crops.dim() == 3:
            crops = crops.unsqueeze(0)
        with torch.no_grad():
            embeddings = model(crops.to(device)).cpu().numpy()
        for j, i in enumerate(keep):
            size = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            faces.append({
                "photo": str(path),
                "embedding": embeddings[j],
                "crop": crops[j],
                "size": float(size),
            })
    return faces


def _cluster(faces):
    if len(faces) == 1:
        return [0]
    embeddings = np.stack([f["embedding"] for f in faces])
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=DISTANCE_THRESHOLD,
        metric="cosine", linkage="average",
    ).fit(embeddings).labels_.tolist()


def recognize_people(photo_paths):
    """Returns (people, photo_to_people):
    - people: [{'id': 1, 'label': 'Person 1', 'face_count', 'photo_count',
                'avatar': data-URI}, ...], most-photographed first
    - photo_to_people: {photo_path: [person_id, ...]}
    Returns ([], {}) when the checkpoint is missing or no faces are found."""
    if not CHECKPOINT.exists():
        return [], {}
    mtcnn, model, device = _load_models()
    faces = _detect_faces(photo_paths, mtcnn, model, device)
    if not faces:
        return [], {}

    labels = _cluster(faces)
    by_cluster = {}
    for face, label in zip(faces, labels):
        by_cluster.setdefault(label, []).append(face)

    ordered = sorted(
        by_cluster.values(),
        key=lambda fs: (-len({f["photo"] for f in fs}), -len(fs)),
    )
    people, photo_to_people = [], {}
    for person_id, members in enumerate(ordered, start=1):
        photos = {f["photo"] for f in members}
        best = max(members, key=lambda f: f["size"])
        people.append({
            "id": person_id,
            "label": f"Person {person_id}",
            "face_count": len(members),
            "photo_count": len(photos),
            "avatar": _avatar_base64(best["crop"]),
        })
        for photo in photos:
            photo_to_people.setdefault(photo, []).append(person_id)
    return people, photo_to_people
