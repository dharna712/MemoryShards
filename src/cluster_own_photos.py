"""
Day 8: the real test — point this at a folder of photos, and it groups
them by person, the same way a phone gallery's "People" view does.

Detects every face in every photo (a group photo counts as multiple
faces), embeds each one with the face-recognition model (fine-tuned
checkpoint if available, pretrained-only otherwise), then clusters the
embeddings with DBSCAN on cosine distance — no fixed number of people
needs to be specified up front.

Usage:
    python src/cluster_own_photos.py --photos_dir "C:/path/to/your/photos"
    python src/cluster_own_photos.py --photos_dir ./data/raw/celeba_subset/1 --checkpoint checkpoints/face_embedding_head.pt

Output:
    - a text report: which photos/faces landed in which cluster
    - (optional, --save_thumbnails) cropped face images sorted into
      out_dir/cluster_0/, cluster_1/, ... folders for visual inspection

Privacy reminder: if you run this on real personal/family photos, don't
publish the output (thumbnails or reports) anywhere public — same rule as
no team names in the repo.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.cluster import DBSCAN

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_photos(photos_dir):
    return sorted(
        p for p in Path(photos_dir).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_model(checkpoint_path, device):
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[model] using fine-tuned checkpoint: {checkpoint_path}")
    else:
        print("[model] no checkpoint found — using pretrained-only backbone "
              "(fine, but the fine-tuned model should cluster better)")
    model.eval()
    return model


def detect_and_embed_faces(photo_paths, mtcnn, model, device):
    """Returns a list of dicts: {photo, face_index, embedding, thumbnail}."""
    records = []
    for photo_path in photo_paths:
        try:
            img = Image.open(photo_path).convert("RGB")
        except Exception as e:
            print(f"  skipping {photo_path.name}: couldn't open ({e})")
            continue

        faces = mtcnn(img)  # keep_all=False by default -> at most one face
        if faces is None:
            print(f"  no face found in {photo_path.name}")
            continue

        with torch.no_grad():
            embedding = model(faces.unsqueeze(0).to(device)).cpu().numpy()[0]

        records.append({
            "photo": str(photo_path),
            "embedding": embedding,
        })

    return records


def cluster_embeddings(records, eps, min_samples):
    if not records:
        return records
    embeddings = np.stack([r["embedding"] for r in records])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    for record, label in zip(records, clustering.labels_):
        record["cluster"] = int(label)
    return records


def print_report(records):
    clusters = {}
    for r in records:
        clusters.setdefault(r["cluster"], []).append(Path(r["photo"]).name)

    print(f"\n{'='*60}")
    print(f"found {len(records)} faces across the photo set")
    print(f"{'='*60}\n")

    for cluster_id in sorted(clusters, key=lambda c: (c == -1, c)):
        photos = clusters[cluster_id]
        label = "UNCLUSTERED (no clear match to anyone else)" if cluster_id == -1 else f"Person {cluster_id}"
        print(f"{label} — {len(photos)} photo(s):")
        for name in photos:
            print(f"    {name}")
        print()


def save_thumbnails(records, mtcnn, out_dir):
    """Re-runs detection to save the actual cropped face (not just the tensor)."""
    out_dir = Path(out_dir)
    for i, r in enumerate(records):
        cluster_dir = out_dir / (f"cluster_{r['cluster']}" if r["cluster"] != -1 else "unclustered")
        cluster_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(r["photo"]).convert("RGB")
        thumb_path = cluster_dir / f"{i}_{Path(r['photo']).stem}.jpg"
        mtcnn(img, save_path=str(thumb_path))
    print(f"thumbnails saved under {out_dir}/ — remember: don't publish these if they're real people")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos_dir", required=True, help="folder of photos to cluster")
    parser.add_argument("--checkpoint", default="checkpoints/face_embedding_head.pt")
    parser.add_argument("--eps", type=float, default=0.4,
                         help="DBSCAN cosine-distance threshold — lower = stricter matching")
    parser.add_argument("--min_samples", type=int, default=1,
                         help="min photos to form a cluster (1 = singletons allowed as their own person)")
    parser.add_argument("--save_thumbnails", action="store_true")
    parser.add_argument("--out_dir", default="data/face_clusters")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=False)
    model = build_model(args.checkpoint, device)

    photo_paths = find_photos(args.photos_dir)
    print(f"[1/3] found {len(photo_paths)} photo(s) in {args.photos_dir}")
    if not photo_paths:
        return

    records = detect_and_embed_faces(photo_paths, mtcnn, model, device)
    print(f"[2/3] detected a face in {len(records)}/{len(photo_paths)} photos")

    records = cluster_embeddings(records, args.eps, args.min_samples)
    n_people = len({r["cluster"] for r in records if r["cluster"] != -1})
    print(f"[3/3] clustered into {n_people} distinct people "
          f"({sum(1 for r in records if r['cluster'] == -1)} unclustered)")

    print_report(records)

    if args.save_thumbnails:
        save_thumbnails(records, mtcnn, args.out_dir)


if __name__ == "__main__":
    main()
