"""
Benchmark the "group photos by person" step on identities the model never
saw in training (the CelebA held-out slice). Embeds every face once, then
sweeps the DBSCAN threshold and reports clustering quality, so the eps used
in the pipeline is chosen from data instead of by eye.

Metrics (pairwise, over all photo pairs):
  precision = of pairs the clusterer put together, how many are truly the
              same person
  recall    = of truly-same-person pairs, how many it put together
  ARI       = adjusted Rand index (1.0 = perfect, ~0 = random)

Usage:
    python src/evaluate_clustering.py
    python src/evaluate_clustering.py --checkpoint checkpoints/face_embedding_head.pt --min_photos 3
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score

from cluster_own_photos import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
HELD_OUT_DIR = REPO_ROOT / "data" / "raw" / "celeba_holdout"


def embed_holdout(checkpoint, min_photos, device):
    mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=False)
    model = build_model(checkpoint, device)
    embeddings, labels = [], []
    for identity_dir in sorted(HELD_OUT_DIR.iterdir()):
        photos = sorted(identity_dir.glob("*.jpg"))
        if len(photos) < min_photos:
            continue
        for p in photos:
            face = mtcnn(Image.open(p).convert("RGB"))
            if face is None:
                continue
            with torch.no_grad():
                emb = model(face.unsqueeze(0).to(device)).cpu().numpy()[0]
            embeddings.append(emb)
            labels.append(identity_dir.name)
    return np.stack(embeddings), np.array(labels)


def pairwise_pr(true, pred):
    same_true = true[:, None] == true[None, :]
    same_pred = (pred[:, None] == pred[None, :]) & (pred[:, None] != -1)
    iu = np.triu_indices(len(true), k=1)
    st, sp = same_true[iu], same_pred[iu]
    tp = (st & sp).sum()
    precision = tp / sp.sum() if sp.sum() else 0.0
    recall = tp / st.sum() if st.sum() else 0.0
    return precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "face_embedding_head.pt"))
    parser.add_argument("--min_photos", type=int, default=3)
    args = parser.parse_args()
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb, labels = embed_holdout(args.checkpoint, args.min_photos, device)
    print(f"{len(labels)} faces, {len(set(labels))} identities (fine-tuned model)")

    def report(name, make):
        print(f"\n--- {name} ---")
        print(f"{'thr':>5} {'clusters':>9} {'precision':>10} {'recall':>8} {'ARI':>6}")
        for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
            pred = make(thr).fit(emb).labels_
            p, r = pairwise_pr(labels, pred)
            print(f"{thr:>5} {len(set(pred)):>9} {p:>10.3f} {r:>8.3f} {adjusted_rand_score(labels, pred):>6.3f}")

    report("DBSCAN (min_samples=1, cosine)",
           lambda t: DBSCAN(eps=t, min_samples=1, metric="cosine"))
    for linkage in ["average", "complete"]:
        report(f"Agglomerative ({linkage} linkage, cosine)",
               lambda t, l=linkage: AgglomerativeClustering(
                   n_clusters=None, distance_threshold=t, metric="cosine", linkage=l))


if __name__ == "__main__":
    main()
