"""
People-grouping sweep: how do v1 and v2 behave as the clustering distance
threshold moves, on three realistic situations the plain CelebA benchmark
doesn't cover? Same pairwise precision / recall / ARI as evaluate_clustering.

  cross_age   held-out AgeDB identities, 5 photos each spread across their
              age range
  cross_age_d same, with every other photo blurred/low-res/dark/compressed
  kids        200 held-out UTKFace children; each appears as a clean view and
              a degraded view (truth: the two views are one person, every
              other child is a different person)

Precision matters most (merging two people is worse than splitting one).

Usage: python src/evaluate_group_sweep.py checkpoints/face_embedding_head.pt checkpoints/face_embedding_v2.pt
"""

import os
import random
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

import csv

from face_data import (RAW, REPO_ROOT, cached_crop, degrade, plain, scan_agedb, scan_ylfw, split_agedb,
                       split_ylfw)

THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]


def pairwise_pr(truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    same_t = truth[:, None] == truth[None, :]
    same_p = pred[:, None] == pred[None, :]
    iu = np.triu_indices(len(truth), 1)
    t, p = same_t[iu], same_p[iu]
    tp = (t & p).sum()
    prec = tp / p.sum() if p.sum() else 1.0
    rec = tp / t.sum() if t.sum() else 1.0
    return prec, rec


def build(mtcnn):
    rng = random.Random(4)
    _, held = split_agedb(scan_agedb())
    ids = [i for i, v in held.items() if len(v) >= 5]
    crops, labels, deg_flags = [], [], []
    for n, ident in enumerate(sorted(ids)):
        photos = sorted(held[ident], key=lambda x: x[1])
        idx = np.linspace(0, len(photos) - 1, 5).round().astype(int)
        for k, i in enumerate(idx):
            crops.append(cached_crop(mtcnn, photos[i][0]))
            labels.append(n)
            deg_flags.append(k % 2 == 1)
    kid_files = sorted((RAW / "utkface" / "kid").glob("*.jpg"), key=lambda p: int(p.stem.split("_")[0]))[-200:]
    kids = [cached_crop(mtcnn, p) for p in kid_files]
    ce_c, ce_l = [], []
    by = {}
    with open(RAW / "celeba_holdout" / "manifest.csv") as f:
        for row in csv.DictReader(f):
            by.setdefault(row["celeb_id"], []).append(RAW / "celeba_holdout" / row["path"])
    for n, (ident, ps) in enumerate(sorted(by.items())):
        for p in ps[:8]:
            ce_c.append(cached_crop(mtcnn, p))
            ce_l.append(n)
    yl_c, yl_l, yl_d = [], [], []
    if (RAW / "ylfw").exists():
        _, held_y = split_ylfw(scan_ylfw())
        for n, (ident, files) in enumerate(sorted(held_y.items())):
            for k, p in enumerate(files[:6]):
                yl_c.append(cached_crop(mtcnn, p))
                yl_l.append(n)
                yl_d.append(k % 2 == 1)
    return crops, labels, deg_flags, kids, (ce_c, ce_l), (yl_c, yl_l, yl_d)


@torch.no_grad()
def embed(model, tensors, device):
    out = [model(torch.stack(tensors[i:i + 64]).to(device)).cpu() for i in range(0, len(tensors), 64)]
    return torch.cat(out).numpy()


SUMMARY = {}


def sweep(name, emb, truth):
    print(f"  {name}")
    for t in THRESHOLDS:
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric="cosine",
                                      linkage="average").fit(emb).labels_
        p, r = pairwise_pr(truth, lab)
        ari = adjusted_rand_score(truth, lab)
        SUMMARY.setdefault(t, []).append((p, r, ari))
        print(f"    thr {t:.2f}  clusters {len(set(lab)):4d}  precision {p:.3f}  recall {r:.3f}  ARI {ari:.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=False)
    crops, labels, deg_flags, kids, (ce_c, ce_l), (yl_c, yl_l, yl_d) = build(mtcnn)
    del mtcnn
    drng = random.Random(8)
    clean_t = [plain(c) for c in crops]
    mixed_t = [degrade(c, drng) if d else plain(c) for c, d in zip(crops, deg_flags)]
    kid_t = [plain(c) for c in kids] + [degrade(c, drng) for c in kids]
    kid_labels = list(range(len(kids))) * 2

    for path in sys.argv[1:]:
        m = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
        m.load_state_dict(torch.load(REPO_ROOT / path, map_location=device)["model_state_dict"])
        m.eval()
        print(f"=== {path}  ({len(set(labels))} AgeDB people, {len(kids)} children)")
        sweep("cross_age", embed(m, clean_t, device), labels)
        sweep("cross_age_degraded", embed(m, mixed_t, device), labels)
        sweep("kids (clean+degraded view each)", embed(m, kid_t, device), kid_labels)
        sweep("celeba_holdout (adult celebrities)", embed(m, [plain(c) for c in ce_c], device), ce_l)
        if yl_c:
            sweep("ylfw_children (held-out child identities)", embed(m, [plain(c) for c in yl_c], device), yl_l)
            sweep("ylfw_children_degraded", embed(m, [degrade(c, drng) if d else plain(c) for c, d in zip(yl_c, yl_d)], device), yl_l)
        print("  --- mean over the identity-labelled sets (cross_age, cross_age_degraded, celeba, ylfw, ylfw_degraded) ---")
        for t, rows in SUMMARY.items():
            rows = [r for i, r in enumerate(rows) if i in (0, 1, 3, 4, 5)] if len(rows) >= 6 else rows
            print(f"    thr {t:.2f}  mean precision {np.mean([r[0] for r in rows]):.3f}  "
                  f"recall {np.mean([r[1] for r in rows]):.3f}  ARI {np.mean([r[2] for r in rows]):.3f}")
        SUMMARY.clear()


if __name__ == "__main__":
    main()
