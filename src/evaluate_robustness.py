"""
Held-out robustness benchmark for the face-embedding checkpoints: how well do
they still verify "same person?" across decades of age, on degraded (blurry,
low-res, dark, compressed) photos, and on children?

Every model is scored on the SAME fixed pair lists (seeded), on identities
and images never used in training:

  celeba_clean       CelebA holdout (the v1 benchmark)
  agedb_clean        AgeDB holdout identities, random pairs
  agedb_gap20        same-person pairs >= 20 years apart vs different people
  agedb_degraded     as agedb_clean, one image of each pair degraded
  celeba_degraded    as celeba_clean, one image of each pair degraded
  kids_retrieval     200 held-out UTKFace children (no identity labels): query
                     a degraded view, retrieve its clean twin among all 200
                     (top-1). Measures telling different children apart.
  old_retrieval      same for 100 held-out faces aged 70+

Metric for verification sets: best-threshold accuracy (as in the v1 eval).

Usage:
    python src/evaluate_robustness.py checkpoints/face_embedding_head.pt checkpoints/face_embedding_v2.pt
"""

import csv
import os
import random
import sys
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

from face_data import (RAW, REPO_ROOT, cached_crop, degrade, plain, scan_agedb,
                       split_agedb)

N_PAIRS = 400
HOLDOUT_PER_ID = 16


def best_acc(same, diff):
    best = 0.0
    for t in sorted(same + diff):
        c = sum(s >= t for s in same) + sum(d < t for d in diff)
        best = max(best, c / (len(same) + len(diff)))
    return best


def build_sets(mtcnn):
    _, held = split_agedb(scan_agedb())
    rng = random.Random(11)
    ag = {i: [(cached_crop(mtcnn, p), a) for p, a in rng.sample(v, min(HOLDOUT_PER_ID, len(v)))]
          for i, v in held.items() if len(v) >= 2}
    ce = defaultdict(list)
    with open(RAW / "celeba_holdout" / "manifest.csv") as f:
        for row in csv.DictReader(f):
            ce[row["celeb_id"]].append((cached_crop(mtcnn, RAW / "celeba_holdout" / row["path"]), 0))

    def pairs(src, n, gap=0):
        ids = list(src)
        out = []
        while len(out) < n:
            i = rng.choice(ids)
            a, b = rng.sample(src[i], 2)
            if abs(a[1] - b[1]) < gap:
                continue
            out.append((a[0], b[0], 1))
            j = rng.choice([x for x in ids if x != i])
            out.append((a[0], rng.choice(src[j])[0], 0))
        return out

    sets = {
        "celeba_clean": (pairs(ce, N_PAIRS // 2), False),
        "agedb_clean": (pairs(ag, N_PAIRS // 2), False),
        "agedb_gap20": (pairs(ag, N_PAIRS // 2, gap=20), False),
        "agedb_degraded": (pairs(ag, N_PAIRS // 2), True),
        "celeba_degraded": (pairs(ce, N_PAIRS // 2), True),
    }

    def utk(kind, n):
        files = sorted((RAW / "utkface" / kind).glob("*.jpg"), key=lambda p: int(p.stem.split("_")[0]))
        return [cached_crop(mtcnn, p) for p in files[-n:]]

    return sets, utk("kid", 200), utk("old", 100)


@torch.no_grad()
def embed(model, crops, device):
    out = []
    for i in range(0, len(crops), 64):
        out.append(model(torch.stack(crops[i:i + 64]).to(device)).cpu())
    return torch.cat(out)


def eval_model(model, device, sets, kids, old):
    res = {}
    for name, (pairs, degraded) in sets.items():
        drng = random.Random(5)
        a = [plain(p[0]) for p in pairs]
        b = [degrade(p[1], drng) if degraded else plain(p[1]) for p in pairs]
        ea, eb = embed(model, a, device), embed(model, b, device)
        sims = F.cosine_similarity(ea, eb).tolist()
        same = [s for s, p in zip(sims, pairs) if p[2] == 1]
        diff = [s for s, p in zip(sims, pairs) if p[2] == 0]
        res[name] = best_acc(same, diff)
    for name, pool in (("kids_retrieval", kids), ("old_retrieval", old)):
        drng = random.Random(9)
        clean = embed(model, [plain(c) for c in pool], device)
        deg = embed(model, [degrade(c, drng) for c in pool], device)
        sim = F.normalize(deg, dim=1) @ F.normalize(clean, dim=1).T
        res[name] = (sim.argmax(dim=1) == torch.arange(len(pool))).float().mean().item()
    return res


def main():
    ckpts = sys.argv[1:] or ["checkpoints/face_embedding_head.pt"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=False)
    sets, kids, old = build_sets(mtcnn)
    del mtcnn

    rows = {}
    base = InceptionResnetV1(pretrained="vggface2", classify=False).to(device).eval()
    rows["pretrained"] = eval_model(base, device, sets, kids, old)
    for path in ckpts:
        m = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
        m.load_state_dict(torch.load(REPO_ROOT / path, map_location=device)["model_state_dict"])
        rows[path.split("/")[-1].replace(".pt", "")] = eval_model(m.eval(), device, sets, kids, old)

    cols = list(next(iter(rows.values())))
    print(f"{'model':28s} " + " ".join(f"{c:>15s}" for c in cols))
    for name, r in rows.items():
        print(f"{name:28s} " + " ".join(f"{r[c]:15.1%}" for c in cols))


if __name__ == "__main__":
    main()
