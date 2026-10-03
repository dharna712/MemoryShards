"""
v2 fine-tune of the face-embedding model: same recipe as the CelebA notebook
(batch-hard triplet loss on top of the VGGFace2 backbone), but the training
mix now covers what real family/phone photos look like:

  - CelebA identities (adult celebrities, as in v1)
  - AgeDB identities (same person across decades, ages ~1-101) with
    cross-age positives preferred
  - UTKFace children and very old faces as self-supervised identities
    (no identity labels: positive = another view of the same photo)

and every positive pair is randomly degraded (blur, motion blur, low-res,
heavy JPEG, noise, low light) so the model keeps matching a face that looks
nothing like its clean counterpart.

Held out from training (used by src/evaluate_robustness.py): 60 AgeDB
identities, the last 200 UTKFace children and 100 old faces, and the same
CelebA holdout identities v1 was scored on.

Usage:
    python src/train_face_v2.py --epochs 20 --out checkpoints/face_embedding_v2.pt
    python src/train_face_v2.py --epochs 20 --unfreeze-block8 --out checkpoints/face_embedding_v2b.pt
"""

import argparse
import csv
import os
import random
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch import optim
from torch.utils.data import DataLoader, Dataset

from face_data import (RAW, REPO_ROOT, cached_crop, degrade, light_aug,
                       scan_agedb, split_agedb)

KID_HOLDOUT = 200
OLD_HOLDOUT = 100
AGEDB_PER_ID = 24
DEGRADE_P = 0.5
MIX = {"agedb": 0.45, "celeba": 0.25, "kid": 0.22, "old": 0.08}


def load_sources(mtcnn):
    print("[data] caching face crops (first run only)...")
    t0 = time.time()
    celeba = defaultdict(list)
    with open(RAW / "celeba_subset" / "manifest.csv") as f:
        for row in csv.DictReader(f):
            celeba[row["celeb_id"]].append(cached_crop(mtcnn, RAW / "celeba_subset" / row["path"]))

    train_ids, _ = split_agedb(scan_agedb())
    rng = random.Random(1)
    agedb = {}
    for ident, items in train_ids.items():
        if len(items) < 2:
            continue
        items = rng.sample(items, min(AGEDB_PER_ID, len(items)))
        agedb[ident] = [(cached_crop(mtcnn, p), age) for p, age in items]

    def utk(kind, holdout):
        files = sorted((RAW / "utkface" / kind).glob("*.jpg"), key=lambda p: int(p.stem.split("_")[0]))
        files = files[:-holdout]
        return [cached_crop(mtcnn, p) for p in files]

    kids, old = utk("kid", KID_HOLDOUT), utk("old", OLD_HOLDOUT)
    print(f"[data] celeba ids={len(celeba)} agedb ids={len(agedb)} kids={len(kids)} old={len(old)} "
          f"({time.time() - t0:.0f}s)")
    return celeba, agedb, kids, old


class MixedPairs(Dataset):
    def __init__(self, celeba, agedb, kids, old, length):
        self.celeba, self.agedb, self.kids, self.old = celeba, agedb, kids, old
        self.celeba_ids, self.agedb_ids = list(celeba), list(agedb)
        self.length = length
        self.label_of = {}
        for i in self.celeba_ids + self.agedb_ids:
            self.label_of[i] = len(self.label_of)

    def __len__(self):
        return self.length

    @staticmethod
    def _views(a, b):
        # each side independently clean-jittered or degraded; never both clean-only most of the time
        va = degrade(a) if random.random() < DEGRADE_P else light_aug(a)
        vb = degrade(b) if random.random() < DEGRADE_P else light_aug(b)
        return va, vb

    def __getitem__(self, _):
        r = random.random()
        c = 0.0
        for kind, p in MIX.items():
            c += p
            if r <= c:
                break
        if kind == "agedb":
            ident = random.choice(self.agedb_ids)
            items = self.agedb[ident]
            if random.random() < 0.6:
                # prefer a big age gap: pick the pair with the largest gap among a few draws
                cands = [random.sample(items, 2) for _ in range(4)]
                (a, ag1), (b, ag2) = max(cands, key=lambda p: abs(p[0][1] - p[1][1]))
            else:
                (a, _), (b, _) = random.sample(items, 2)
            va, vb = self._views(a, b)
            return va, vb, self.label_of[ident]
        if kind == "celeba":
            ident = random.choice(self.celeba_ids)
            a, b = random.sample(self.celeba[ident], 2)
            va, vb = self._views(a, b)
            return va, vb, self.label_of[ident]
        pool = self.kids if kind == "kid" else self.old
        a = random.choice(pool)
        # same photo, two views: one clean-jittered, one degraded
        return light_aug(a), degrade(a), 100000 + random.randrange(10**9)


def batch_hard_triplet_loss(anchor, positive, labels, margin=0.2):
    all_emb = torch.cat([anchor, positive])
    all_labels = torch.cat([labels, labels])
    dist = torch.cdist(anchor, all_emb)
    dist = dist.masked_fill(all_labels[None, :] == labels[:, None], float("inf"))
    hardest = dist.argmin(dim=1)
    neg = all_emb[hardest]
    d_ap = (anchor - positive).norm(dim=1)
    d_an = (anchor - neg).norm(dim=1)
    return torch.clamp(d_ap - d_an + margin, min=0.0), d_ap, d_an


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze-block8", action="store_true")
    ap.add_argument("--trunk-bn-train", action="store_true", help="v1 behaviour: model.train() everywhere")
    ap.add_argument("--degrade-p", type=float, default=0.5)
    ap.add_argument("--mix", default="")
    ap.add_argument("--out", default="checkpoints/face_embedding_v2.pt")
    args = ap.parse_args()

    global DEGRADE_P
    DEGRADE_P = args.degrade_p
    if args.mix:
        MIX.clear()
        MIX.update({k: float(v) for k, v in (kv.split('=') for kv in args.mix.split(','))})
    random.seed(3)
    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=False)
    celeba, agedb, kids, old = load_sources(mtcnn)
    del mtcnn

    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    for p in model.parameters():
        p.requires_grad = False
    trainable = [model.last_linear, model.last_bn]
    if args.unfreeze_block8:
        trainable.append(model.block8)
    for m in trainable:
        for p in m.parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable params: {sum(p.numel() for p in params):,} "
          f"(block8 {'on' if args.unfreeze_block8 else 'off'})")
    opt = optim.Adam(params, lr=args.lr)

    ds = MixedPairs(celeba, agedb, kids, old, length=args.steps_per_epoch * args.batch)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    out = REPO_ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        # frozen trunk stays in eval mode so its BatchNorm statistics don't drift
        if args.trunk_bn_train:
            model.train()
        else:
            model.eval()
            for m in trainable:
                m.train()
        total, nonzero, n, dap, dan = 0.0, 0, 0, 0.0, 0.0
        t0 = time.time()
        for a, b, lab in loader:
            a, b, lab = a.to(device), b.to(device), lab.to(device)
            loss_vec, d_ap, d_an = batch_hard_triplet_loss(model(a), model(b), lab)
            loss = loss_vec.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(loss_vec)
            nonzero += (loss_vec > 0).sum().item()
            n += len(loss_vec)
            dap += d_ap.sum().item()
            dan += d_an.sum().item()
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total / n:.4f}  nonzero {nonzero}/{n}  "
              f"d_ap {dap / n:.3f}  d_an {dan / n:.3f}  ({time.time() - t0:.0f}s)")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
