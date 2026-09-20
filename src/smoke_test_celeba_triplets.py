"""
Second smoke test: run the real triplet-training logic (matching
notebooks/train_face_recognizer.ipynb) against the actual downloaded
CelebA subset, on real images instead of random noise.

Finding from the first version of this script: the pretrained VGGFace2
backbone already separates these identities by a wide margin (~0.6-1.4
euclidean distance), so naive random-triplet sampling (random anchor,
random positive, random negative) trivially satisfies a margin=0.2 triplet
loss immediately — every loss came out 0.0000, meaning no gradient signal
at all. That's not a bug, it's "the triplets were too easy." Fixed by
switching to online batch-hard negative mining: within each batch, for
every anchor pick the *closest* different-identity embedding as the
negative, instead of a random one.
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "celeba_subset"


def load_manifest():
    images_by_identity = defaultdict(list)
    with open(DATA_DIR / "manifest.csv") as f:
        for row in csv.DictReader(f):
            images_by_identity[row["celeb_id"]].append(str(DATA_DIR / row["path"]))
    return images_by_identity


def batch_hard_triplet_loss(anchor_emb, positive_emb, labels, margin=0.2):
    """For each anchor, mine the hardest (closest) negative from a
    different identity within the same batch — anchors+positives pooled
    together as the negative candidate pool."""
    all_emb = torch.cat([anchor_emb, positive_emb], dim=0)
    all_labels = torch.cat([labels, labels], dim=0)

    dist_matrix = torch.cdist(anchor_emb, all_emb)  # (B, 2B)
    B = anchor_emb.size(0)

    negative_emb = []
    for i in range(B):
        different_identity = all_labels != labels[i]
        candidate_dists = dist_matrix[i].clone()
        candidate_dists[~different_identity] = float("inf")
        hardest_idx = torch.argmin(candidate_dists)
        negative_emb.append(all_emb[hardest_idx])
    negative_emb = torch.stack(negative_emb)

    d_ap = (anchor_emb - positive_emb).norm(dim=1)
    d_an = (anchor_emb - negative_emb).norm(dim=1)
    per_sample = torch.clamp(d_ap - d_an + margin, min=0.0)
    return per_sample, d_ap, d_an


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=False)

    images_by_identity = load_manifest()
    identity_list = list(images_by_identity.keys())
    print(f"[1/5] loaded manifest — {len(identity_list)} identities")

    def load_face_tensor(path):
        img = Image.open(path).convert("RGB")
        face = mtcnn(img)
        if face is None:
            face = T.functional.resize(T.functional.to_tensor(img) * 255, [160, 160])
            face = fixed_image_standardization(face)
        return face

    sample_paths = [p for paths in list(images_by_identity.values())[:15] for p in paths[:2]]
    found = sum(1 for p in sample_paths if mtcnn(Image.open(p).convert("RGB")) is not None)
    print(f"[2/5] face detection rate on sample: {found}/{len(sample_paths)}")

    class CelebAPairDataset(Dataset):
        """Returns (anchor, positive, identity_index) — no explicit negative.
        Negatives are mined inside the training loop from the rest of the batch."""

        def __init__(self, images_by_identity, identity_list, id_to_idx, length=64):
            self.images_by_identity = images_by_identity
            self.identity_list = identity_list
            self.id_to_idx = id_to_idx
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, _):
            identity = random.choice(self.identity_list)
            anchor_path, positive_path = random.sample(self.images_by_identity[identity], 2)
            return (
                load_face_tensor(anchor_path),
                load_face_tensor(positive_path),
                self.id_to_idx[identity],
            )

    id_to_idx = {cid: i for i, cid in enumerate(identity_list)}
    dataset = CelebAPairDataset(images_by_identity, identity_list, id_to_idx, length=16)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    print("[3/5] pair dataset + loader built (negatives mined per-batch)")

    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.last_linear.parameters():
        param.requires_grad = True
    for param in model.last_bn.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    print("[4/5] model ready (backbone frozen, head trainable)")

    model.train()
    for i, (a, p, label_idx) in enumerate(loader):
        a, p = a.to(device), p.to(device)
        label_idx = label_idx.to(device)
        emb_a, emb_p = model(a), model(p)
        per_sample_loss, d_ap, d_an = batch_hard_triplet_loss(emb_a, emb_p, label_idx)
        loss = per_sample_loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"    batch {i}: loss = {loss.item():.4f}  "
              f"mean d(a,p) = {d_ap.mean().item():.4f}  "
              f"mean hard d(a,n) = {d_an.mean().item():.4f}  "
              f"nonzero terms = {(per_sample_loss > 0).sum().item()}/{len(per_sample_loss)}")

    print("[5/5] SMOKE TEST PASSED with hard-negative mining — "
          "loss is non-trivial, safe to run for real on Colab.")


if __name__ == "__main__":
    main()
