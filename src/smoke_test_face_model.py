"""
Day-1 smoke test for the face-recognition training component.

Goal: prove the environment can (1) load a pretrained face-embedding
backbone, (2) run face detection, and (3) execute one real training step
(forward, loss, backward, optimizer step) before any real training time
is spent. This is a plumbing check, not a training run — a handful of
synthetic/random images is enough to prove the loop works.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn, optim
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np


def make_fake_face_image(size=160):
    """Random RGB image standing in for a real photo until we have CelebA."""
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] device = {device}")

    # --- face detection ---
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    print("[2/4] MTCNN detector loaded")

    # --- pretrained embedding backbone (VGGFace2-pretrained InceptionResnetV1) ---
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    print("[3/4] InceptionResnetV1 (pretrained=vggface2) loaded")

    # unfreeze only the last linear layer, matching the "fine-tune the head" plan
    for param in model.parameters():
        param.requires_grad = False
    for param in model.last_linear.parameters():
        param.requires_grad = True
    for param in model.last_bn.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    triplet_loss = nn.TripletMarginLoss(margin=0.2)

    # --- one fake training step: anchor/positive/negative triplet ---
    # real training will swap this for actual CelebA identity triplets
    faces = torch.stack(
        [
            torch.rand(3, 160, 160),  # anchor
            torch.rand(3, 160, 160),  # positive (same identity)
            torch.rand(3, 160, 160),  # negative (different identity)
        ]
    ).to(device)

    model.train()
    embeddings = model(faces)
    anchor, positive, negative = embeddings[0:1], embeddings[1:2], embeddings[2:3]

    loss = triplet_loss(anchor, positive, negative)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[4/4] one training step completed — loss = {loss.item():.4f}")
    print("\nSMOKE TEST PASSED: detector + pretrained backbone + trainable head "
          "+ triplet loss + optimizer step all work end-to-end on this machine.")


if __name__ == "__main__":
    main()
