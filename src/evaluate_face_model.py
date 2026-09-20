"""
Day-8 prep: quantitatively test the face-embedding model — with and
without the fine-tuned checkpoint — on identities it has NEVER seen.

This downloads a small held-out slice of CelebA (identities past the ones
used for training) and runs a face-verification test: for many same-person
and different-person pairs, compute cosine similarity, then report how well
a single threshold separates them. This is the standard "verification
accuracy" metric used for face-recognition benchmarks (like LFW), just at
a scale that fits a two-week solo project.

Comparing "pretrained only" vs "pretrained + our fine-tuning" on this
held-out set is the evidence that fine-tuning actually helped — good for
the report, not just a vibe check.

Usage:
    python src/evaluate_face_model.py
    python src/evaluate_face_model.py --checkpoint checkpoints/face_embedding_head.pt
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_dataset
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
HELD_OUT_DIR = REPO_ROOT / "data" / "raw" / "celeba_holdout"

# training used the first 120 identity blocks (src/download_celeba_subset.py) —
# skip well past that so this is a genuinely unseen set
SKIP_IDENTITY_BLOCKS = 300
TARGET_IDENTITIES = 30
MAX_IMAGES_PER_IDENTITY = 8
MIN_IMAGES_PER_IDENTITY = 4
NUM_PAIRS = 200


def download_holdout_set():
    if (HELD_OUT_DIR / "manifest.csv").exists():
        print(f"[data] held-out set already downloaded at {HELD_OUT_DIR}")
        return

    print("[data] downloading held-out CelebA identities (unseen during training)...")
    HELD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("flwrlabs/celeba", split="train", streaming=True)

    per_identity_count = defaultdict(int)
    manifest_rows = []
    identities_seen = []
    blocks_skipped = 0
    prev_cid = None

    for example in ds:
        cid = example["celeb_id"]
        if cid != prev_cid:
            prev_cid = cid
            if blocks_skipped < SKIP_IDENTITY_BLOCKS:
                blocks_skipped += 1
                continue
            if cid not in identities_seen:
                if len(identities_seen) >= TARGET_IDENTITIES:
                    break
                identities_seen.append(cid)
        elif blocks_skipped < SKIP_IDENTITY_BLOCKS:
            continue

        if cid not in identities_seen:
            continue
        if per_identity_count[cid] >= MAX_IMAGES_PER_IDENTITY:
            continue

        idx = per_identity_count[cid]
        person_dir = HELD_OUT_DIR / str(cid)
        person_dir.mkdir(exist_ok=True)
        img_path = person_dir / f"{idx}.jpg"
        example["image"].convert("RGB").save(img_path, "JPEG", quality=92)
        manifest_rows.append((img_path.relative_to(HELD_OUT_DIR).as_posix(), cid))
        per_identity_count[cid] += 1

    kept_rows = [r for r in manifest_rows if per_identity_count[r[1]] >= MIN_IMAGES_PER_IDENTITY]
    with open(HELD_OUT_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "celeb_id"])
        writer.writerows(kept_rows)

    kept_ids = {cid for _, cid in kept_rows}
    print(f"[data] held-out set: {len(kept_rows)} images across {len(kept_ids)} identities")


def load_manifest():
    images_by_identity = defaultdict(list)
    with open(HELD_OUT_DIR / "manifest.csv") as f:
        for row in csv.DictReader(f):
            images_by_identity[row["celeb_id"]].append(str(HELD_OUT_DIR / row["path"]))
    return images_by_identity


def build_model(checkpoint_path, device):
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[model] loaded fine-tuned checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print("[model] no checkpoint given/found — evaluating the pretrained-only baseline")
    model.eval()
    return model


def make_pairs(images_by_identity, num_pairs):
    identities = list(images_by_identity.keys())
    pairs = []
    for _ in range(num_pairs // 2):
        # same-person pair
        pid = random.choice(identities)
        a, b = random.sample(images_by_identity[pid], 2)
        pairs.append((a, b, 1))
        # different-person pair
        id1, id2 = random.sample(identities, 2)
        a = random.choice(images_by_identity[id1])
        b = random.choice(images_by_identity[id2])
        pairs.append((a, b, 0))
    random.shuffle(pairs)
    return pairs


def evaluate(model, mtcnn, device, pairs):
    def embed(path):
        img = Image.open(path).convert("RGB")
        face = mtcnn(img)
        if face is None:
            face = T.functional.resize(T.functional.to_tensor(img) * 255, [160, 160])
            face = fixed_image_standardization(face)
        with torch.no_grad():
            return model(face.unsqueeze(0).to(device))

    same_sims, diff_sims = [], []
    for path_a, path_b, label in pairs:
        sim = F.cosine_similarity(embed(path_a), embed(path_b)).item()
        (same_sims if label == 1 else diff_sims).append(sim)

    # sweep thresholds, report the best verification accuracy (standard for this kind of eval)
    all_sims = sorted(same_sims + diff_sims)
    best_acc, best_thresh = 0.0, 0.0
    for t in all_sims:
        correct = sum(1 for s in same_sims if s >= t) + sum(1 for s in diff_sims if s < t)
        acc = correct / (len(same_sims) + len(diff_sims))
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    return {
        "mean_same_sim": sum(same_sims) / len(same_sims),
        "mean_diff_sim": sum(diff_sims) / len(diff_sims),
        "best_accuracy": best_acc,
        "best_threshold": best_thresh,
        "n_pairs": len(pairs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/face_embedding_head.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=False)

    download_holdout_set()
    images_by_identity = load_manifest()
    pairs = make_pairs(images_by_identity, NUM_PAIRS)
    print(f"[eval] built {len(pairs)} verification pairs from "
          f"{len(images_by_identity)} unseen identities\n")

    print("=== Baseline: pretrained VGGFace2, no fine-tuning ===")
    baseline_model = build_model(None, device)
    baseline_result = evaluate(baseline_model, mtcnn, device, pairs)
    print(f"  mean same-person similarity:      {baseline_result['mean_same_sim']:.3f}")
    print(f"  mean different-person similarity: {baseline_result['mean_diff_sim']:.3f}")
    print(f"  best verification accuracy:       {baseline_result['best_accuracy']:.1%} "
          f"(threshold {baseline_result['best_threshold']:.3f})\n")

    if os.path.exists(args.checkpoint):
        print("=== Fine-tuned: pretrained + our CelebA training ===")
        finetuned_model = build_model(args.checkpoint, device)
        finetuned_result = evaluate(finetuned_model, mtcnn, device, pairs)
        print(f"  mean same-person similarity:      {finetuned_result['mean_same_sim']:.3f}")
        print(f"  mean different-person similarity: {finetuned_result['mean_diff_sim']:.3f}")
        print(f"  best verification accuracy:       {finetuned_result['best_accuracy']:.1%} "
              f"(threshold {finetuned_result['best_threshold']:.3f})\n")

        delta = finetuned_result["best_accuracy"] - baseline_result["best_accuracy"]
        print(f"RESULT: fine-tuning changed held-out verification accuracy by {delta:+.1%}")
    else:
        print(f"No checkpoint found at {args.checkpoint} yet — only ran the baseline.")
        print("Download face_embedding_head.pt from Drive into checkpoints/ and re-run "
              "to compare against the fine-tuned model.")


if __name__ == "__main__":
    main()
