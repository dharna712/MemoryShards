"""
Day 9 — the Caption stage: describe each photo/event with a pretrained
vision-language model (BLIP). Inference only — no training, no fine-tuning,
same as the plan says. This is intentionally the "downloaded checkpoint
used as-is" half of the pipeline, contrasted with the trained Recognize
stage.

Usage:
    python src/caption_events.py
"""

import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "Salesforce/blip-image-captioning-base"

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        print(f"[caption] loading {MODEL_NAME} (pretrained, inference only)...")
        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        _model.eval()
    return _processor, _model


def caption_image(path):
    processor, model = _load_model()
    image = Image.open(path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)


def caption_event(photo_paths):
    """Captions every photo in an event cluster and returns the first
    non-trivial one — good enough for a v1; could summarize/vote later."""
    captions = [caption_image(p) for p in photo_paths]
    return captions[0] if captions else None


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "wikimedia_geotagged"
    with open(data_dir / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:8]  # smoke test on a handful, not all 42

    for row in rows:
        path = data_dir / row["path"]
        caption = caption_image(path)
        print(f"  {row['title']:<60} -> {caption}")


if __name__ == "__main__":
    main()
