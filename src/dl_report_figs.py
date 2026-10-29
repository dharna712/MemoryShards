"""Figures for the Deep Learning project report (report/dl/*.png): redrawn architecture diagrams that match the code,
and Figures 5-9 regenerated from measured data (eval_metrics.json, event_eval.json, latency_profile.json) instead of unsourced numbers."""
import json, os, random, sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "dl"; OUT.mkdir(parents=True, exist_ok=True)
R = json.load(open(ROOT / "report" / "eval_metrics.json"))
EV = json.load(open(ROOT / "report" / "event_eval.json"))
LAT = json.load(open(ROOT / "report" / "latency_profile.json"))
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
NAVY = "#1F3A5F"


def box(ax, x, y, w, h, title, body="", fc="#eaf1fb", ec="#3b6ea5", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=fs, weight="bold", color=NAVY)
    if body:
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=fs - 1.2, color="#333", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.3, color="#444", connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=7.5, color="#444", style="italic")


# ---------------------------------------------------------------- Fig 1: architecture
fig, ax = plt.subplots(figsize=(10.5, 5.2)); ax.set_xlim(0, 21); ax.set_ylim(0, 10.4); ax.axis("off")
ax.text(10.5, 10.0, "MemoryShards: multimodal episodic-memory reconstruction architecture", ha="center", fontsize=11, weight="bold", color=NAVY)
box(ax, 0.2, 3.6, 3.6, 3.6, "Input", "Photos (JPEG, PNG)\nEXIF time + GPS\nundated files\n(e.g. from WhatsApp)", "#fff4e0", "#d9942b")
box(ax, 5.0, 6.5, 4.6, 2.9, "1. Metadata stage", "EXIF time + GPS parsing\noffline reverse geocoding\n(reverse_geocoder KD-tree)", "#e6f6ec", "#3a9a5c")
box(ax, 5.0, 3.0, 4.6, 2.9, "2. Event clustering", "joint time + distance metric\nDBSCAN (1 km, 3 h)\nstandalone photo = own event", "#e6f6ec", "#3a9a5c")
box(ax, 5.0, 0.3, 4.6, 2.2, "3. Captioning", "Salesforce BLIP (base)\none caption per event\n(first photo of the event)", "#f1eafb", "#7a55b8")
box(ax, 10.8, 3.0, 5.0, 5.0, "4. Face recognition", "MTCNN detect + align (160x160)\nInception-ResNet-V1 (FaceNet)\nfine-tuned: batch-hard triplet loss\n512-d L2-normalised embedding\naverage-linkage clustering, cosine 0.45\n-> people shared across events", "#fdecec", "#c0504d")
box(ax, 17.0, 3.0, 3.8, 5.0, "5. Fusion + output", "ordered timeline JSON\nundated events appended\nFlask REST API\nweb viewer (SVG map)", "#fff4e0", "#d9942b")
arrow(ax, 3.8, 5.9, 5.0, 7.9); arrow(ax, 3.8, 5.4, 5.0, 4.6); arrow(ax, 3.8, 4.6, 5.0, 1.8); arrow(ax, 3.8, 5.4, 10.8, 5.5, rad=-0.12)
arrow(ax, 7.3, 6.5, 7.3, 5.9); arrow(ax, 7.3, 3.0, 7.3, 2.5)
arrow(ax, 9.6, 4.4, 10.8, 5.2, rad=0.0); arrow(ax, 9.6, 1.4, 17.0, 3.6, rad=0.15); arrow(ax, 15.8, 5.5, 17.0, 5.5)
fig.tight_layout(); fig.savefig(OUT / "fig1_architecture.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: flowchart that matches build_timeline.py
fig, ax = plt.subplots(figsize=(10.5, 6.6)); ax.set_xlim(0, 21); ax.set_ylim(0, 13.2); ax.axis("off")
ax.text(10.5, 12.8, "Pipeline execution and partial-metadata decision flow", ha="center", fontsize=11, weight="bold", color=NAVY)
box(ax, 7.5, 11.0, 6.0, 1.3, "Uploaded photo batch", "", "#eef2f7", "#7b8fa6")
box(ax, 7.5, 9.0, 6.0, 1.5, "Read each photo + EXIF", "Pillow + timestamp (0x9003) and GPS (0x8825)", "#e8f1fb", "#3b6ea5")
box(ax, 15.2, 9.0, 5.4, 1.5, "Unreadable file", "listed under 'skipped'\n(never silently lost)", "#fdecec", "#c0504d")
ax.add_patch(plt.Polygon([[10.5, 8.4], [13.4, 7.2], [10.5, 6.0], [7.6, 7.2]], fc="#fff7cc", ec="#c9a800", lw=1.4))
ax.text(10.5, 7.2, "time AND GPS\nboth present?", ha="center", va="center", fontsize=8.5, weight="bold", color=NAVY)
box(ax, 0.6, 4.4, 7.0, 2.0, "Yes: dated + placed", "reverse geocode (offline)\nDBSCAN on joint time/distance metric", "#e6f6ec", "#3a9a5c")
box(ax, 13.4, 4.4, 7.0, 2.0, "No: time or GPS missing (or both)", "photo becomes its own 'Undated' event\nplace shown as 'Unknown location'", "#fff4e0", "#d9942b")
box(ax, 0.6, 2.0, 7.0, 1.7, "Events", "multi-photo clusters; a single photo\nnot in any cluster is its own event", "#e6f6ec", "#3a9a5c")
box(ax, 7.0, 0.2, 7.0, 1.4, "BLIP caption (first photo of each event)", "", "#f1eafb", "#7a55b8")
box(ax, 14.4, 2.0, 6.2, 1.7, "Faces on ALL photos", "MTCNN + FaceNet -> people", "#fdecec", "#c0504d")
arrow(ax, 10.5, 11.0, 10.5, 10.5); arrow(ax, 13.5, 9.75, 15.2, 9.75, "error"); arrow(ax, 10.5, 9.0, 10.5, 8.4)
arrow(ax, 8.4, 6.8, 5.0, 6.4, "yes"); arrow(ax, 12.6, 6.8, 16.2, 6.4, "no")
arrow(ax, 4.1, 4.4, 4.1, 3.7); arrow(ax, 4.1, 2.0, 8.8, 1.6); arrow(ax, 16.9, 4.4, 16.9, 3.7)
arrow(ax, 14.0, 0.9, 17.5, 2.0, "fuse into timeline JSON", rad=0.1)
fig.tight_layout(); fig.savefig(OUT / "fig2_flowchart.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: face pipeline that matches the training code
fig, ax = plt.subplots(figsize=(10.5, 3.6)); ax.set_xlim(0, 21); ax.set_ylim(0, 7.2); ax.axis("off")
ax.text(10.5, 6.8, "Face recognition and identity clustering architecture", ha="center", fontsize=11, weight="bold", color=NAVY)
xs = [0.2, 4.3, 8.4, 12.5, 16.6]; w = 3.9
specs = [("Raw photo", "any size\nseveral faces possible", "#eef2f7", "#7b8fa6"),
         ("MTCNN", "P-Net > R-Net > O-Net\nbox + landmarks\ncrop 160x160", "#e8f1fb", "#3b6ea5"),
         ("Inception-ResNet-V1", "VGGFace2 weights\ntrunk frozen\n(BatchNorm in train mode)", "#f1eafb", "#7a55b8"),
         ("Fine-tuned last layer", "linear + BatchNorm (3.9% of\nparameters), 512-d, L2-norm\nbatch-hard triplet loss", "#e6f6ec", "#3a9a5c"),
         ("Agglomerative clustering", "average linkage\ncosine distance < 0.45\n-> person IDs", "#fff4e0", "#d9942b")]
for x, (t, b, fc, ec) in zip(xs, specs):
    box(ax, x, 1.8, w, 3.2, t, b, fc, ec, fs=8.5)
for x in xs[:-1]:
    arrow(ax, x + w, 3.4, x + w + 0.4, 3.4)
fig.tight_layout(); fig.savefig(OUT / "fig4_face_pipeline.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: baseline preprocessing (measured)
V = R["verification"]; PK = "CelebA adults"
labels = ["(a) Pretrained\nraw 0-255 input", "(b) Pretrained\nstandardised input", "(c) Fine-tuned v1\n(raw, as trained)", "(d) Fine-tuned v3\n(raw, as trained)"]
keys = ["Pretrained, raw input", "Pretrained, standardised", "v1 (CelebA)", "v3 (shipped)"]
acc = [V[k][PK]["best_acc"] * 100 for k in keys]; aucs = [V[k][PK]["auc"] for k in keys]
fig, ax = plt.subplots(figsize=(8.2, 4.4)); cols = ["#e24a4a", "#2f7fd1", "#3aa56c", "#2c8c5d"]
b = ax.bar(range(4), acc, color=cols, edgecolor="#222", width=0.58)
for i, (a, u) in enumerate(zip(acc, aucs)):
    ax.text(i, a + 1.2, f"{a:.1f}%\nAUC {u:.3f}", ha="center", fontsize=8.5, weight="bold")
ax.set_xticks(range(4), labels, fontsize=8); ax.set_ylim(0, 126); ax.set_ylabel("held-out verification accuracy (%)")
ax.set_title("Verification on held-out CelebA (799 same + 799 different pairs, 29 unseen people)", fontsize=9.5, color=NAVY, weight="bold")
ax.annotate("Preprocessing matters: the same pretrained\nnetwork goes from chance to 93.6%", xy=(1, 93.6), xytext=(0.05, 112), fontsize=8.2,
            bbox=dict(boxstyle="round", fc="#eaf4ff", ec="#2f7fd1"), arrowprops=dict(arrowstyle="->", color="#2f7fd1"))
fig.tight_layout(); fig.savefig(OUT / "fig5_baseline.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Fig 6: age gap and degradation (measured here)
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from face_data import RAW, REPO_ROOT, cached_crop, degrade, plain, scan_agedb, split_agedb
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mt = MTCNN(image_size=160, margin=14, device=dev, post_process=False)
_, held = split_agedb(scan_agedb())
crops, ids, ages = [], [], []
for n, ident in enumerate(sorted(i for i, v in held.items() if len(v) >= 5)):
    photos = sorted(held[ident], key=lambda x: x[1])
    for i in sorted(set(np.linspace(0, len(photos) - 1, min(8, len(photos))).round().astype(int))):
        crops.append(cached_crop(mt, photos[i][0])); ids.append(n); ages.append(photos[i][1])
del mt
rng = random.Random(8); DEG = [degrade(c, rng) for c in crops]
ids = np.array(ids); ages = np.array(ages)
bins = [("0-10 y", 0, 10), ("10-20 y", 10, 20), ("20-30 y", 20, 30), ("30+ y", 30, 200)]


@torch.no_grad()
def emb(m, tensors, std):
    X = [(t - 127.5) / 128.0 if std else t for t in tensors]
    return torch.cat([m(torch.stack(X[i:i + 64]).to(dev)).cpu() for i in range(0, len(X), 64)]).numpy()


MODELS = [("Pretrained (standardised)", None, True, "#9aa5b8"), ("v1 (CelebA)", "checkpoints/face_embedding_head.pt", False, "#4fd1c5"),
          ("v2 (+UTKFace, AgeDB)", "checkpoints/face_embedding_v2.pt", False, "#3b82c4"), ("v3 (+YLFW, shipped)", "checkpoints/face_embedding_v3.pt", False, "#1f5fa8")]
res = {}
for name, path, std, _ in MODELS:
    m = InceptionResnetV1(pretrained="vggface2", classify=False).to(dev)
    if path:
        m.load_state_dict(torch.load(REPO_ROOT / path, map_location=dev)["model_state_dict"])
    m.eval(); E = emb(m, [plain(c) for c in crops], std); Ed = emb(m, DEG, std)
    N = len(ids); rec = []
    for _, lo, hi in bins:
        hit = tot = 0
        for i in range(N):
            for j in range(i + 1, N):
                if ids[i] == ids[j] and lo <= abs(ages[i] - ages[j]) < hi:
                    tot += 1; hit += (1 - float(E[i] @ E[j])) < 0.45
        rec.append(100 * hit / max(tot, 1))
    hit = tot = 0
    for i in range(N):
        for j in range(i + 1, N):
            if ids[i] == ids[j]:
                tot += 1; hit += (1 - float(E[i] @ Ed[j])) < 0.45
    rec.append(100 * hit / max(tot, 1)); res[name] = rec
    print(name, [round(x, 1) for x in rec], flush=True)
json.dump(res, open(ROOT / "report" / "dl_agegap.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(9, 4.4)); w = 0.2; labs = [b[0] for b in bins] + ["Half-degraded\n(blur/JPEG/dark)"]
for k, (name, _, _, col) in enumerate(MODELS):
    ax.bar(np.arange(5) + (k - 1.5) * w, res[name], w, color=col, label=name, edgecolor="#222", lw=0.4)
ax.set_xticks(range(5), labs, fontsize=8.5); ax.set_ylabel("same-person pairs matched at threshold 0.45 (%)"); ax.set_ylim(0, 100)
ax.set_title("Cross-decade robustness and degradation tolerance on held-out AgeDB (recall of same-person pairs)", fontsize=9.3, color=NAVY, weight="bold")
ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig6_agegap.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Fig 7: threshold sensitivity on the 230 held-out CelebA faces (v3, measured)
sw = R["sweep_all"]["v3 (shipped)"]["CelebA adults"]
th = [x["thr"] for x in sw]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.plot(th, [x["precision"] * 100 for x in sw], "o-", color="#2a7d5a", label="Pairwise precision (%)")
ax.plot(th, [x["recall"] * 100 for x in sw], "s-", color="#d9742b", label="Pairwise recall (%)")
ax.plot(th, [x["ari"] * 100 for x in sw], "^--", color="#3b82c4", label="Adjusted Rand index (x100)")
ax.axvline(0.45, color="#c0392b", ls="-.", lw=1.3, label="Shipped threshold (0.45)")
ax.set_xlabel("cosine distance threshold (average linkage)"); ax.set_ylabel("clustering metric"); ax.set_ylim(0, 105)
ax.set_title("Threshold sensitivity on 230 held-out CelebA faces (v3)", fontsize=9.8, color=NAVY, weight="bold"); ax.legend(fontsize=8, loc="lower center", ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig7_threshold.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Fig 8: event-clustering grid (synthetic labelled trips, measured)
GEO = [0.25, 0.5, 1.0, 2.0, 5.0]; HRS = [1.0, 3.0, 6.0, 12.0]
grid = np.array([[(3 * EV["grid"][f"{g}|{h}"] + 2 * EV["grid_hard"][f"{g}|{h}"]) / 5 for g in GEO] for h in HRS])
fig, ax = plt.subplots(figsize=(7.6, 4.4)); im = ax.imshow(grid[::-1], cmap="YlGnBu", vmin=0.7, vmax=1.0, aspect="auto")
for yi, h in enumerate(HRS[::-1]):
    for xi, g in enumerate(GEO):
        v = grid[HRS.index(h), xi]; ax.text(xi, yi, f"{v:.2f}" + (" *" if (g, h) == (1.0, 3.0) else ""), ha="center", va="center", fontsize=9, color="white" if v < 0.85 else "black", weight="bold" if (g, h) == (1.0, 3.0) else None)
ax.set_xticks(range(5), [f"{g} km" for g in GEO]); ax.set_yticks(range(4), [f"{h:g} h" for h in HRS[::-1]]); ax.grid(False)
ax.set_xlabel("geographic scale (sigma_geo)"); ax.set_ylabel("temporal scale (sigma_time)")
ax.set_title("Event-clustering pairwise F1 over 5 labelled synthetic scenarios (* = shipped 1 km, 3 h)", fontsize=9, color=NAVY, weight="bold")
fig.colorbar(im, ax=ax, label="pairwise F1"); fig.tight_layout(); fig.savefig(OUT / "fig8_event_grid.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Fig 9: measured latency breakdown (this machine, CPU)
p = LAT["per_photo_seconds"]; parts = [("EXIF extraction", p["exif"]), ("Offline reverse geocoding", p["geocode"]), ("Event clustering (DBSCAN)", p["event_clustering"]),
                                    ("BLIP captioning", p["caption"]), ("MTCNN + FaceNet recognition", p["face_recognition"])]
tot = sum(v for _, v in parts); cols = ["#6bb0e8", "#52c9bd", "#f0b04f", "#b393f0", "#f08080"]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
wedges, _ = ax.pie([v for _, v in parts], colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white"))
ax.text(0, 0, f"{tot:.2f} s\nper photo", ha="center", va="center", fontsize=12, weight="bold", color=NAVY)
ax.legend(wedges, [f"{n}  ({v:.3f} s, {100 * v / tot:.1f}%)" for n, v in parts], loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=8.5, frameon=False)
ax.set_title(f"Measured processing time per photo ({LAT['photos']} photos, CPU-only PyTorch)", fontsize=9.8, color=NAVY, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig9_latency.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures done")
