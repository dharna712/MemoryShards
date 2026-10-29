"""
Model-performance evaluation for the written report: every number and figure comes from the real checkpoints
scored on identities/images never used in training (held-out CelebA, AgeDB, YLFW).

Models compared: pretrained FaceNet (VGGFace2) vs our v1, v2 and v3 fine-tunes. All use the same fixed (seeded) pairs.
Outputs: report/img/eval_*.png and report/eval_metrics.json

Usage: python src/report_eval.py
"""

import csv
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, auc, roc_curve

from face_data import RAW, REPO_ROOT, cached_crop, degrade, plain, scan_agedb, scan_ylfw, split_agedb, split_ylfw

IMG = REPO_ROOT / "report" / "img"
IMG.mkdir(parents=True, exist_ok=True)
THR = 0.45
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = [("Pretrained, raw input", None), ("Pretrained, standardised", None), ("v1 (CelebA)", "checkpoints/face_embedding_head.pt"),
          ("v2 (+AgeDB, kids, blur)", "checkpoints/face_embedding_v2.pt"), ("v3 (shipped)", "checkpoints/face_embedding_v3.pt")]
STD = {"Pretrained, standardised"}   # the original VGGFace2 FaceNet expects (x - 127.5) / 128; our fine-tunes were trained on raw 0-255 crops
COL = {"Pretrained, raw input": "#bdbdbd", "Pretrained, standardised": "#7f7f7f", "v1 (CelebA)": "#1f77b4", "v2 (+AgeDB, kids, blur)": "#ff7f0e", "v3 (shipped)": "#2ca02c"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def load_model(path):
    m = InceptionResnetV1(pretrained="vggface2", classify=False).to(dev)
    if path:
        m.load_state_dict(torch.load(REPO_ROOT / path, map_location=dev)["model_state_dict"])
    return m.eval()


@torch.no_grad()
def embed(m, tensors, std=False):
    if std:
        tensors = [(t - 127.5) / 128.0 for t in tensors]
    out = [m(torch.stack(tensors[i:i + 64]).to(dev)).cpu() for i in range(0, len(tensors), 64)]
    return torch.cat(out).numpy()


# ------------------------------------------------------------------ data
print("[data] loading held-out sets", flush=True)
mt = MTCNN(image_size=160, margin=14, device=dev, post_process=False)
by = {}
with open(RAW / "celeba_holdout" / "manifest.csv") as f:
    for row in csv.DictReader(f):
        by.setdefault(row["celeb_id"], []).append(RAW / "celeba_holdout" / row["path"])
ce_c, ce_l = [], []
for n, (ident, ps) in enumerate(sorted(by.items())):
    for p in ps[:8]:
        ce_c.append(cached_crop(mt, p)); ce_l.append(n)
_, held = split_agedb(scan_agedb())
ag_c, ag_l, ag_age = [], [], []
for n, ident in enumerate(sorted(i for i, v in held.items() if len(v) >= 5)):
    photos = sorted(held[ident], key=lambda x: x[1])
    idx = sorted(set(np.linspace(0, len(photos) - 1, min(8, len(photos))).round().astype(int)))
    for i in idx:
        ag_c.append(cached_crop(mt, photos[i][0])); ag_l.append(n); ag_age.append(photos[i][1])
_, held_y = split_ylfw(scan_ylfw())
yl_c, yl_l = [], []
for n, (ident, files) in enumerate(sorted(held_y.items())):
    for p in files[:6]:
        yl_c.append(cached_crop(mt, p)); yl_l.append(n)
del mt
SETS = {"CelebA adults": (ce_c, ce_l, None), "AgeDB cross-age (gap>=20y)": (ag_c, ag_l, ag_age), "YLFW children": (yl_c, yl_l, None)}
print(f"[data] celeba {len(ce_c)} faces/{len(set(ce_l))} ids | agedb {len(ag_c)}/{len(set(ag_l))} | ylfw {len(yl_c)}/{len(set(yl_l))}", flush=True)


def make_pairs(labels, ages=None, min_gap=None, n=1500, seed=0):
    rng = random.Random(seed); N = len(labels)
    pos = [(i, j) for i in range(N) for j in range(i + 1, N) if labels[i] == labels[j]
           and (min_gap is None or abs(ages[i] - ages[j]) >= min_gap)]
    rng.shuffle(pos); pos = pos[:n]
    neg = []
    while len(neg) < len(pos):
        i, j = rng.randrange(N), rng.randrange(N)
        if labels[i] != labels[j]:
            neg.append((i, j))
    return pos, neg


PAIRS = {
    "CelebA adults": make_pairs(ce_l),
    "CelebA adults, one photo degraded": make_pairs(ce_l, seed=1),
    "AgeDB cross-age (gap>=20y)": make_pairs(ag_l, ag_age, 20),
    "YLFW children": make_pairs(yl_l, seed=2),
    "YLFW children, one photo degraded": make_pairs(yl_l, seed=3),
}
BASE = {"CelebA adults, one photo degraded": "CelebA adults", "YLFW children, one photo degraded": "YLFW children"}
drng = random.Random(8)
DEG = {k: [degrade(c, drng) for c in SETS[k][0]] for k in ("CelebA adults", "YLFW children")}
for k, (p, n) in PAIRS.items():
    print(f"[pairs] {k}: {len(p)} same + {len(n)} different", flush=True)


def dist_pairs(E, Ed, pairs):
    return np.array([1 - float(E[i] @ (Ed if Ed is not None else E)[j]) for i, j in pairs])


def verification_metrics(pd, nd):
    y = np.r_[np.ones(len(pd)), np.zeros(len(nd))]; score = -np.r_[pd, nd]
    fpr, tpr, thr = roc_curve(y, score); a = auc(fpr, tpr)
    fnr = 1 - tpr; eer = float(fpr[np.nanargmin(np.abs(fnr - fpr))])
    ths = np.sort(np.r_[pd, nd]); accs = [((pd < t).sum() + (nd >= t).sum()) / len(y) for t in ths]
    bi = int(np.argmax(accs))
    tp = int((pd < THR).sum()); fn = len(pd) - tp; fp = int((nd < THR).sum()); tn = len(nd) - fp
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn); f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n_same=len(pd), n_diff=len(nd), auc=float(a), eer=eer, best_acc=float(accs[bi]), best_thr=float(ths[bi]),
                tp=tp, fn=fn, fp=fp, tn=tn, acc_at_thr=(tp + tn) / len(y), precision=prec, recall=rec, f1=f1), (fpr, tpr)


def pair_conf(truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    st = truth[:, None] == truth[None, :]; sp = pred[:, None] == pred[None, :]; iu = np.triu_indices(len(truth), 1)
    t, p = st[iu], sp[iu]
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum()); tn = int((~t & ~p).sum())
    return tp, fp, fn, tn


def cluster(E, labels, thr):
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average").fit(E).labels_
    tp, fp, fn, tn = pair_conf(labels, lab)
    prec = tp / (tp + fp) if tp + fp else 1.0; rec = tp / (tp + fn) if tp + fn else 1.0
    return dict(precision=prec, recall=rec, ari=float(adjusted_rand_score(labels, lab)), clusters=int(len(set(lab))),
                true_people=int(len(set(labels))), tp=tp, fp=fp, fn=fn, tn=tn)


R = {"verification": {}, "clustering": {}, "sweep": {}}
ROC = {}
EMB = {}
for mname, path in MODELS:
    print(f"[model] {mname}", flush=True)
    m = load_model(path)
    s_ = mname in STD
    E = {k: embed(m, [plain(c) for c in SETS[k][0]], s_) for k in SETS}
    Ed = {k: embed(m, DEG[k], s_) for k in DEG}
    EMB[mname] = E
    R["verification"][mname] = {}; R["clustering"][mname] = {}
    for pk, (pos, neg) in PAIRS.items():
        base = BASE.get(pk, pk); deg = Ed[base] if pk in BASE else None
        met, roc = verification_metrics(dist_pairs(E[base], deg, pos), dist_pairs(E[base], deg, neg))
        R["verification"][mname][pk] = met; ROC[(mname, pk)] = roc
    for k in SETS:
        R["clustering"][mname][k] = cluster(E[k], SETS[k][1], THR)
    R.setdefault("sweep_all", {})[mname] = {k: [dict(thr=tt, **{x: cluster(E[k], SETS[k][1], tt)[x] for x in ("precision", "recall", "ari")}) for tt in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]] for k in SETS}
    del m


# threshold sweep (v3)
ths = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
sw = {k: [cluster(EMB["v3 (shipped)"][k], SETS[k][1], t) for t in ths] for k in SETS}
R["sweep"] = {"thresholds": ths, "per_set": {k: [dict(precision=x["precision"], recall=x["recall"], ari=x["ari"]) for x in v] for k, v in sw.items()}}
json.dump(R, open(REPO_ROOT / "report" / "eval_metrics.json", "w"), indent=1)
print("[metrics] saved", flush=True)

# ------------------------------------------------------------------ figures
# 1. ROC curves
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, pk in zip(axs, ["CelebA adults", "AgeDB cross-age (gap>=20y)", "YLFW children"]):
    for mname, _ in MODELS:
        fpr, tpr = ROC[(mname, pk)]
        ax.plot(fpr, tpr, color=COL[mname], lw=1.8, label=f"{mname} (AUC {R['verification'][mname][pk]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_title(pk); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(IMG / "eval_roc.png", dpi=200); plt.close(fig)

# 2. distance distributions
fig, axs = plt.subplots(2, 3, figsize=(11, 5.4))
for r, pk in enumerate(["CelebA adults", "AgeDB cross-age (gap>=20y)"]):
    for c, mname in enumerate(["Pretrained, raw input", "Pretrained, standardised", "v3 (shipped)"]):
        ax = axs[r][c]; E = EMB[mname][pk if pk in EMB[mname] else pk]
        pos, neg = PAIRS[pk]
        pdist = dist_pairs(E, None, pos); ndist = dist_pairs(E, None, neg)
        ax.hist(pdist, bins=40, alpha=0.65, color="#2ca02c", label="same person", density=True)
        ax.hist(ndist, bins=40, alpha=0.65, color="#d62728", label="different people", density=True)
        ax.axvline(THR, color="k", ls="--", lw=1, label="threshold 0.45")
        ax.set_title(f"{mname}: {pk}", fontsize=8.5); ax.set_xlabel("cosine distance"); ax.set_ylabel("density")
        if r == 0 and c == 0: ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(IMG / "eval_dist.png", dpi=200); plt.close(fig)

# 3. confusion matrices (v3, threshold 0.45)
fig, axs = plt.subplots(1, 4, figsize=(11, 3.1))
for ax, pk in zip(axs, ["CelebA adults", "CelebA adults, one photo degraded", "AgeDB cross-age (gap>=20y)", "YLFW children"]):
    v = R["verification"]["v3 (shipped)"][pk]; M = np.array([[v["tp"], v["fn"]], [v["fp"], v["tn"]]])
    ax.imshow(M / M.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]}\n({M[i, j] / M[i].sum() * 100:.0f}%)", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] / M[i].sum() > 0.6 else "black")
    ax.set_xticks([0, 1], ["pred same", "pred different"], fontsize=7.5); ax.set_yticks([0, 1], ["true same", "true different"], fontsize=7.5)
    ax.set_title(f"{pk}\nacc {v['acc_at_thr'] * 100:.1f}%  F1 {v['f1']:.2f}", fontsize=8); ax.grid(False)
fig.suptitle("Pair-verification confusion matrices, v3 at the shipped threshold 0.45", fontsize=10)
fig.tight_layout(); fig.savefig(IMG / "eval_confusion.png", dpi=200); plt.close(fig)

# 4. model comparison bars
pks = ["CelebA adults", "CelebA adults, one photo degraded", "AgeDB cross-age (gap>=20y)", "YLFW children", "YLFW children, one photo degraded"]
short = ["CelebA\nadults", "CelebA\ndegraded", "AgeDB\ncross-age", "YLFW\nchildren", "YLFW\ndegraded"]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6)); w = 0.16
for k, (mname, _) in enumerate(MODELS):
    axs[0].bar(np.arange(5) + (k - 2) * w, [R["verification"][mname][p]["best_acc"] * 100 for p in pks], w, color=COL[mname], label=mname)
    axs[1].bar(np.arange(5) + (k - 2) * w, [R["verification"][mname][p]["auc"] for p in pks], w, color=COL[mname])
axs[0].set_ylim(40, 100); axs[0].set_ylabel("best-threshold accuracy (%)"); axs[1].set_ylim(0.4, 1.0); axs[1].set_ylabel("ROC AUC")
for ax in axs: ax.set_xticks(range(5), short, fontsize=7.5)
axs[0].legend(fontsize=6.5, loc="lower left"); axs[0].set_title("Verification accuracy"); axs[1].set_title("ROC AUC")
fig.tight_layout(); fig.savefig(IMG / "eval_bars.png", dpi=200); plt.close(fig)

# 5. threshold sweep (v3)
fig, ax = plt.subplots(figsize=(6, 3.6))
mean = {s: np.mean([[x[s] for x in R["sweep"]["per_set"][k]] for k in SETS], axis=0) for s in ("precision", "recall", "ari")}
for s, col in zip(("precision", "recall", "ari"), ("#2ca02c", "#1f77b4", "#ff7f0e")):
    ax.plot(ths, mean[s], "o-", color=col, label={"precision": "precision", "recall": "recall", "ari": "ARI"}[s])
ax.axvline(THR, color="k", ls="--", lw=1); ax.text(THR + 0.005, 0.05, "shipped 0.45", fontsize=8)
ax.set_xlabel("average-linkage cosine-distance threshold"); ax.set_ylabel("pairwise score (mean of 3 held-out sets)"); ax.set_ylim(0, 1.02)
ax.set_title("Grouping threshold sweep (v3): precision-first choice"); ax.legend()
fig.tight_layout(); fig.savefig(IMG / "eval_sweep.png", dpi=200); plt.close(fig)

# 6. t-SNE of CelebA embeddings
ids = [i for i, _ in sorted(((i, ce_l.count(i)) for i in set(ce_l)), key=lambda x: -x[1])[:10]]
sel = [k for k, l in enumerate(ce_l) if l in ids]
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, mname in zip(axs, ["Pretrained, raw input", "Pretrained, standardised", "v3 (shipped)"]):
    X = TSNE(n_components=2, perplexity=min(15, len(sel) // 4), random_state=0, init="pca").fit_transform(EMB[mname]["CelebA adults"][sel])
    ax.scatter(X[:, 0], X[:, 1], c=[ids.index(ce_l[k]) for k in sel], cmap="tab10", s=22); ax.set_title(f"{mname}: t-SNE of 10 unseen people"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(IMG / "eval_tsne.png", dpi=200); plt.close(fig)

# 7. training curves from the real v3 log
ep, loss, dap, dan = [], [], [], []
for line in open(RAW / "train_v3.log", encoding="utf-8", errors="ignore"):
    m = re.match(r"epoch (\d+)/\d+\s+loss ([\d.]+).*d_ap ([\d.]+)\s+d_an ([\d.]+)", line)
    if m: ep.append(int(m.group(1))); loss.append(float(m.group(2))); dap.append(float(m.group(3))); dan.append(float(m.group(4)))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
axs[0].plot(ep, loss, "o-", color="#d62728"); axs[0].set_title("Training: batch-hard triplet loss (v3)"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("loss")
axs[1].plot(ep, dap, "o-", color="#2ca02c", label="anchor-positive distance"); axs[1].plot(ep, dan, "o-", color="#d62728", label="anchor-negative distance")
axs[1].set_title("Hardest positive vs hardest negative distance"); axs[1].set_xlabel("epoch"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(IMG / "eval_train.png", dpi=200); plt.close(fig)
R["training"] = dict(epochs=ep, loss=loss, d_ap=dap, d_an=dan); json.dump(R, open(REPO_ROOT / "report" / "eval_metrics.json", "w"), indent=1)

# 8. pairwise grouping confusion (v3, 0.45)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
for ax, k in zip(axs, SETS):
    c = R["clustering"]["v3 (shipped)"][k]; M = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
    ax.imshow(M / M.sum(1, keepdims=True), cmap="Greens", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:,}\n({M[i, j] / M[i].sum() * 100:.1f}%)", ha="center", va="center", fontsize=8.5)
    ax.set_xticks([0, 1], ["grouped together", "kept apart"], fontsize=7.5); ax.set_yticks([0, 1], ["same person", "different people"], fontsize=7.5)
    ax.set_title(f"{k}\nprecision {c['precision']:.2f}  recall {c['recall']:.2f}  ARI {c['ari']:.2f}", fontsize=8); ax.grid(False)
fig.suptitle("People-grouping confusion matrices over all face pairs (v3, threshold 0.45)", fontsize=10)
fig.tight_layout(); fig.savefig(IMG / "eval_group_confusion.png", dpi=200); plt.close(fig)

# 9. measured live-deployment charts
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
pts = [(1, 23), (4, 37), (9, 45), (12, 73), (12, 91), (16, 124), (16, 127), (16, 129), (16, 134), (30, 224)]
x, y = zip(*pts); axs[0].scatter(x, y, color="#1f77b4"); co = np.polyfit(x, y, 1); xs = np.linspace(0, 32, 10)
axs[0].plot(xs, np.polyval(co, xs), "--", color="#555", label=f"fit: {co[0]:.1f} s per photo + {co[1]:.0f} s")
axs[0].set_xlabel("photos in one upload"); axs[0].set_ylabel("seconds (live AWS t3.micro)"); axs[0].set_title("Processing time on AWS"); axs[0].legend(fontsize=8)
sc = ["Clean faces\n(5 people)", "Across decades\n(4 people)", "Blurred +\ndecades", "Children\n(8 kids)"]
P = [1.00, 1.00, 1.00, 0.67]; Rr = [1.00, 0.58, 0.33, 1.00]
axs[1].bar(np.arange(4) - 0.18, P, 0.36, color="#2ca02c", label="precision"); axs[1].bar(np.arange(4) + 0.18, Rr, 0.36, color="#1f77b4", label="recall")
for i in range(4):
    axs[1].text(i - 0.18, P[i] + 0.01, f"{P[i]:.2f}", ha="center", fontsize=7.5); axs[1].text(i + 0.18, Rr[i] + 0.01, f"{Rr[i]:.2f}", ha="center", fontsize=7.5)
axs[1].set_xticks(range(4), sc, fontsize=7.5); axs[1].set_ylim(0, 1.15); axs[1].set_title("End-to-end people grouping on the live site"); axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(IMG / "eval_live.png", dpi=200); plt.close(fig)
print("[figures] done", flush=True)
