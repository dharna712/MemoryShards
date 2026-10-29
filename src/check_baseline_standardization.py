"""
Is the pretrained-baseline comparison fair? Our fine-tunes were trained on RAW 0-255 crops, but the original VGGFace2 FaceNet
expects fixed standardisation ((x - 127.5) / 128). Score the pretrained model both ways on the same held-out pairs as report_eval.py.
Writes report/baseline_check.json.
"""
import csv, json, os, random, sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, auc, roc_curve

from face_data import RAW, REPO_ROOT, cached_crop, plain, scan_agedb, scan_ylfw, split_agedb, split_ylfw

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mt = MTCNN(image_size=160, margin=14, device=dev, post_process=False)
by = {}
with open(RAW / "celeba_holdout" / "manifest.csv") as f:
    for row in csv.DictReader(f):
        by.setdefault(row["celeb_id"], []).append(RAW / "celeba_holdout" / row["path"])
ce_c, ce_l = [], []
for n, (i, ps) in enumerate(sorted(by.items())):
    for p in ps[:8]:
        ce_c.append(cached_crop(mt, p)); ce_l.append(n)
_, held = split_agedb(scan_agedb())
ag_c, ag_l, ag_age = [], [], []
for n, ident in enumerate(sorted(i for i, v in held.items() if len(v) >= 5)):
    photos = sorted(held[ident], key=lambda x: x[1])
    for i in sorted(set(np.linspace(0, len(photos) - 1, min(8, len(photos))).round().astype(int))):
        ag_c.append(cached_crop(mt, photos[i][0])); ag_l.append(n); ag_age.append(photos[i][1])
_, held_y = split_ylfw(scan_ylfw())
yl_c, yl_l = [], []
for n, (i, files) in enumerate(sorted(held_y.items())):
    for p in files[:6]:
        yl_c.append(cached_crop(mt, p)); yl_l.append(n)
SETS = {"CelebA adults": (ce_c, ce_l, None, None), "AgeDB cross-age": (ag_c, ag_l, ag_age, 20), "YLFW children": (yl_c, yl_l, None, None)}


def pairs(labels, ages, gap, n=1500, seed=0):
    rng = random.Random(seed); N = len(labels)
    pos = [(i, j) for i in range(N) for j in range(i + 1, N) if labels[i] == labels[j] and (gap is None or abs(ages[i] - ages[j]) >= gap)]
    rng.shuffle(pos); pos = pos[:n]; neg = []
    while len(neg) < len(pos):
        i, j = rng.randrange(N), rng.randrange(N)
        if labels[i] != labels[j]: neg.append((i, j))
    return pos, neg


@torch.no_grad()
def embed(m, crops, standardise):
    X = [(plain(c) - 127.5) / 128.0 if standardise else plain(c) for c in crops]
    return torch.cat([m(torch.stack(X[i:i + 64]).to(dev)).cpu() for i in range(0, len(X), 64)]).numpy()


def pw(truth, pred):
    t = np.asarray(truth); p = np.asarray(pred); st = t[:, None] == t[None, :]; sp = p[:, None] == p[None, :]; iu = np.triu_indices(len(t), 1)
    a, b = st[iu], sp[iu]; tp = (a & b).sum(); return (tp / b.sum() if b.sum() else 1.0), (tp / a.sum() if a.sum() else 1.0)


m = InceptionResnetV1(pretrained="vggface2", classify=False).to(dev).eval()
out = {}
for std in (False, True):
    key = "standardised (x-127.5)/128" if std else "raw 0-255 (as in our report)"
    out[key] = {}
    for name, (crops, labels, ages, gap) in SETS.items():
        E = embed(m, crops, std); pos, neg = pairs(labels, ages, gap)
        pd = np.array([1 - float(E[i] @ E[j]) for i, j in pos]); nd = np.array([1 - float(E[i] @ E[j]) for i, j in neg])
        y = np.r_[np.ones(len(pd)), np.zeros(len(nd))]; fpr, tpr, _ = roc_curve(y, -np.r_[pd, nd])
        ths = np.sort(np.r_[pd, nd]); accs = [((pd < t).sum() + (nd >= t).sum()) / len(y) for t in ths]; b = int(np.argmax(accs))
        # grouping: best threshold by ARI in a sweep, and at the shipped 0.45
        res = {}
        for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            lab = AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric="cosine", linkage="average").fit(E).labels_
            pr, rc = pw(labels, lab); res[t] = dict(precision=float(pr), recall=float(rc), ari=float(adjusted_rand_score(labels, lab)), groups=int(len(set(lab))))
        bt = max(res, key=lambda k: res[k]["ari"])
        out[key][name] = dict(auc=float(auc(fpr, tpr)), best_acc=float(accs[b]), best_thr=float(ths[b]), same_mean=float(1 - pd.mean()), diff_mean=float(1 - nd.mean()),
                              group_at_0_45=res[0.45], group_best_ari=dict(thr=bt, **res[bt]), true_people=int(len(set(labels))))
        r = out[key][name]
        print(f"{key:30s} {name:16s} AUC {r['auc']:.3f} best acc {r['best_acc'] * 100:.1f}% (thr {r['best_thr']:.2f}) | sim same {r['same_mean']:.2f} diff {r['diff_mean']:.2f} | "
              f"grouping @0.45 P {res[0.45]['precision']:.3f} R {res[0.45]['recall']:.3f} ARI {res[0.45]['ari']:.3f} ({res[0.45]['groups']} groups) | best ARI {res[bt]['ari']:.3f} @ {bt}", flush=True)
json.dump(out, open(REPO_ROOT / "report" / "baseline_check.json", "w"), indent=1)
