# Status Log

Running log of what's done, what's in progress, and what's next. Update this on
every commit that changes project direction or completes a milestone.

## 2026-09-12

- Topic finalized (was decided 2026-08-22, formalized here): AI memory
  reconstruction from fragmented digital data.
- Phase 1 evaluation starts 2026-09-14 (lab evaluation, no fixed deliverable
  format announced — treating it as: problem statement + dataset choice +
  working baseline pipeline).
- Repo scaffolded: `context/`, `data/`, `notebooks/`, `src/`.
- Data strategy decided: public geotagged/timestamped photo dataset for
  training, own personal photos/chat export as qualitative test-time demo.
- Not yet done: dataset sourced/downloaded, no code written, no model run.

## 2026-09-12 (later)

- Researched demo UI approach: native View Transitions API for a
  "scattered fragments → chronological timeline" morph. Decision and
  implementation notes in `context/UI_RESEARCH.md`. Not yet coded.
- Researched a target aesthetic (a "Scrapeverse" landing page, analyzed from
  a screen-recorded Instagram reel) — confirmed direction: commit fully to
  one visual metaphor rather than a generic look. Chose "shattered glass
  shards reassembling," tied directly to the project name. Full breakdown
  in `UI ideas.txt` (Projects root), item 4.

## 2026-09-12 (build)

- Built `web/index.html` — the actual demo/landing page. A real photo
  (Gateway of India, Wikimedia Commons, CC BY-SA 4.0, credited in the page
  footer) splits into 8 glass-shard pieces via CSS `clip-path`, click-toggles
  between scattered and reconstructed states with a timeline caption reveal.
  Below it: the 4-stage pipeline (Extract/Cluster/Caption/Reconstruct) and
  the full reconstructed timeline using 4 real reference photos (Marine
  Drive, Gateway of India, a restaurant thali, Four Seasons Mumbai — all
  Wikimedia Commons, properly licensed and credited).
  All content is illustrative/reference photos for the demo, NOT the
  training or personal test data — no pipeline code exists yet behind it.
- Added scroll-driven section transitions (jagged-edge panels sliding
  over each other) and a spinning 3D "memory reel" carousel of the 4
  reference photos as a closing coda. Deployed live at
  https://memoryshards.onrender.com (Render, auto-deploys off `main`).
- Debugged and fixed a real bug: an initial 3-stacked-`position:sticky`
  implementation of the panel transition had an unresolved browser
  release bug (the last panel never unstuck, blocking the reel/footer
  entirely). Replaced with `animation-timeline: view()`-driven overlap
  instead — verified working end-to-end on the deployed site.
- Deliberately skipped a further UI idea (torn-photo-then-zips-together
  effect, logged in `UI ideas.txt` item 5) to stop layering more visual
  effects — the UI is in good shape for a demo; the pipeline is not.

## 2026-09-15

- Pipeline revised to add a genuinely trained component. First considered a
  scene/event-type classifier (transfer learning on Wikimedia category tags),
  then settled on **face recognition** instead: fine-tune a face-embedding
  model (FaceNet-style) on CelebA (public, labeled identities), then cluster
  embeddings on our own photos to group "same person" across a day — same
  approach phone galleries use for People albums. Preferred over the scene
  classifier because CelebA gives clean public labels and the result demos
  more intuitively. This is still idea-pitching stage for Phase 1 — nothing
  has been trained yet, this is the plan going into the pitch so the project
  has a real "we trained a model" component and isn't purely
  pretrained-inference.
- Privacy constraint noted: face-clustering demo output (real photos of
  teammates/family) must stay out of anything pushed publicly — same rule
  as no team member names in the repo. Fine to demo live at evaluation only.

## 2026-09-20 — locked-in solo timeline (2 weeks, Harsh doing all the work)

Professor liked the idea but flagged face-recognition training as risky for
the timeline. Decision: keep face recognition (not falling back to the
scene classifier) — it's the whole pitch — but de-risk the schedule since
this is solo work, not split across the team.

**14-day plan:**

| Days | Task |
|---|---|
| 1 | Smoke test: pretrained face-embedding backbone + MTCNN + one real training step, before spending real time. **DONE 2026-09-20** — see below. |
| 2 | Download a small CelebA subset (~300–500 identities, ~20 photos each) — not the full 200K. Also grab the Wikimedia geotagged subset. |
| 3–4 | EXIF extraction + reverse geocoding script. DBSCAN event clustering on time+geo. Milestone: classical timeline pipeline runs end-to-end, no DL yet — safety net if training runs long. |
| 5–7 | Fine-tune the face-embedding model on the CelebA subset (triplet/contrastive loss, only last layer(s) unfrozen). |
| 8 | Run the trained model on personal photos: detect, embed, cluster by person. |
| 9 | Captioning integration (BLIP/CLIP, pretrained inference). First thing to cut if behind. |
| 10 | Fusion — merge time/geo clusters + face/person clusters + captions/notes into one timeline. |
| 11 | Full qualitative demo on own photos, end to end. Debug. |
| 12 | Buffer, reserved for re-running face-model training if days 5–7 didn't converge well. |
| 13 | Polish output, prep report/demo narrative. |
| 14 | Rehearse. No new code. |

**Day 1 smoke test — DONE.** `src/smoke_test_face_model.py`: loads MTCNN
detector, loads `InceptionResnetV1(pretrained="vggface2")` from
`facenet-pytorch`, freezes all but the last linear+batchnorm layer, runs one
triplet-loss forward/backward/optimizer step. Ran successfully on this
machine (CPU only, no CUDA — real training will need Colab or similar GPU
access). `facenet-pytorch` had to be installed with `--no-deps` (its own
`setup.py` pins an old Pillow that fails to build on Python 3.13; the
already-installed modern Pillow/numpy/torch/torchvision satisfy it fine).
Environment needs `KMP_DUPLICATE_LIB_OK=TRUE` set (OpenMP duplicate-runtime
conflict from having both a conda/MKL torch build and Anaconda's own OpenMP
on the same machine — cosmetic, not a correctness issue). Logged in
`requirements.txt`.

## 2026-09-20 (later) — Day 2 done: real CelebA subset + training notebook

- `src/download_celeba_subset.py` pulls a small, real CelebA subset — not
  the full ~200K images. Source: the `flwrlabs/celeba` mirror on the
  Hugging Face Hub, which serves the official CelebA images + `celeb_id`
  identity labels with no Google Drive/Kaggle auth needed. Rows are
  grouped by identity in this mirror, so the script streams just the
  first ~120 identity blocks instead of downloading everything.
  Result: **2,128 real images across 110 identities, 33MB**, saved to
  `data/raw/celeba_subset/` (gitignored) with a `manifest.csv`.
- **Real finding, caught before burning any Colab GPU time:** ran the
  actual triplet-training logic against this real data
  (`src/smoke_test_celeba_triplets.py`) and every loss came back exactly
  `0.0000`. Not a bug — the pretrained VGGFace2 backbone already separates
  these identities by a wide margin (~0.6–1.4 euclidean distance between
  random pairs), so naive random-triplet sampling (random anchor/positive/
  negative) trivially satisfies a margin=0.2 loss immediately, meaning
  zero gradient signal from the start. Fixed by switching to **online
  batch-hard negative mining**: the dataset only returns
  (anchor, positive, identity) pairs, and each training step mines the
  *hardest* (closest) different-identity embedding from within the same
  batch as the negative. Verified this produces real, nonzero loss with
  actual gradient signal before touching Colab.
- Built `notebooks/train_face_recognizer.ipynb` — self-contained Colab
  notebook: installs deps, mounts Google Drive (checkpoints saved to
  `MyDrive/MemoryShards/checkpoints/face_embedding_head.pt` so a
  disconnected session resumes instead of restarting), re-downloads the
  same CelebA subset directly in Colab, uses the corrected hard-mining
  training loop, trains 8 epochs, and runs a same-person vs
  different-person cosine-similarity sanity check at the end.
- Infra decision: Colab (free GPU) + Google Drive (checkpoint persistence)
  is the training infra — no paid hosting, nothing to set up beyond
  opening the notebook and clicking through. This machine is CPU-only
  (confirmed on Day 1), so real training has to happen on Colab, not here.

## 2026-09-20 (later still) — first real Colab run hit a GPU-only bug

Ran `notebooks/train_face_recognizer.ipynb` on Colab for real. Training
cell crashed on the first batch:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA
with multiprocessing, you must use the 'spawn' start method
```

Cause: `MTCNN` is instantiated with `device='cuda'` in the main process,
but the `DataLoader` had `num_workers=2`, which forks worker subprocesses
to load data — and a forked process can't reuse a CUDA context that was
already initialized in its parent. This only shows up on GPU; the local
CPU smoke test never exercised this path, which is exactly why the
smoke-test discipline caught the *training logic* bug (Day 2) but not
this *infra* bug — different failure classes need different tests.

Fix: `num_workers=0`. The subset is small enough that single-process data
loading isn't a real bottleneck, so it's simpler than switching
multiprocessing start methods. Notebook updated and re-sent.

## 2026-09-20 (evening) — evaluation harness + pretrained baseline

Built `src/evaluate_face_model.py`: downloads a held-out CelebA slice
(identities well past the ~120 used for training, so genuinely unseen),
runs face-verification on 200 same/different-person pairs, and reports
the best achievable accuracy at an optimal similarity threshold. Compares
"pretrained only" vs "pretrained + our checkpoint" when a checkpoint is
present — the before/after delta is the evidence fine-tuning did
something, for the report.

**Baseline result (pretrained VGGFace2, no fine-tuning, n=200 pairs,
29 unseen identities):**
- mean same-person cosine similarity: 0.949
- mean different-person cosine similarity: 0.935
- best verification accuracy: **58.5%** (threshold 0.927)

Confirmed this isn't a face-detection artifact (100% MTCNN detection rate
on the held-out set) — it's a genuine measurement. Barely-better-than-chance
baseline accuracy is actually a good sign for the pitch: it means there's
real room for the fine-tuned model to show a measurable improvement,
rather than the task already being solved by the pretrained model alone.
Re-run this script with `--checkpoint checkpoints/face_embedding_head.pt`
once the Colab training run's checkpoint is downloaded, to get the
after-fine-tuning number.

Also built `src/cluster_own_photos.py` — Day 8's actual deliverable:
point it at a folder of photos, it detects faces, embeds them (using the
fine-tuned checkpoint if present, pretrained-only otherwise), and clusters
by person (DBSCAN, cosine distance) — the phone-gallery-style grouping
that's the whole point of this stage.

Tested it against 16 images of 4 known CelebA identities (4 each) before
trusting it on real photos. Result, consistent with the 58.5% baseline
verification accuracy above: **the pretrained-only model merged all 4
people into a single cluster** at the default threshold, and even a much
stricter threshold (eps 0.4 -> 0.06) only pulled out 1 of 16 as a
singleton — everyone else still got merged. The script itself works
correctly (100% face detection, clustering ran fine) — this is the same
finding as the verification test, just visible as an actual clustering
failure instead of an accuracy number: the pretrained backbone alone
cannot separate these people. This is the concrete before/after evidence
for the report — re-run the same test once the fine-tuned checkpoint is
in `checkpoints/` and it should correctly split into ~4 clusters.

## 2026-09-21 — Day 3-4 done: classical pipeline, the safety-net milestone

Built and validated the whole non-DL half of the pipeline — EXIF
extraction, reverse geocoding, DBSCAN event clustering — with zero
training or model inference involved. This runs end-to-end and produces
a real timeline right now, independent of how the face-recognition
training turns out.

**Data**: `src/download_wikimedia_geotagged.py` searches Commons across
10 travel-shaped topics (beach, temple, street market, etc.) for photos
with confirmed location + date metadata. Two real bugs caught and fixed
along the way: (1) Commons file URLs carry `?utm_source=...` tracking
params, which broke naive file-extension checks; (2) only 42/98 initial
downloads had genuine *embedded* EXIF GPS (the rest only had it as
Commons page metadata, not in the file itself) — kept only the 42 with
real embedded GPS, and shrank originals from 795MB to 16MB by resizing
while explicitly re-attaching the original EXIF bytes (a plain resize
would've stripped it).

**Extract** (`src/extract_metadata.py`): reads DateTimeOriginal + GPS
straight from each photo's own EXIF (converts GPS DMS to decimal,
handles the GPS sub-IFD), then reverse-geocodes coordinates to a place
name via `reverse_geocoder` (offline, bundled lookup table, no API
calls/rate limits). 29/42 photos yielded full usable metadata.

**Cluster** (`src/cluster_events.py`): DBSCAN over a joint time+geo
distance (haversine km + hours, combined into one scale). Tested two
ways since the real data alone can't prove both directions:
- *Real Wikimedia photos* — found 2 unexpected genuine multi-photo
  events (a 5-photo Melbourne market series, a 2-photo Hong Kong café
  pair) sitting in the "unrelated" search results — turned out to be a
  single photographer's numbered photo series from one outing in each
  case. Correct behavior, not a bug — initially mis-flagged this as a
  failure before checking by hand.
- *Synthetic same-trip photos* (clearly fabricated, jittered around
  Gateway of India / Marine Drive) — correctly recovered exactly the
  2 fabricated events from 7 photos. This is the positive-case proof the
  real data alone couldn't provide (nothing in the real set was
  actually taken close together on purpose).

## Next Steps

- [x] Day 1 — face-model smoke test.
- [x] Day 2 — CelebA subset downloaded, training notebook built and
      logic-verified locally on real data. Fixed a GPU-only
      CUDA-forking crash (`num_workers=2` -> `0`) found on the first
      real Colab run.
- [x] Day 3-4 — classical pipeline (Extract + reverse geocode + DBSCAN
      cluster) built and validated on real Wikimedia data + a synthetic
      positive-case check. Safety-net milestone reached: a real timeline
      can be produced with zero DL involved.
- [x] Evaluation harness built + pretrained baseline recorded (58.5%
      verification accuracy on unseen identities) — ready to compare
      against the fine-tuned checkpoint once training finishes.
- [x] `src/cluster_own_photos.py` built and verified on CelebA data —
      ready to point at real personal photos on Day 8.
- [ ] Day 5–7 — actually run `notebooks/train_face_recognizer.ipynb` on
      Colab for real (8 epochs); download the resulting checkpoint into
      the repo's `checkpoints/` folder (gitignored).
- [ ] Day 8 — face clustering on personal photos, using the trained
      checkpoint.
- [x] Day 9 — captioning integration (BLIP/CLIP, inference only).
- [x] Day 10 — timeline fusion + rendering.
- [x] Live backend + UI integration (scope change, see below — not in the
      original 14-day plan, added at explicit request; ate into the
      Day 12-14 buffer).
- [ ] Day 11 — full qualitative demo on own photos (keep face-cluster output
      out of anything published — demo live only).
- [ ] Day 12–14 — training buffer, polish, rehearse (compressed to make
      room for the backend/UI work above).

## 2026-09-21 (later) — Day 9-10 done, plus a scope change: live backend + UI

Decision: instead of keeping the site as a static illustrative mockup for
Phase 1, we're wiring it to a real backend that actually runs the
pipeline on uploaded photos. Flagged before starting that this costs real
days not in the original plan — proceeding anyway, compressing the
Day 12-14 buffer to make room.

**Day 9 — Caption** (`src/caption_events.py`): BLIP
(`Salesforce/blip-image-captioning-base`), pretrained, inference only —
no training, matching the plan. Verified on real Wikimedia photos, real
generated captions (one had BLIP's known repetition quirk on a temple
photo — an off-the-shelf inference limitation, not something to fix
since we're not training this model).

**Day 10 — Fusion** (`src/build_timeline.py`): runs Extract -> Cluster ->
Caption over a folder of photos and merges the result into one ordered
timeline. **Real bug caught here**: DBSCAN's `-1` label means "not part
of any multi-photo cluster," not "these are all the same event" — the
first version grouped every standalone photo under the literal key `-1`,
merging 22 unrelated singleton photos into one fake 22-photo mega-event.
Fixed by giving every standalone record its own unique event. Verified
against the full 42-photo Wikimedia set post-fix: 24 correctly-separated
events (22 real singles + the 2 genuine multi-photo clusters found on
Day 3-4).

**Backend** (`src/api.py`): Flask API wrapping `build_timeline()` behind
`POST /api/timeline` (multipart photo upload -> JSON timeline with
base64 thumbnails). Real bug caught and fixed: Flask's `debug=True`
file-watching reloader was falsely detecting changes in torch/stdlib
files mid-request and restarting the server, killing in-flight uploads
with connection resets — set `debug=False` (production/gunicorn doesn't
use this reloader anyway, so this only affected local testing). Verified
with a real 12-photo upload end-to-end: 6 correctly-clustered events
returned over HTTP, ~3 minutes on CPU (captioning is the slow part).

**Frontend** (`web/index.html`): added a "Try it on your own photos"
section, clearly distinguished from the existing illustrative "What comes
out the other side" example above it — this one is real, calls the
backend, renders actual results. `API_BASE` auto-detects localhost vs
production so the same file works for local testing and the deployed
site. **Also caught and fixed a real, pre-existing bug while testing
this**: the page had no `<meta charset="utf-8">` at all, so em-dashes and
middots throughout the *entire existing site* (not just the new section)
were silently mojibake-corrupted in some browser/server configurations.
Tested the whole upload -> backend -> render flow live in a browser with
real photos (via a local static server + the local Flask API) — 3 real
events rendered correctly with real captions and places, and confirmed
the charset fix resolved the encoding across the whole page.

**Not done yet**: actually deploying the backend (Render web service).
Real open question before deploying: `torch` + `transformers` +
`facenet-pytorch` together are heavy — free-tier Render (512MB RAM) may
not be enough to hold the models in memory. Needs a decision (accept
free-tier risk and see, or move to a paid tier) before going live —
holding off on that until discussed.

## 2026-09-25 — Day 5-7 done: face model trained, evaluated on unseen identities

Trained `notebooks/train_face_recognizer.ipynb` on Colab (8 epochs, final
checkpoint from epoch 7) and pulled `face_embedding_head.pt` into
`checkpoints/` (gitignored, 112MB).

**Held-out verification** (`src/evaluate_face_model.py`, 200 pairs from 29
CelebA identities never seen in training):

| Model | Same-person sim | Different-person sim | Best accuracy |
|---|---|---|---|
| Pretrained VGGFace2 only | 0.946 | 0.928 | 59.0% |
| + our CelebA fine-tuning | 0.625 | 0.029 | **94.0%** |

Fine-tuning moved held-out accuracy by +35 points. The pretrained
embeddings barely separated same/different pairs (0.946 vs 0.928); the
fine-tuned ones separate them cleanly (0.625 vs 0.029).

**Clustering test** (`src/cluster_own_photos.py`, 14 photos of 4 unseen
identities — the same test where the pretrained-only model merged everyone
into one cluster): at eps 0.5 it recovers 3 of the 4 identities exactly
(4/4/4 photos, no mixing). The 4th identity has only 2 photos and splits
into two singletons; eps 0.6 still splits it and eps 0.7 merges it into
another person, so those two photos are simply hard (a real limit, not a
threshold issue). Default eps stays a judgment call for real photos.

- [x] Day 5-7 — train + download checkpoint.

## 2026-09-25 (later) — Recognize stage benchmarked and merged into the pipeline + UI

**Unseen-face clustering benchmark** (`src/evaluate_clustering.py`): 230
faces, 29 identities the model never saw. Compared clustering methods on
the fine-tuned embeddings (pairwise precision / recall / ARI):

| Method (best threshold) | Precision | Recall | ARI |
|---|---|---|---|
| DBSCAN, min_samples=1 (0.35) | 0.931 | 0.706 | 0.798 |
| Agglomerative, complete linkage (0.7) | 0.963 | 0.786 | 0.862 |
| **Agglomerative, average linkage (0.5)** | **0.988** | **0.804** | **0.883** |

DBSCAN with min_samples=1 chains different people together: precision
falls 0.93 -> 0.28 between eps 0.35 and 0.45, so it only works in a narrow
window. Average linkage holds ARI > 0.83 across thresholds 0.45-0.6, so the
pipeline uses it (threshold 0.5). Precision is what matters most here —
better to split one person in two than to merge two people. Recall 0.80
means some people get split across clusters (50 clusters for 29 people).

**Preprocessing check (a wrong turn worth recording):** training and
`evaluate_face_model.py` feed the model the raw 0-255 MTCNN crops;
`fixed_image_standardization` only runs in the rare no-face fallback path.
Adding standardization at inference collapses every face into one cluster.
Inference must match training: raw crops, no standardization.

**Merged into the app:** new `src/recognize_people.py` (detects all faces
with MTCNN, filters by detection probability >= 0.97 and size >= 48px,
embeds, clusters, returns per-person avatar + counts). `src/api.py` adds a
`people` list to the response and a `people` id list per event; if the
checkpoint is missing or recognition throws, the timeline is still
returned without people. `web/try.html` shows a "N people found" strip with
face avatars and puts the matching avatars on each event row. Landing page
now lists Recognize as stage 3 (five stages) and the hero stat reads 94%.

**End-to-end test:** 14 EXIF+GPS-tagged photos built from four unseen
CelebA identities (synthetic fixtures, temp folder only) -> 2 events, 4
people found (truth: 4), each event listing the correct people. First
request ~62s cold (model load + captioning), then faster.

**Not yet verified:** how it does on the team's own photos (different
lighting/ages/phone cameras than CelebA). Face output from personal photos
stays private: demo live only, never publish it.

- [x] Day 8 — clustering benchmarked on unseen faces; integrated.

## 2026-09-25 (evening) — Day 8b: first run on real personal photos

Ran the Recognize stage on a folder of 91 personal photos (downloaded
social-media style JPEGs, no EXIF, so they skip the timeline but exercise
face recognition). Results stay private: no crops or names recorded here.

- 82 faces kept across 64/91 photos; 14 photos had 2+ faces.
- 15 clusters: one large (59 faces in 58 photos), one mid (7), the rest
  small, 11 singletons.
- Eyeballed contact sheets (scratch only, deleted): the large cluster is
  visibly one person across glasses / sunglasses / B&W / bad lighting; the
  7-face cluster is one person. The small clusters of *children* look
  mixed, which fits the model (trained on adult celebrity faces).
- Recall on real photos is lower than CelebA: some faces of the main
  subject fall out as singletons.

**Bug found by this run (couldn't show up on CelebA, one face per photo):**
`MTCNN(keep_all=False)` makes `extract()` return only the first face, so
any group photo crashed `recognize_people`. Fixed with `keep_all=True`. The
API swallows recognition errors, so before the fix group photos silently
returned no people.

- [x] Day 8b — first real-photo run done.
- [ ] Day 11 — full qualitative demo on own photos with EXIF+GPS (these
      have none) so the timeline and the people strip appear together.

## 2026-09-25 (night) — UI redesign v2 (rated 7.5/10 before this pass)

Studied razorpay.com/buildathon and rebuilt the visual system around one
idea: the page is a day being reconstructed. Warm near-black + cream +
amber, Satoshi type with amber payoff words, hero question + giant answer,
a "no albums / no tagging / just photos in, a day out" scene, the five
stages as dossiers each carrying a measured "bar", a big-numeral proof
section (94% / 98.8% / 1) with an honest limits line, a scroll-driven HUD
clock (08:00 -> 22:00), film grain, paper-style buttons, display footer.
Try page inherits the system. Sticky panels fall back to normal flow on
small/short screens so content is never cut off. Notes in
`Projects/UI ideas.txt` section 6.
