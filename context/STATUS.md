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

## 2026-09-25 (late) — UI v3 (8.5 -> targeting 9)

Hero loop video (original artwork via `tools/render_hero_loop.py`), tech-stack
strip, CTA scramble. Then: cross-page view transitions (header stays put),
cursor tilt on the hero + magnetic primary buttons, spotlight on stage cards,
a live pipeline stepper while a request runs (no longer loops), and on the try
page the uploaded photos morph into their timeline rows (View Transitions; the
API now returns each event's filenames). Found and fixed a real mobile bug:
neither page had a viewport meta tag, so phones would have rendered the desktop
layout shrunk. Also added share metadata (description, theme-color, OG tags)
and a one-row header on phones.

## 2026-09-25 (late) — Laptop-only cleanup

The demo is shown on a laptop only, so all phone support was removed: every
`max-width` media query, the `hover: hover` and `innerWidth` guards in JS, and
the one-row mobile header. Kept: a `max-height: 720px` fallback that turns the
sticky panels into normal flow on short laptop screens so content never gets
cut off. Also removed dead code: unused CSS (old hero stats, the pre-dossier
pipeline steps), the Render-only `build.sh` and `gunicorn`, an empty tracked
file, two `.gitkeep`s, and the fallback URL to a hosted API that no longer
exists. The try page now defaults to `http://localhost:5000` (the backend runs
on the demo laptop; `?api=` still overrides), and the API answers Chrome's
private-network preflight so the deployed https page can call it.
`cluster_own_photos.py` is now a thin CLI over `recognize_people.py` instead of
a second, older copy of the logic. Setup and run steps are in README.md.

## 2026-09-25 (late) — Sample-photos button and results map

Try page: "Or try the sample photos" loads 11 CC-licensed geotagged Wikimedia
Commons photos (`web/samples/`, with `credits.json`) and runs the real pipeline
on them in one click, so the demo needs no uploads and works as a fallback.
The authors and licences (CC BY, BY-SA, CC0) are shown under the results.
Results also gain a map: coastlines from Natural Earth (public domain, 27 KB,
`web/media/land.json`, built by `tools/make_land.py`) with one numbered pin per
event, connected in time order and animated in; hovering a timeline row lights
its pin. The view frames the pins (world view for the samples, a regional view
for a single trip). The API now returns each event's lat/lon. No map service
or tiles are involved. Date column in the timeline widened so dates no longer
wrap.

## 2026-09-25 (night) — Nearby places, white theme, more videos

**"Will it be messy when photos are close?"** Simulated six trips (Pune
neighbourhoods 6-10 km apart, a short walk, a dense market, a same-state trip,
a six-state trip, same place morning and evening). Event clustering came out
right in every case, so the problems were around it:
- Place names: the offline geocoder only knows the nearest listed *town*, so
  neighbourhoods came back wrong (Baner -> "Khadki", Gateway of India -> "Uran",
  30 km away). New `src/places.py` checks the distance to that town and hedges
  ("near Khadki", or just the region when nothing is close). Optional, off by
  default: OpenStreetMap Nominatim for suburb-level names ("Baner, Pune"); it
  sends each event's coordinates to a third party, so the try page has an
  explicit checkbox. Event location is now the centroid of its photos.
- Map: adapts to scale (local grid + scale bar for one neighbourhood, coastlines
  from ~1 degree up, world view for multi-country trips), pushes overlapping pins
  apart with a leader line to the true spot, and labels them.
- Timeline: grouped by day with a stops/photos summary; rows show the time only.

**White theme is now the default** (pure white, ink-black buttons, darker amber,
soft card shadows), with the dark theme one click away. The code that followed
the OS colour scheme was removed as dead. Hero loop rendered in a white variant.

**More videos** (all original artwork, 150-290 KB each, dark and white
versions, rendered by `tools/render_scene_loops.py`): a scan sweeping across
photo frames (try page), shards assembling into a cracked photograph (closing
call to action), contact-sheet film strips ("No manual albums" scene). One
shared loader in `common.js` picks the file by theme, plays only while on
screen, and stays off for reduced-motion and data-saver.

## Accessibility pass
Lighthouse (desktop): perf 85/91, a11y 93, best-practices 100, SEO 100. Fixed the two a11y failures (low-contrast amber/muted text on white, missing main landmark); a11y now 100 locally.

## UI polish pass
Stronger hero subheadline, larger wordmark, darker dropzone/ghost-button borders and option label on the try page, results summary strip (photos/events/people) with auto-scroll after a run.

## Gallery alignment fix
Fixed .tl-gallery left offset (112px -> 202px) to match the tl-row grid math (10 padding + 96 time col + 16 gap + 64 thumb col + 16 gap), so expanded multi-photo thumbnails line up under the row caption text.

## Real geo-scenario test data (local only, not in repo)
Added src/download_scenario_photos.py: pulls real, EXIF-verified geotagged Wikimedia Commons photos clustered into the 4 scenarios discussed (same neighbourhood, same city, same state, cross-state), re-checking embedded GPS+DateTimeOriginal after download since Commons page metadata alone is not reliable. Ran all 4 against build_timeline: place labels came out honest in every case (Pune / Shivaji Nagar / near Khadki / near Pimpri for the Pune scenarios; Mumbai/Pune/Nashik separated correctly; Bangalore/Haora/Bankra separated correctly for cross-state). Photos live in data/raw/scenario_* (gitignored, not published).

## Performance: de-duplicated inline images
index.html embedded 4 unique demo photos as base64 data-URIs 10 times over (hero shatter, timeline example rows, reel cards) -> 227KB of HTML, all shipped inline with zero caching. Extracted to web/media/demo-1..4.jpg, referenced by url(), page dropped to ~19KB and the 4 images now load once and cache across every reuse. Verified all 10 reuse points render identically (hero, zip-stage example, reel).

## Error notice polish + contrast fix
demo-status.error was plain red mono text (#e3695f on white ~= 3.25:1 contrast, fails WCAG AA at normal text size). Now a tinted notice card matching the existing demo-skip-note pattern, with a proper #b3261e-based red (~6.5:1 contrast) and an icon badge. Verified visually.

## Visual: proof-section count-up + accent
The 94%/98.8%/1 proof stats now count up from 0 the first time they scroll into view (respects prefers-reduced-motion), and the headline stat (94%) is now amber to match the hero's "94% face match" callout and tie the two sections together visually.

## Visual: connecting thread through the example timeline
Added a literal amber thread with small beads running through the landing-page example timeline (gap between time and thumbnail columns), drawing in on scroll -- reinforces the "reassembled" copy. Reuses the existing .reveal/IntersectionObserver mechanism, no new JS. Verified aligned in both themes.

## Visual: built the tear-and-zip fragment effect
The "One fragment at a time" section used to just fade a single static photo in. Replaced it with the actual torn-photo-halves-zip-together effect that had been sitting as an unbuilt idea in UI ideas.txt since 2026-09-12: two jagged-seam clip-path halves of the same photo start apart and rotated, converge into place on scroll, with a small amber zipper-pull sliding down the seam. Cleaned up the now-dead .zip-half/.zip-caption entries from the shared reveal observer since the new markup drives itself off .zip-stage.is-visible descendant selectors. Verified via computed styles (before: torn apart + opacity 0; after: converged + opacity 1) in both themes; a local screenshot tool glitch on this scroll position blocked a visual capture but did not reflect a real rendering issue.

## Fixes: zip felt too fast, light theme felt empty
1) Zip effect: was 1.15s with a tiny 22px offset, barely readable. Now 2.4s with an 85-90px/4.5deg starting displacement, pull and caption re-timed to trail the pieces closing instead of overlapping.
2) Light theme empty-page feedback: pure #ffffff body + flat panel-face backgrounds outside the hero had zero texture. Added the same amber/crack radial-gradient wash (from the hero) to body (fixed/ambient) and to panel-zip/panel-pipeline/panel-timeline panel-faces, and unified those three panels onto --surface instead of panel-zip/timeline being pure white while pipeline alone was --surface. Verified visually: footer/credits area now reads warm cream instead of stark white.

## Rebuilt the zip effect per feedback: real zip, scroll-reactive both ways
Dropped the jagged torn-seam clip-path entirely (user: no zag lines). New version:
- Straight vertical seam, both photo halves split on a plain 50% line.
- Minimal flat zipper teeth: a dashed amber tick line down the seam (repeating-linear-gradient), plus a small pull tab -- both fade out via opacity: var(--zip-open) as it closes, so once fully zipped the photo reads whole with no leftover seam graphic (fades back in if you scroll back up).
- Fully scroll-reactive/bidirectional: driven by a single CSS custom property --zip-open (registered via @property as a <number> for smooth interpolation), read by every dependent transform/opacity via calc().
- Initially tried animation-timeline: view() for this (zero JS) per the CSS scroll-driven-animations research already in UI ideas.txt, but .zip-stage lives inside .panel-zip which is position: sticky (part of the existing stacked-panel system) -- confirmed by direct measurement that view() does not track progress correctly once an ancestor is sticky-pinned (--zip-open stuck at 1 through 1000px of scroll). Switched to a small scroll listener that computes progress from .panel-zip.offsetTop/offsetHeight (static geometry, unaffected by its own sticky positioning) rather than the stage's own getBoundingClientRect() (which stays constant while pinned). Verified bidirectionally correct via direct scrollTo tests at 10%/25%/48%/60% through the panel's range, matching the expected formula exactly each time.

## Fixed the real bug: zip stuck open at realistic viewport sizes
The previous fix (reading panel.offsetTop live every scroll tick) looked correct in testing but was wrong: offsetTop for a position:sticky element drifts WITH scrollY once the element engages sticky (measured directly: 1888 at load, 2012 after only 112px of scroll), instead of staying at its static flow position. That kept scrollY-offsetTop pinned near a near-constant small value forever once stuck -- looked "permanently stuck open, photo split, teeth showing" exactly as reported, and only showed up at realistic full-size viewports (900px tall), not the short/narrow viewport tested with earlier.
Fix: measure the true static top ONCE by briefly forcing position:static (getBoundingClientRect + scrollY), reverting immediately (synchronous, no visible flash), and caching that value + offsetHeight instead of re-reading them on every scroll tick. Recomputed on resize.
Verified with real wheel-scroll events (not programmatic scrollTo, which turned out not to reliably fire scroll events under this session's viewport emulation and produced false negatives/positives during debugging) at a realistic 1440x900 viewport: open=1 before the panel, correctly interpolates through the panel's range in both directions, closes fully past it, reopens on scrolling back up. Matches the formula exactly at multiple sampled scroll positions.

## Zip: real bottom-anchored hinge, per reference screenshot
Reworked the geometry per user feedback with a reference screenshot: previously both photo halves translated apart uniformly (equal gap top and bottom), which isn't how a real zipper looks. Now both pieces share transform-origin: 50% 100% (the seam's bottom point) and only rotate (+/-11deg at full open) -- the bottom stays joined like a zipper's fixed base, and only the top swings open into a wedge shape. Also flipped the pull's direction: it now starts near the BOTTOM when fully open (matching a real zipper's resting/unzipped position) and travels UP to the top as it closes, instead of the previous top-to-bottom motion. Teeth still fade via opacity: var(--zip-open), invisible once fully closed. Verified visually in both themes at the exact scroll position that shows the mid-open wedge shape.

## Zip: fixed timing (next panel covering too early) and geometry (looked disconnected)
Two real bugs from user feedback with screenshots:
1. Next panel's torn-edge card was already covering the zip photo well before the zip finished closing. Measured directly: the next sticky panel (.panel-pipeline) engages its own pin almost immediately (constant viewport position confirmed from as little as 10% into panel-zip's own range) -- the old 50%-of-panel-height close threshold left the zip visibly still-open the whole time. Shortened the close window to 16% of panel-zip's height so the zip finishes essentially at the very start of scrolling into the section, well before the next card's edge climbs over it. Verified via document.elementFromPoint at the stage center across the whole scroll range: covering never begins before --zip-open reaches 0.
2. The bottom-hinge rotation looked "disconnected" because rotating around a single pivot point necessarily swings the FAR corner of the flap up/down too (elementary circular-arc geometry) -- measured a ~55px unwanted vertical drift on the outer bottom corner at 55% open, which is exactly why the bottom didn't read as joined. Replaced rotate() with skewX(): a skew shifts x purely as a function of y with zero y-displacement, so the entire bottom edge stays flush at any angle. Re-measured after the fix: the flap/base seam gap is exactly 0px and perfectly x-aligned regardless of the current --zip-open value.
Also split each side into a flat "base" (sealed portion, no transform) and a skewed "flap" (open portion) with fixed-pixel background-size (not `cover`) so the two pieces show the correct continuation of the same image at the same scale -- confirmed the background-position math (base's image shifted up by exactly the flap's current height in pixels) is exact.
A local screenshot-tool glitch on this exact scroll position (persistent this session, unrelated to the page) blocked a final visual capture, but the geometry and background-position were confirmed correct by direct DOM measurement rather than eyeballing a screenshot.

## Zip: fixed hourglass bug (skew sign was backwards) + teeth moved to the actual edges
The skewX fix from the previous round had left/right signs swapped: skewX(-24deg) on the LEFT flap actually shifted its top edge TOWARD center (and even past it, crossing into the right half), while skewX(24deg) on the right flap did the same mirrored -- both pieces' tops collapsed inward instead of opening outward, which combined with the full-width sealed base below reads as an hourglass/bowtie. Verified the bug and the fix by computing the transformed seam-corner position directly from the CSS matrix and transform-origin (getBoundingClientRect doesn't reflect clip-path, so it can't be used to check this): with the corrected signs, the left piece's seam-top lands at 33.3% of stage width and the right piece's at 66.7%, symmetric with no crossover -- a proper wedge, confirmed mathematically.
Also moved the teeth: they were a single straight dashed line fixed at the stage's horizontal center, which floated in the middle of the open gap instead of tracking either photo edge. Now implemented as a ::after pseudo-element on each flap at that flap's own inner edge, so the teeth are skewed together with that piece's fabric and visually stay glued to it as it opens -- teeth on the edge of each piece, as asked. Removed teeth from the sealed base entirely (real interlocked/closed zipper teeth aren't visible either).

## Try page visual overhaul
The try page (the actual live demo) was noticeably plainer than the landing page -- headless screenshots confirmed a narrow 720px content column swimming in mostly-empty background at real desktop widths, a bare numbered-list stepper, and a flat dropzone. Widened the content column to 860px, redesigned the 1-2-3 guide steps into amber-outlined circles connected by thin lines (matching the dossier/proof visual language used elsewhere), and gave the dropzone icon a bigger circular amber-tinted badge with a small lift on hover. Left the results view (map, timeline, people strip) untouched since it was already verified working earlier this session and carries no regression risk from this pass.

## Results map: click-to-zoom, drag-to-pan, zoom buttons
The map was purely static (auto-fit projection, hover-to-highlight only, no zoom/pan). Added interactivity:
- Click a pin or its timeline row to zoom to 3x centered on that point.
- +/- buttons (increment scale by 1, clamped 1-6) and a reset button, top-right of the map card.
- Drag to pan once zoomed in (pointer events, grab/grabbing cursor); a drag is distinguished from a click so panning doesn't also re-trigger a pin's zoom.
Implementation: everything except the scale bar lives in one <g class="geo-zoom-layer">, and pan/zoom is a single CSS `transform: translate() scale()` on that group -- the browser interpolates and pans it for free, no per-frame viewBox math. The scale bar is deliberately left outside the zoom layer since its printed km value is only accurate for the initial fit; scaling it with zoom would make a stale number look freshly correct.
Verified the actual logic directly (inline transform values matched hand-calculated expected translate/scale for zoom-to-point, drag delta, and the zoom-out floor clamp) rather than trusting getComputedStyle -- this session's browser pane is hidden, which appears to pause CSS transition frames for backgrounded tabs in Chromium, so computed style read stale values after clicks even though the inline style (and therefore what a real, visible tab would render) was correct every time.

## Try page: results summary counts up on reveal
The photos/events/people summary strip above the timeline used to just appear with its final numbers. Now counts up from 0 with the same ease-out cubic used by the landing page's proof-section stats, so the results moment has a small payoff instead of static text. Verified the easing math directly (0/175/350/525/700ms -> 0/3/5/6/6 for a target of 6, settles exactly on target); couldn't observe the animation actually play in this session's hidden browser pane since Chromium pauses requestAnimationFrame for backgrounded tabs, but the code is the identical, already-shipped pattern from the proof section.

## Try page: more polish -- person card hover, results-summary reveal glow
- Person cards in the people strip now lift slightly and their avatar ring swaps to the crack-blue accent on hover, matching the tactile hover feel used elsewhere on the site (timeline thumbs, buttons).
- The results-summary card (photos/events/people counts) now has a one-shot amber glow as it slides in, giving the results moment a small payoff instead of a flat appearance. Pure CSS keyframe animation (no rAF), confirmed applied via computed animation-name.

## Fixed another instance of the low-contrast red
Found `.preview-remove:hover` (the x button on an uploaded photo thumbnail) still used the old #e3695f red fixed earlier for the error notice (~3.25:1 contrast, fails WCAG AA). Swapped to the same #b3261e already used elsewhere (~6.5:1), added a small hover scale for tactile feedback, and grepped the whole stylesheet for the old color to confirm no other instances remain.
