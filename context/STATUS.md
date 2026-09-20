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

## Next Steps

- [x] Day 1 — face-model smoke test.
- [x] Day 2 — CelebA subset downloaded, training notebook built and
      logic-verified locally on real data. Fixed a GPU-only
      CUDA-forking crash (`num_workers=2` -> `0`) found on the first
      real Colab run.
- [x] Evaluation harness built + pretrained baseline recorded (58.5%
      verification accuracy on unseen identities) — ready to compare
      against the fine-tuned checkpoint once training finishes.
- [x] `src/cluster_own_photos.py` built and verified on CelebA data —
      ready to point at real personal photos on Day 8.
- [ ] Day 3–4 — EXIF extraction + reverse geocoding + DBSCAN clustering
      (classical pipeline, no DL — safety-net milestone).
- [ ] Day 5–7 — actually run `notebooks/train_face_recognizer.ipynb` on
      Colab for real (8 epochs); download the resulting checkpoint into
      the repo's `checkpoints/` folder (gitignored).
- [ ] Day 8 — face clustering on personal photos, using the trained
      checkpoint.
- [ ] Day 9 — captioning integration (BLIP/CLIP, inference only).
- [ ] Day 10 — timeline fusion + rendering.
- [ ] Day 11 — full qualitative demo on own photos (keep face-cluster output
      out of anything published — demo live only).
- [ ] Day 12–14 — training buffer, polish, rehearse.

UI/demo page is done for now (see above) — do not add more visual
effects before the pipeline exists. Phase 1 eval is 2026-09-14.
