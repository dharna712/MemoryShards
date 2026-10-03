# Report notes: problems we faced and how we solved them

Raw material for the written report. Every number below is a held-out measurement recorded in STATUS.md; the commands that produce them are in `src/`.

## 1. What we built (one paragraph)

Photos in, a reconstructed timeline out. Extract (EXIF time + GPS) -> reverse geocode -> cluster into events (DBSCAN over time/geo) -> caption each event (BLIP, pretrained) -> Recognize people (MTCNN face detection + a FaceNet/Inception-ResNet-V1 embedding model we fine-tuned ourselves, then average-linkage clustering on cosine distance) -> fuse into one ordered timeline, served by a Flask API behind a static site.

## 2. Headline results (held-out data only)

| Measurement | Result |
|---|---|
| Face verification, 200 pairs from 29 unseen CelebA identities | pretrained 59% -> fine-tuned ~93% (v1 94.0%, v3 92.0-93.0% over three draws) |
| Grouping 230 unseen faces, average linkage | 98.7% precision, 76% recall at the shipped 0.45 (v1 98.8% / 80% at 0.5) |
| Held-out child identities (YLFW), precision at 0.45 | v1 ~0.1, v2 0.74, v3 0.77 |
| Same person across decades (AgeDB), grouping precision | v1 0.82 (at 0.5) -> v3 0.96 (at 0.45) |
| Mean over five identity-labelled sets at 0.45 | precision 0.88, recall 0.60, ARI 0.69 (v2: 0.86 / 0.54 / 0.63) |
| Live API end-to-end suite | 10/10, run through the public tunnel |

Known limits to state plainly: children are much better but not solved; the same person decades apart is usually split into two people; blurred cross-age photos are the weakest case.

## 3. Problems and solutions, in the order they happened

**P1. The pretrained face model could not separate people.** Off-the-shelf VGGFace2 FaceNet scored 59% verification on unseen identities and merged four different people into one cluster; same-person similarity (0.946) was barely above different-person (0.928). Fix: fine-tune on CelebA with a triplet loss. Same-person similarity 0.625 vs different-person 0.029, 94% accuracy. The baseline was measured before training so the +35 points is evidence, not a vibe.

**P2. Random triplets gave zero loss.** With the pretrained backbone, random (anchor, positive, negative) triplets already satisfied the margin, so every loss was 0.0000 and nothing could learn. Fix: batch-hard mining, taking the hardest different-identity embedding from the same batch as the negative. Loss became real and decreasing.

**P3. GPU-only crash on the first Colab run.** DataLoader workers forked a process that already held a CUDA context. Fix: `num_workers=0`.

**P4. Inference preprocessing had to match training.** Adding `fixed_image_standardization` at inference collapsed every face into one cluster. Training had used raw 0-255 MTCNN crops. Fix: inference uses exactly the training preprocessing.

**P5. DBSCAN chained different people together.** At min_samples=1, precision fell from 0.93 to 0.28 between eps 0.35 and 0.45. Fix: benchmark three clustering methods on 230 unseen faces; average linkage held ARI > 0.83 across thresholds 0.45-0.6.

**P6. Event clustering bug: `-1` is not an event.** DBSCAN's -1 label means "no cluster", and the first fusion version merged 22 unrelated singleton photos into one fake mega-event. Fix: every standalone photo is its own event.

**P7. Group photos silently produced no people.** `MTCNN(keep_all=False)` returns only the first face, so any multi-face photo crashed recognition, and the API swallowed the error. Found only on real personal photos (CelebA is one face per image). Fix: `keep_all=True`.

**P8. WhatsApp-shared photos were dropped.** WhatsApp strips EXIF, and photos with no timestamp/GPS were filtered into a `skipped` list and never appeared, even though the face model could recognise them. Fix: an undated photo becomes its own "Undated" event; `skipped` is reserved for unreadable files.

**P9. Backend infrastructure.** Flask's debug reloader restarted the server mid-request when it saw torch/stdlib files change, killing uploads (fix: `debug=False`). The page had no `<meta charset>`, corrupting dashes site-wide. Render's auto-deploy webhook stopped firing, so pushes never went live until deploys were triggered by hand.

**P10. Our own improvement attempt failed, and we diagnosed why.** The first age/kids/blur retraining scored ~60%, close to the untrained model. The old notebook calls `model.train()` on the whole network, so the frozen layers' BatchNorm statistics adapt to the new faces; our v2 script froze them (eval mode). Running the old recipe on clean data reproduced 92%, which isolated BatchNorm adaptation as most of the original gain. Retraining with it on restored accuracy and then improved it. Lesson: reproduce the baseline recipe before changing the data.

**P11. Our evaluation was too easy, so we built harder ones.** The CelebA benchmark is adult celebrities, mostly clean. We added held-out sets for what real photos look like: AgeDB identities across decades, blurred / low-res / JPEG-crushed / dark versions, held-out child identities. Everything is scored on fixed seeded pairs from identities never trained on.

**P12. Children.** The model was trained on adult celebrities, and v1 merged every child into one person (precision ~0.1). First attempt: UTKFace children with no identity labels, trained as "two views of one photo" (self-supervised). It helped (v2 precision 0.74 on real child identities) but cannot teach true identity. Second attempt: YLFW, an identity-labelled child dataset, added v3 0.77. Still the weakest area.

**P13. One threshold tuned on one population.** 0.5 was tuned on adult CelebA alone. A sweep over adults, cross-age and child sets showed it merged different people too readily; 0.45 raised mean precision 0.81 -> 0.88 at the cost of mean recall 0.66 -> 0.60. We prioritise precision because merging two people is a worse error than splitting one.

**P14. A dataset that downloaded at one file every half second.** YLFW is 29,436 tiny files; the standard loader fetches them one at a time (hours of pure latency), and the Hub's parquet export holds only file references. Fix: download just the aligned folder (9,810 files) over 24 parallel connections with backoff (64 connections were rate-limited), resumable. A few minutes instead of hours.

**P15. Test fixtures can lie.** Re-encoding a photo with PIL strips its EXIF, which made our own EXIF test fail; the fix was to upload the original bytes. Separately, a background job that appeared dead was still training and a job that appeared alive had been killed, so we now confirm state from the process and log instead of assuming.

## 4. Datasets and licences

| Dataset | Used for | Note |
|---|---|---|
| CelebA (via `flwrlabs/celeba`) | base fine-tune; adult held-out identities | public |
| Wikimedia Commons geotagged photos | timeline logic, EXIF/GPS tests | CC-licensed |
| AgeDB (HF mirror) | same person across decades | research use; training/eval only |
| YLFW (HF mirror `hieupth/ylfw`) | identity-labelled children | research use; training/eval only |
| UTKFace (HF mirror) | children and elderly, self-supervised | research use; training/eval only |

`data/` and `checkpoints/` are gitignored; nothing is redistributed. Face output on the team's own photos is never published (live demo only).

## 5. What we would do next

Cross-age-aware merge pass; harder negative mining for children or a second identity-labelled child set; parse messages/notes (the problem statement lists them; the pipeline does not yet); a captioning benchmark; evaluate and tune the event-clustering parameters against labelled events. Product ideas beyond the model are listed in the discussion section of the session notes (user-correctable people clusters, event co-occurrence).
