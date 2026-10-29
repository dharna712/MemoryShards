# Report notes: problems we faced and how we solved them

Raw material for the written report. Every number below is a held-out measurement recorded in STATUS.md; the commands that produce them are in `src/`.

## 1. What we built (one paragraph)

Photos in, a reconstructed timeline out. Extract (EXIF time + GPS) -> reverse geocode -> cluster into events (DBSCAN over time/geo) -> caption each event (BLIP, pretrained) -> Recognize people (MTCNN face detection + a FaceNet/Inception-ResNet-V1 embedding model we fine-tuned ourselves, then average-linkage clustering on cosine distance) -> fuse into one ordered timeline, served by a Flask API behind a static site.

## 2. Headline results (held-out data only)

| Measurement | Result |
|---|---|
| Face verification, 200 pairs from 29 unseen CelebA identities | pretrained 59% (raw pixels, mis-scaled: see P28) vs fine-tuned ~93% (v1 94.0%, v3 92.0-93.0% over three draws) |
| Grouping 230 unseen faces, average linkage | 98.7% precision, 76% recall at the shipped 0.45 (v1 98.8% / 80% at 0.5) |
| Held-out child identities (YLFW), precision at 0.45 | v1 ~0.1, v2 0.74, v3 0.77 |
| Same person across decades (AgeDB), grouping precision | v1 0.82 (at 0.5) -> v3 0.96 (at 0.45) |
| Mean over five identity-labelled sets at 0.45 | precision 0.88, recall 0.60, ARI 0.69 (v2: 0.86 / 0.54 / 0.63) |
| Live API end-to-end suite | 10/10, run through the public tunnel |

Known limits to state plainly: children are much better but not solved; the same person decades apart is usually split into two people; blurred cross-age photos are the weakest case.

## 3. Problems and solutions, in the order they happened

**P1. The pretrained face model could not separate people.** Off-the-shelf VGGFace2 FaceNet scored 59% verification on unseen identities and merged four different people into one cluster; same-person similarity (0.946) was barely above different-person (0.928). Fix: fine-tune on CelebA with a triplet loss. Same-person similarity 0.625 vs different-person 0.029, 94% accuracy. The baseline was measured before training. **Superseded: see P28. That baseline was scored on raw pixels, so the +35 points was an artefact.**

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

## 6. AWS deployment (CA3 / ICC submission, 2026-10-13)

Live: http://ec2-13-234-31-73.ap-south-1.compute.amazonaws.com/ (site + API on one EC2 t3.micro, nginx -> gunicorn). Report: `report/ICC01_3.docx` (built by `report/make_report.py`). Launcher on the Desktop now checks this AWS backend; the old tunnel launcher is kept as "(local backend)".

Design: private S3 bucket (site, code, model weights; encrypted, public access blocked), EC2 role with S3 GetObject on three prefixes + SSM only, security group TCP 80 only (no SSH; admin via Session Manager), IMDSv2, encrypted gp3, 200 MB upload cap, nginx security headers, $1 budget alert.

**P16. Old project's deployment had to go.** A previous Retail-Agentic-AI EC2 instance and S3 bucket were removed to stay in free tier; the bucket was emptied and deleted from the AWS console.
**P17. CloudFront refused.** CreateDistribution returned "account must be verified" (AWS Support ticket needed), so there is no HTTPS front door. Fallback: nginx on the instance serves the site and proxies /api/ (same origin), over HTTP. Planned fix: CloudFront/ALB + ACM once the account is verified.
**P18. 1 GB RAM.** t3.micro with a 4 GB swap file works for torch + BLIP + FaceNet; 9 photos ~45 s, 16 photos ~124 s.
**P19. First-boot install failed silently.** The user-data pip install of torch did not complete (gunicorn: ModuleNotFoundError torch), cloud-init finished in ~95 s. Diagnosed from the setup log via SSM, fixed by re-running the install; lesson: check logs, do not assume user-data succeeded.
**P20. IAM drift.** The instance role lacked s3:ListBucket, so `aws s3 sync` failed; shipped the site as one tarball under code/ instead of widening the policy.
**P21. Test fixture bug again.** A 9-photo AWS test "found 1 person" because the test reused filenames across identities; with unique names 16 photos of 4 people gave 4 people.
**P22. Credentials hygiene.** Deployment keys were kept in a plain-text file, and one command printed a token to the terminal. All keys and tokens must be rotated after submission, and key files deleted.

## 7. Cloud-security hardening (2026-10-14/15)

Live: https://13-234-31-73.nip.io/ (Let's Encrypt cert on a wildcard-DNS name for the instance IP, expires 2027-01-12, auto-renew timer; HTTP redirects to HTTPS; HSTS). Added: nginx rate limits (6 uploads/min, 10 req/s, 20 conns per IP, 429), API load shedding (one heavy upload at a time, 503 + Retry-After, max 40 photos), watchdog timer (restart API when /api/health fails), CloudWatch alarms (CPU, instance status, system status with EC2 auto-recover) -> SNS e-mail. No load balancer / WAF (cost, certificate); the report calls the above the single-instance substitute and ALB+WAF the next step.

**P23. "Relevant cloud security" was under-delivered at first.** First deployment was HTTP-only with one open port. Hardened as above; CloudFront was unavailable (account verification), so TLS is terminated on the instance.
**P24. Deployment steps were scripted.** Instance setup, IAM, security-group and teardown steps are boto3 scripts run from a terminal (commands on the instance went through SSM), so every step is re-runnable and documented.
**P25. Evidence.** AWS console session expired, so security-group and alarm evidence in the report is tables built from API reads, not screenshots.
Caveat: the hostname depends on the public IP; stopping/starting the instance changes it and breaks the certificate until reissued (an Elastic IP would fix that).


## 8. Model-performance evaluation added to the report (2026-10-15)

`src/report_eval.py` scores the pretrained FaceNet and our v1/v2/v3 on identical held-out pairs and writes `report/eval_metrics.json` plus the figures `report/img/eval_*.png` (ROC, distance histograms, confusion matrices, accuracy/AUC bars, threshold sweep, t-SNE, training curves from `data/raw/train_v3.log`, live AWS charts). Held-out sets: CelebA 29 ids/230 faces, AgeDB 60/480, YLFW 80/334; 799/755/583 balanced same/different pairs.
Headline (v3, best-threshold accuracy / AUC): adults 92.5% / 0.953 (pretrained 58.1% / 0.597); cross-age 88.9% / 0.957 (51.1% / 0.501); children 95.7% / 0.990 (60.2% / 0.633). At 0.45: adults precision 0.99, recall 0.73; grouping adults P 0.987 R 0.761 ARI 0.856; children grouping precision 0.818 (v1 0.085). Mean over three held-out sets at 0.45: precision 0.93, recall 0.71, ARI 0.79.
**Correction (P26):** earlier notes/summaries said cross-age grouping precision rose 0.82 -> 0.96; that compared v1 at its old threshold 0.5 with v3 at 0.45. At the same threshold v1 is 0.954 vs v3 0.981, so the real cross-age gain is in AUC (0.944 -> 0.957) and recall (0.49 -> 0.53), not precision. Children: v1 grouping precision 0.085 -> v3 0.818 is a real gain. Also: v1/v2/v3 are equal within noise on clean adults.
**P27:** no per-epoch validation curve was kept during training (training loss only; it plateaus at ~0.088 because only 3.9% of parameters are trainable); stated as a limitation.


## 9. Corrections and the DL project report (2026-10-19/20)

**P28. The "fine-tuning gained 35 points" story was a baseline artefact.** The pretrained VGGFace2 FaceNet expects input standardised as (x-127.5)/128; our fine-tunes were trained on raw 0-255 crops, and the baseline was scored on raw crops too (near chance). Re-measured on 799+799 held-out CelebA pairs (`src/check_baseline_standardization.py`, `src/report_eval.py`): pretrained raw 58.1% (AUC 0.597), pretrained standardised **93.6% (AUC 0.959)**, v3 92.5% (AUC 0.953). Across decades: 89.4% vs 88.9%. Children: 94.7% vs 95.7%. Grouping at 0.45 is identical for adults and cross-age (precision 0.987, recall 0.76 / 0.53) but **children differ: precision 0.098 (baseline) vs 0.818 (v3)**. So fine-tuning's value is children and one shared threshold, not adults. Do not claim the large gain.
**P29. Path traversal in the upload endpoint.** `f.save(tmp_path / f.filename)` used the client's filename. Reproduced (a `../../` name wrote outside the temp dir), fixed with base-name-only handling, regression test `src/test_filename_traversal.py`, pushed (commit 6196048). An earlier "path-traversal check" in the live security suite only verified the server did not crash.
**P30. The team's first DL report had claims the code does not support:** ArcFace training, 12 h / midnight cluster splitting, visual-medoid captioning, conflicting ARI values (0.899 / 0.885 vs measured 0.856), unsourced Table 5 / Fig. 8 / Fig. 9 numbers, a missing Chapter 6 and Table 3, and wrong software versions. Corrected report: `MemoryShards_Project_Report_3` (docx + pdf, built by `report/make_dl_report.py`, figures by `src/dl_report_figs.py`).
**P31. New measurements.** Event clustering on labelled SYNTHETIC trips (`src/evaluate_events.py`, 200 trips x 5 scenarios): mean pairwise F1 0.975 at the default 1 km / 3 h; typical day, same place on different days, same time in different places about 1.000; hard back-to-back case F1 0.875 (exact count in 52% of trips); shorter time scales separate close events better. Latency (`src/profile_latency.py`, CPU): 1.35 s per photo (BLIP caption 1.05 s = 78%, MTCNN+FaceNet 0.27 s = 20%, EXIF + geocode + clustering under 25 ms); about 7 s per photo on the 1 GB AWS t3.micro. Same-person pairs 30+ years apart matched at 0.45: baseline 26.5%, v3 32.6%.
