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

## Next Steps — PRIORITY: pipeline has zero code as of now

- [ ] Confirm and download training dataset (Wikimedia Commons "Category:Media
      with GPS EXIF" proposed — no signup needed, ~2-3k images, <300MB).
- [ ] EXIF extraction + reverse geocoding script.
- [ ] Event clustering (DBSCAN on time+geo).
- [ ] Captioning model integration (BLIP/CLIP).
- [ ] Timeline fusion + rendering.
- [ ] Run qualitative demo on own photos before Sept 14.

UI/demo page is done for now (see above) — do not add more visual
effects before the pipeline exists. Phase 1 eval is 2026-09-14.
