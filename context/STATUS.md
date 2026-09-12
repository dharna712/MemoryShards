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

## Next Steps

- [ ] Confirm and download training dataset (Wikimedia Commons "Category:Media
      with GPS EXIF" proposed — no signup needed, ~2-3k images, <300MB).
- [ ] EXIF extraction + reverse geocoding script.
- [ ] Event clustering (DBSCAN on time+geo).
- [ ] Captioning model integration (BLIP/CLIP).
- [ ] Timeline fusion + rendering.
- [ ] Run qualitative demo on own photos before Sept 14.
- [ ] Demo UI: fragments-to-timeline morph (research done, see
      `UI_RESEARCH.md` — native View Transitions API, not yet built).

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
  Not committed to git yet.
