# UI Research — Fragments-to-Timeline Morph

For the demo/viva UI: a "Fragments" view (scattered photo/message/timestamp
cards) that morphs into a "Timeline" view (same cards, reflowed into
chronological order) on interaction. This is a same-document shared-element
transition, not a page navigation.

## Chosen approach: native View Transitions API

Plain HTML/CSS/JS, no framework, no build step — matches project constraints
(no frontend exists yet, tight deadline).

### How it works

- Wrap the DOM update (re-parenting/reflowing the cards from scattered
  positions into the timeline layout) in `document.startViewTransition(() =>
  { /* mutate DOM here */ })`.
- The browser snapshots the before/after state and animates position, size,
  and appearance for any element that has a matching `view-transition-name`
  in both states.
- Progressive enhancement is built in: if `document.startViewTransition`
  doesn't exist, the callback still runs (DOM updates), just without the
  animation — call the function directly as a fallback path.

### Critical gotcha: unique names, and only one at a time

`view-transition-name` values must be **unique within the document at the
moment of the transition** — if two elements share a name simultaneously, the
browser silently skips the transition for that pair and jumps state instead
(no error thrown, just a hard cut — easy to debug-loop over if you don't know
this).

Since we have N cards (unknown count, data-driven), the pattern is:
1. Assign each card a stable `view-transition-name` (e.g. `card-<id>`)
   dynamically, right before calling `startViewTransition`.
2. Clear/reassign names after the transition settles, if cards get
   reused/recycled, to avoid stale duplicate names on the next transition.

### Per-element animation control

Default transition is a simple cross-fade + resize. To customize duration/
easing per element (e.g. photos morph slower than text cards), target the
browser-generated pseudo-elements:
`::view-transition-group(card-<id>)`, `::view-transition-old(...)`,
`::view-transition-new(...)`.

### Accessibility

Must respect `prefers-reduced-motion`. Pattern: wrap the "should we animate"
decision in one helper that checks both feature support AND the media query,
so both concerns are handled in one place instead of scattered checks. When
reduced motion is preferred, either skip `startViewTransition` entirely or set
`animation: none !important` on the view-transition pseudo-elements via CSS.

### Browser support (as of testing, 2026)

Same-document transitions (what we need) are the older, better-supported half
of this API — this is not the newer cross-document/MPA navigation feature, so
support is broader and safe to rely on as the primary experience, with the
non-animated fallback (see above) covering anything that doesn't support it.

## Why not Framer Motion / GSAP Flip

Both are valid alternatives (noted in `PERMANENT_INSTRUCTIONS.txt` point 6)
but assume a framework (React) or more setup than we have time for. Framer
Motion's `layoutId` + `AnimatePresence` would be the natural upgrade path if
this ever becomes a React app later — same mental model (matching IDs across
states), just framework-managed instead of manual.

## References

- [MDN: Using the View Transition API](https://developer.mozilla.org/en-US/docs/Web/API/View_Transition_API/Using)
- [Chrome for Developers: Same-document view transitions for SPAs](https://developer.chrome.com/docs/web-platform/view-transitions/same-document)
- [web.dev: View transitions for single page applications](https://web.dev/learn/css/view-transitions-spas)
- [Animation Patterns: Shared Element Layout Transition](https://animationpatterns.art/animations/shared-element-layout-transition/)
- [CodeFronts: Dynamic View Transition Name](https://codefronts.com/motion/css-view-transitions/dynamic-view-transition-name/)
- [CodeFronts: Feature Detection & Graceful Fallback](https://codefronts.com/motion/css-view-transitions/feature-detection-graceful-fallback/)

## Alternate/complementary option: scroll-scrubbed transition

For "scroll down and the next view tries to come in; stop scrolling and it
freezes; scroll up and it reverses" — that's the **CSS Scroll-Driven
Animations API** (`animation-timeline: scroll()` / `view()`), not View
Transitions. Progress is tied directly to scroll position, so pause/reverse
behavior is native — no JS math needed. This solves a different problem than
the fragments↔timeline morph (continuous scroll-linked reveal vs. a discrete
click-triggered state swap) and the two can be combined. Full writeup,
gotchas, and a minimal code example: see `C:\Users\Harsh\Projects\UI ideas.txt`
(item 1) — kept there since it's a general-purpose reference, not
MemoryShards-specific.

## Not yet decided (needs a decision before implementation)

- Exact trigger for the transition (button click vs. scroll vs. auto-play for
  a viva demo — auto-play on a loop might be best for a hands-off
  presentation).
- Whether cards carry real pipeline output (photo thumbnail + generated
  caption + place name) or illustrative placeholder data for the first cut.
- Visual design direction — avoid the generic "AI-generated" look per
  `PERMANENT_INSTRUCTIONS.txt` point 5 (written for ChainBreach, but the
  same principle should apply here: no default purple gradients, no
  emoji-as-icons, no stock SaaS look).
