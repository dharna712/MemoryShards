# MemoryShards — AI Memory Reconstruction from Fragmented Digital Data

## Problem Statement

Build a system that takes fragmented digital data — photos, messages, documents,
timestamps, locations, notes — and automatically reconstructs it into a
chronological timeline of events.

Example: given hundreds of scattered photos and messages from a trip, the system
identifies dates, locations, people, and activities, and produces something like:

```
10 June, 10:30 AM – Reached Mumbai
       11:15 AM – Visited Gateway of India
        2:00 PM – Lunch at a restaurant
        6:30 PM – Returned to hotel
```

## Team

- Harsh Salunkhe (PRN 66)
- Fatima Hasan
- Dharna Sharma

## Approach

1. **Metadata extraction** — pull EXIF (timestamp, GPS) from photos; timestamp-parse
   any text notes/messages.
2. **Reverse geocoding** — GPS coordinates → human-readable place names.
3. **Event clustering** — group items into discrete "events" by time + location
   proximity (DBSCAN over a joint time/geo feature space).
4. **Captioning (the DL component)** — run each event's photo cluster through a
   pretrained vision-language model (BLIP or CLIP) to generate a short natural-
   language description of what's happening ("at the beach", "eating dinner").
5. **Fusion** — merge captioned photo-events with any parsed text notes/messages
   into a single ordered timeline.
6. **Output** — render the chronological timeline (see example above).

## Data Strategy

- **Training/dev**: public geotagged, timestamped photo dataset (travel/trip
  photos with EXIF). Candidate: a filtered YFCC100M-style subset. Exact source
  to be confirmed before download.
- **Test/demo**: our own personal photos and WhatsApp export, used purely as a
  qualitative held-out demo (no labels needed) — shows the pipeline works on a
  real, personal, multimodal timeline rather than just the training distribution.

## Why this project (context, not for the report)

Alternatives considered and dropped: AI waste classification, network intrusion
detection, multimodal fake info detection, drug response prediction, money-
laundering network detection, audio deepfake detection, keystroke dynamics
authentication, AI-generated image detection. This one was picked for being
visually demo-able and distinct from the group's other tabular/graph/image-only
ideas.
