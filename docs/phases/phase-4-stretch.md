# Phase 4 — Anti-forgetting and Denser Supervision (stretch)

**Status:** Not started
**Goal:** Replay and distillation, tiled DHF training, and wider Apple GPU coverage.
**Spec sections:** `docs/SPEC.md` §5.3, §10, decision D1

The task checklist and exit criteria are in the spec, §10. This file holds working notes for whoever
implements the phase. Add to it as you learn things; do not copy the checklist here.

## Notes for the implementer

- **Tiled training.** The vendored `build_two_stream_time_mask` already takes `min_context` and
  `block_size`, so fixed-length tiling needs no new mask builder. It does need its own target-frame
  definition and parity tests.
- **Sampled block lengths** (as in the paper) need a new builder; the distribution is unpublished.
- **Replay** cannot use Mistral's corpus; it is unreleased.

## Open questions

- Whether tiled training is worth its doubled token count for small fine-tuning sets.

## Log of what was done

Add dated bullets here as work lands, with the evidence (which tests pass, on what).
