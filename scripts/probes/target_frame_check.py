"""Probe: which target frame are the released checkpoints calibrated in? (spec §4.3, decision D2)

    uv run python scripts/probes/target_frame_check.py --size 5M
    uv run python scripts/probes/target_frame_check.py --size 28M

Background. The loss compares predictions with targets normalised per horizon patch in a frame
`(loc_j, scale_j)`. Patch 0 is framed by the context alone. For later patches there are two
candidate definitions: `median_folded` (context plus the model's own predicted medians of earlier
patches; what upstream's forecaster denormalises with) and `true_values` (context plus the true
values of earlier patches). The paper's wording does not settle which one pretraining used.

Method. Take the pretrained model's normalised quantiles and score them against the targets
normalised in each frame, split by horizon patch. A model is calibrated in the frame it was
trained in, so the frame with nominal interval coverage and lower pinball loss on patches >= 1 is
the likely training frame. Patch 0 is identical under both frames and serves as the control.

The two frames only differ when earlier forecast patches move the running statistics, so the probe
uses short contexts (2 and 8 patches), a long horizon (16 patches) and series with real forecast
error (random walks, trend plus seasonality with noise, level shifts, changing volatility).

Reading the result (fixed before running):

- Confirms D2 on evidence: `median_folded` has lower loss on patches >= 1 with the bootstrap
  interval of the difference excluding zero, and coverage at least as close to nominal.
- Pretraining used `true_values`: the reverse. D2 is still what deployment does (acceptance test
  T3), so the default would stay, but the spec must then say that fine-tuning starts from a
  mismatched head, and Phase 3 must compare both frames on deployed raw-space metrics.
- Inconclusive: the interval includes zero, or the two criteria disagree.

Note added 2026-11-21, after the first run. By the rule above the outcome is "inconclusive",
because the two criteria disagree: `true_values` has the lower loss, `median_folded` has the
nominal coverage. The loss criterion was ill-posed and should not have been in the rule. Each
frame's loss is measured in that frame's own normalised units, and a frame with larger scales
scores lower whatever the model predicts. The probe now prints the signed mean of
`log(scale_A / scale_B)` so that this can be checked against the loss gap. Coverage does not have
that problem: a model is calibrated in the frame it was trained in. Coverage alone decides; see
docs/VERIFICATION.md for the numbers and their limits.

Results are recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from timestral_tune.data.batch import TrainingBatch
from timestral_tune.losses import pinball_loss
from timestral_tune.models import load_model, normalise_targets, resolve_device, target_frames, training_forward

HORIZON_PATCHES = 16
CONTEXT_PATCHES = (2, 8)
PATCH_GROUPS = (("patch 0 (control)", slice(0, 1)), ("patches 1-3", slice(1, 4)), ("patches 4-15", slice(4, 16)))
INTERVALS = ((0.5, 0.25, 0.75), (0.8, 0.10, 0.90), (0.9, 0.05, 0.95))
FRAMES = ("median_folded", "true_values")


def generate(kind: str, rng: np.random.Generator, series: int, steps: int) -> np.ndarray:
    """`(series, steps)` synthetic series with real forecast error."""
    t = np.arange(steps)
    noise = rng.standard_normal((series, steps))
    if kind == "random_walk":
        drift = rng.normal(0, 0.05, (series, 1))
        return (noise + drift).cumsum(-1) + rng.normal(0, 20, (series, 1))
    if kind == "trend_seasonal":
        period = rng.choice([24, 48, 96], (series, 1))
        amplitude = rng.uniform(1, 5, (series, 1))
        slope = rng.normal(0, 0.02, (series, 1))
        phase = rng.uniform(0, 2 * np.pi, (series, 1))
        return amplitude * np.sin(2 * np.pi * t / period + phase) + slope * t + noise + rng.normal(0, 10, (series, 1))
    if kind == "level_shift":
        values = np.zeros((series, steps))
        for i in range(1, steps):
            values[:, i] = 0.9 * values[:, i - 1] + noise[:, i]
        shift_at = rng.integers(steps // 4, steps, (series, 1))
        return values + (t >= shift_at) * rng.normal(0, 6, (series, 1)) + rng.normal(0, 10, (series, 1))
    if kind == "volatility":
        log_vol = np.zeros((series, steps))
        shocks = rng.standard_normal((series, steps)) * 0.15
        for i in range(1, steps):
            log_vol[:, i] = 0.98 * log_vol[:, i - 1] + shocks[:, i]
        return (np.exp(log_vol) * noise).cumsum(-1) * 0.5 + rng.normal(0, 10, (series, 1))
    raise ValueError(kind)


KINDS = ("random_walk", "trend_seasonal", "level_shift", "volatility")


def make_batch(values: np.ndarray, context_steps: int) -> TrainingBatch:
    data = torch.as_tensor(values, dtype=torch.float32)
    rows = data.shape[0]
    context, targets = data[:, :context_steps], data[:, context_steps:]
    return TrainingBatch(
        context=context,
        context_mask=torch.ones_like(context, dtype=torch.bool),
        group_ids=torch.arange(rows),
        role_ids=None,
        known_future=None,
        known_future_mask=None,
        targets=targets,
        loss_weight=torch.ones_like(targets, dtype=torch.bool),
    )


def bootstrap_interval(per_series: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> tuple[float, float]:
    """95% bootstrap interval of the mean of `per_series`."""
    index = rng.integers(0, per_series.size, (draws, per_series.size))
    means = per_series[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="5M", choices=["5M", "28M", "123M", "209M"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--series", type=int, default=256, help="series per generator and context length")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = load_model(f"mistralai/Timestral-{args.size}", device=device)
    config = model.config
    patch = config.patch_size
    levels = torch.tensor(config.quantile_knots)
    level_index = {round(level, 2): i for i, level in enumerate(config.quantile_knots)}
    rng = np.random.default_rng(args.seed)
    horizon_steps = HORIZON_PATCHES * patch
    print(f"== Timestral-{args.size} on {device.type} | torch {torch.__version__} | {args.series} series per cell ==")

    for context_patches in CONTEXT_PATCHES:
        context_steps = context_patches * patch
        # per frame: per-position pinball loss and normalised targets, (N, H); shared quantiles (N, H, Q)
        losses = {frame: [] for frame in FRAMES}
        targets = {frame: [] for frame in FRAMES}
        frames = {frame: [] for frame in FRAMES}
        quantiles = []
        for kind in KINDS:
            batch = make_batch(generate(kind, rng, args.series, context_steps + horizon_steps), context_steps).to(
                device
            )
            with torch.no_grad():
                output = training_forward(model, batch)
                quantiles.append(output.quantiles.cpu())
                for frame in FRAMES:
                    loc, scale = target_frames(model, batch, output, frame)
                    z = normalise_targets(model, batch, loc, scale)
                    losses[frame].append(pinball_loss(output.quantiles, z, levels, reduction="none").cpu())
                    targets[frame].append(z.cpu())
                    frames[frame].append((loc.cpu(), scale.cpu()))
        q = torch.cat(quantiles)
        loss = {frame: torch.cat(losses[frame]).unflatten(-1, (HORIZON_PATCHES, patch)) for frame in FRAMES}
        z = {frame: torch.cat(targets[frame]).unflatten(-1, (HORIZON_PATCHES, patch)) for frame in FRAMES}
        q = q.unflatten(1, (HORIZON_PATCHES, patch))
        loc = {frame: torch.cat([f[0] for f in frames[frame]]) for frame in FRAMES}
        scale = {frame: torch.cat([f[1] for f in frames[frame]]) for frame in FRAMES}

        print(f"\n-- context {context_patches} patches ({context_steps} steps), horizon {HORIZON_PATCHES} patches --")
        kind_of = np.repeat(np.array(KINDS), args.series)
        for name, group in PATCH_GROUPS:
            gap_loc = ((loc["median_folded"] - loc["true_values"]).abs() / scale["true_values"])[:, group].mean().item()
            log_ratio = (scale["median_folded"] / scale["true_values"]).log()[:, group]
            per_series = {frame: loss[frame][:, group].mean(dim=(1, 2)).numpy() for frame in FRAMES}
            difference = per_series["median_folded"] - per_series["true_values"]
            low, high = bootstrap_interval(difference, rng)
            print(f"{name}")
            print(f"   frame gap: |loc_A - loc_B| / scale_B = {gap_loc:.3f}, |log(scale_A / scale_B)| = "
                  f"{log_ratio.abs().mean().item():.3f}, signed mean log(scale_A / scale_B) = "
                  f"{log_ratio.mean().item():+.3f}")  # fmt: skip
            inside_by_frame = {}
            for frame in FRAMES:
                coverage = []
                for nominal, lower, upper in INTERVALS:
                    inside = (z[frame][:, group] >= q[:, group, :, level_index[lower]]) & (
                        z[frame][:, group] <= q[:, group, :, level_index[upper]]
                    )
                    coverage.append(f"{int(nominal * 100)}%: {inside.float().mean().item():.3f}")
                    if nominal == 0.8:
                        inside_by_frame[frame] = inside
                print(f"   {frame:14s} pinball {per_series[frame].mean():.4f} | coverage {', '.join(coverage)}")
            print(f"   pinball(median_folded) - pinball(true_values) = {difference.mean():+.4f}  "
                  f"95% bootstrap interval [{low:+.4f}, {high:+.4f}]")  # fmt: skip
            # 80% coverage by generator, to show that no single kind of series drives the result.
            for frame in FRAMES:
                per_kind = [
                    f"{kind} {inside_by_frame[frame][torch.as_tensor(kind_of == kind)].float().mean().item():.3f}"
                    for kind in KINDS
                ]
                print(f"   80% coverage by generator, {frame:14s}: {', '.join(per_kind)}")


if __name__ == "__main__":
    main()
