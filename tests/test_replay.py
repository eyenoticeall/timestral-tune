"""Tests for Phase 4 replay primitives: ReplayConfig, SyntheticReplaySource, distillation_loss, disable_lora."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from timestral_tune.config import LoraConfig, ReplayConfig
from timestral_tune.data import DHFDataCollator, TimeSeriesDataset
from timestral_tune.models import TuneTimestralModel, disable_lora, inject_lora, merge_lora, training_forward
from timestral_tune.replay import ALL_SYNTHETIC_FAMILIES, SyntheticReplaySource, distillation_loss

from .helpers import make_tiny_forecaster


def _fresh_tiny_model(seed: int = 0) -> TuneTimestralModel:
    return make_tiny_forecaster(seed=seed).model


class TestReplayConfig:
    def test_defaults_and_roundtrip(self) -> None:
        cfg = ReplayConfig()
        assert cfg.replay_fraction == 0.25
        assert cfg.distill_weight == 0.0
        assert cfg.batch_size is None
        assert cfg.sign_flip is None
        assert cfg.cache_base_outputs is True
        assert cfg.resolved_batch_size(32) == 8
        assert cfg.resolved_batch_size(1) == 1
        assert ReplayConfig.from_dict(cfg.to_dict()) == cfg

    def test_distill_only_config(self) -> None:
        cfg = ReplayConfig(replay_fraction=0.0, distill_weight=0.5, batch_size=None)
        assert cfg.resolved_batch_size(32) == 8
        assert cfg.resolved_batch_size(2) == 1
        explicit = ReplayConfig(replay_fraction=0.0, distill_weight=0.5, batch_size=6, sign_flip=False)
        assert explicit.resolved_batch_size(32) == 6
        assert ReplayConfig.from_dict(explicit.to_dict()) == explicit

    def test_validation_rejects_invalid_values(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ReplayConfig(replay_fraction=0.0, distill_weight=0.0)
        for bad_frac in (-0.1, 1.0, 1.5, float("nan"), float("inf"), True):
            with pytest.raises(ValueError, match="replay_fraction"):
                ReplayConfig(replay_fraction=bad_frac, distill_weight=0.5)  # type: ignore[arg-type]
        for bad_dw in (-0.01, float("nan"), float("inf"), True):
            with pytest.raises(ValueError, match="distill_weight"):
                ReplayConfig(replay_fraction=0.2, distill_weight=bad_dw)  # type: ignore[arg-type]
        for bad_bs in (0, -4, True, 1.5):
            with pytest.raises(ValueError, match="batch_size"):
                ReplayConfig(replay_fraction=0.2, batch_size=bad_bs)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="sign_flip"):
            ReplayConfig(replay_fraction=0.2, sign_flip="yes")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cache_base_outputs"):
            ReplayConfig(replay_fraction=0.2, cache_base_outputs=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown ReplayConfig keys"):
            ReplayConfig.from_dict({"replay_fraction": 0.2, "bogus": 1})


class TestSyntheticReplaySource:
    def test_deterministic_and_diverse_across_families(self) -> None:
        tiny_model = _fresh_tiny_model()
        a = SyntheticReplaySource(num_series=15, context_length=32, forecast_horizon=12, seed=42)
        b = SyntheticReplaySource(num_series=15, context_length=32, forecast_horizon=12, seed=42)
        c = SyntheticReplaySource(num_series=15, context_length=32, forecast_horizon=12, seed=43)

        assert isinstance(a, TimeSeriesDataset)
        assert len(a) == len(b) == 15 * 4  # default series_length = ctx + 4 * h, stride = h
        assert set(a.series_families) == set(ALL_SYNTHETIC_FAMILIES)

        for sa, sb in zip(a.series, b.series, strict=True):
            assert np.array_equal(sa.targets, sb.targets)
            assert np.all(np.isfinite(sa.targets))
        assert not np.array_equal(a.series[0].targets, c.series[0].targets)

        # Collates and runs cleanly through training_forward.
        collator = DHFDataCollator.for_model(tiny_model.config, a.context_length)
        batch = collator([a[0], a[5], a[10]])
        out = training_forward(tiny_model, batch)
        assert torch.isfinite(out.quantiles).all()

    def test_missing_rate_preserves_valid_windows(self) -> None:
        ds = SyntheticReplaySource(num_series=10, context_length=64, forecast_horizon=32, seed=7, missing_rate=0.1)
        assert len(ds) > 0
        nan_frac = float(np.mean([np.isnan(s.targets).mean() for s in ds.series]))
        assert 0.03 < nan_frac < 0.20

    def test_validation_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="num_series"):
            SyntheticReplaySource(num_series=0)
        with pytest.raises(ValueError, match="series_length"):
            SyntheticReplaySource(context_length=64, forecast_horizon=32, series_length=80)
        with pytest.raises(ValueError, match="missing_rate"):
            SyntheticReplaySource(missing_rate=0.5)
        with pytest.raises(ValueError, match="unknown synthetic families"):
            SyntheticReplaySource(families=("not_a_family",))  # type: ignore[arg-type]


class TestDistillationLoss:
    def test_exact_masked_l1_and_teacher_detached(self) -> None:
        student = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[[0.0, 2.0, 5.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            requires_grad=True,
        )
        # Only (0, 0) and (0, 1) active; row 1 masked out completely.
        mask = torch.tensor([[True, True], [False, False]])
        loss = distillation_loss(student, teacher, mask)

        # (0,0): mean(|1|, |0|, |-2|) = 1.0; (0,1): mean(|3|, |4|, |5|) = 4.0 -> mean = 2.5
        assert loss.item() == pytest.approx(2.5, abs=1e-6)
        loss.backward()
        assert student.grad is not None
        assert torch.all(student.grad[1] == 0.0)
        assert torch.any(student.grad[0] != 0.0)
        assert teacher.grad is None

    def test_all_false_mask_returns_zero_with_zero_grad(self) -> None:
        student = torch.randn(2, 4, 21, requires_grad=True)
        teacher = torch.randn(2, 4, 21)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        loss = distillation_loss(student, teacher, mask)
        assert loss.item() == 0.0
        loss.backward()
        assert student.grad is not None
        assert torch.all(student.grad == 0.0)

    def test_unmasked_and_promotion_and_errors(self) -> None:
        s = torch.ones(2, 3, 5, dtype=torch.bfloat16, requires_grad=True)
        t = torch.zeros(2, 3, 5, dtype=torch.bfloat16)
        loss = distillation_loss(s, t)
        assert loss.dtype == torch.float32
        assert loss.item() == pytest.approx(1.0)

        with pytest.raises(ValueError, match="3-D"):
            distillation_loss(torch.ones(2, 3), torch.ones(2, 3))
        with pytest.raises(ValueError, match="does not match"):
            distillation_loss(torch.ones(2, 3, 5), torch.ones(2, 4, 5))
        with pytest.raises(ValueError, match="weight_mask"):
            distillation_loss(torch.ones(2, 3, 5), torch.ones(2, 3, 5), torch.ones(2, 4, dtype=torch.bool))
        with pytest.raises(ValueError, match="non-negative"):
            distillation_loss(
                torch.ones(2, 3, 5), torch.ones(2, 3, 5), torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
            )


class TestDisableLora:
    def test_bypass_matches_base_unmerged_and_merged_and_restores_on_error(self) -> None:
        tiny_model = _fresh_tiny_model()
        ds = SyntheticReplaySource(num_series=2, context_length=32, forecast_horizon=16, seed=0)
        batch = DHFDataCollator.for_model(tiny_model.config, 32)([ds[0], ds[1]])

        tiny_model.eval()
        with torch.no_grad():
            base_out = training_forward(tiny_model, batch).quantiles.clone()

        inject_lora(tiny_model, LoraConfig(r=4, lora_alpha=8.0, axes=("time",), targets=("q", "k", "v", "o")))
        # Make LoRA non-zero so adapted output diverges from base.
        with torch.no_grad():
            for name, param in tiny_model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.25)

        with torch.no_grad():
            adapted_out = training_forward(tiny_model, batch).quantiles.clone()
        assert not torch.allclose(adapted_out, base_out, atol=1e-4)

        # Unmerged bypass equals base identically.
        with torch.no_grad(), disable_lora(tiny_model):
            disabled_out = training_forward(tiny_model, batch).quantiles.clone()
        assert torch.equal(disabled_out, base_out)
        # Restored afterward.
        with torch.no_grad():
            assert torch.equal(training_forward(tiny_model, batch).quantiles, adapted_out)

        # Merged bypass also unmerges cleanly and re-merges on exit (even on exception).
        merge_lora(tiny_model)
        with torch.no_grad():
            pre_merged_out = training_forward(tiny_model, batch).quantiles.clone()
        with pytest.raises(RuntimeError, match="boom"):
            with torch.no_grad(), disable_lora(tiny_model):
                merged_disabled_out = training_forward(tiny_model, batch).quantiles.clone()
                assert torch.equal(merged_disabled_out, base_out)
                raise RuntimeError("boom")
        with torch.no_grad():
            post_merged_out = training_forward(tiny_model, batch).quantiles
            assert torch.equal(post_merged_out, pre_merged_out)
            assert torch.allclose(post_merged_out, adapted_out, atol=5e-4)
