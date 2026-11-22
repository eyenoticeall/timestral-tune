"""Tests for Phase 4 replay primitives: ReplayConfig, SyntheticReplaySource, distillation_loss, disable_lora."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from timestral_tune._vendor.timestral.forecaster import TimestralForecaster
from timestral_tune.config import LoraConfig, ReplayConfig, TrainingConfig
from timestral_tune.data import ColumnRoles, DHFDataCollator, SeriesData, TimeSeriesDataset, TrainingBatch
from timestral_tune.data.batch import pad_batch_to_series
from timestral_tune.models import TuneTimestralModel, disable_lora, inject_lora, merge_lora, training_forward
from timestral_tune.replay import ALL_SYNTHETIC_FAMILIES, SyntheticReplaySource, distillation_loss
from timestral_tune.trainer import TimestralTrainer
from timestral_tune.trainer import trainer as trainer_module

from .helpers import make_tiny_forecaster


def _fresh_tiny_model(seed: int = 0) -> TuneTimestralModel:
    return make_tiny_forecaster(seed=seed).model


class TestReplayConfig:
    def test_defaults_and_roundtrip(self) -> None:
        cfg = ReplayConfig()
        assert cfg.replay_fraction == 0.0
        assert cfg.distill_weight == 0.1
        assert cfg.batch_size is None
        assert cfg.sign_flip is None
        assert cfg.cache_base_outputs is True
        assert cfg.smooth_beta == 1e-3
        assert cfg.resolved_batch_size(32) == 8
        assert cfg.resolved_batch_size(1) == 1
        assert ReplayConfig.from_dict(cfg.to_dict()) == cfg

    def test_distill_only_and_mix_configs(self) -> None:
        cfg = ReplayConfig(replay_fraction=0.25, distill_weight=0.0, batch_size=None)
        assert cfg.resolved_batch_size(32) == 8
        assert cfg.resolved_batch_size(2) == 1
        explicit = ReplayConfig(replay_fraction=0.0, distill_weight=0.5, batch_size=6, sign_flip=False, smooth_beta=0.0)
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
        for bad_beta in (-1e-3, float("nan"), float("inf"), True):
            with pytest.raises(ValueError, match="smooth_beta"):
                ReplayConfig(distill_weight=0.2, smooth_beta=bad_beta)  # type: ignore[arg-type]
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
        # (0,0): mean(|1|, |0|, |-2|) = 1.0; (0,1): mean(|3|, |4|, |5|) = 4.0 -> mean = 2.5
        loss_l1 = distillation_loss(student, teacher, mask, smooth_beta=0.0)
        assert loss_l1.item() == pytest.approx(2.5, abs=1e-6)
        # Default smooth_beta=1e-3 shifts each |diff|>=1e-3 element by -0.5*beta (5 of 6 entries non-zero).
        loss = distillation_loss(student, teacher, mask)
        assert loss.item() == pytest.approx(2.5 - (5.0 / 6.0) * 0.5e-3, abs=1e-6)
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
        loss = distillation_loss(s, t, smooth_beta=0.0)
        assert loss.dtype == torch.float32
        assert loss.item() == pytest.approx(1.0)
        # Near-zero round-off (1e-7) produces near-zero gradient under default smooth_beta=1e-3 (no +/-1 kink).
        s_near = torch.full((1, 2, 5), 1e-7, requires_grad=True)
        distillation_loss(s_near, torch.zeros_like(s_near)).backward()
        assert s_near.grad is not None and float(s_near.grad.abs().max()) < 1e-3

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
            assert torch.allclose(post_merged_out, adapted_out, atol=5e-3)


def _make_covariate_dataset(
    *,
    num_series: int = 4,
    length: int = 80,
    context_length: int = 32,
    forecast_horizon: int = 10,
    seed: int = 10,
    shift: float = 0.0,
) -> TimeSeriesDataset:
    rng = np.random.default_rng(seed)
    ts = np.arange(length, dtype=np.int64)
    empty_cat = np.empty((0, length), dtype=np.float64)
    series: list[SeriesData] = []
    for i in range(num_series):
        t = np.linspace(0.0, 4.0 * np.pi, length, dtype=np.float32)
        past = np.cos(t + i)[None, :].astype(np.float32)
        known = np.sin(2.0 * t + i)[None, :].astype(np.float32)
        y = (shift + 2.0 * known[0] + 0.5 * past[0] + rng.normal(0.0, 0.1, size=length)).astype(np.float32)
        series.append(
            SeriesData(
                series_id=f"dom_{i}",
                timestamps=ts,
                targets=y[None, :],
                past=past,
                known=known,
                categorical_past=empty_cat,
                categorical_known=empty_cat,
            )
        )
    return TimeSeriesDataset(
        series,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=forecast_horizon,
        min_context=context_length,
        roles=ColumnRoles(targets=("y",), past=("p0",), known=("k0",)),
    )


class TestTrainerReplayAndDistillation:
    def test_constructor_validation_and_default_synthetic_source(self) -> None:
        train_ds = _make_covariate_dataset()
        cfg = TrainingConfig(model_name_or_path="tiny", mode="head_only", batch_size=4, max_steps=2, device="cpu")
        with pytest.raises(ValueError, match="replay_dataset requires a ReplayConfig"):
            TimestralTrainer(cfg, train_ds, replay_dataset=train_ds, model=_fresh_tiny_model())
        with pytest.raises(TypeError, match="replay_config"):
            TimestralTrainer(cfg, train_ds, replay_config={"replay_fraction": 0.2}, model=_fresh_tiny_model())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="replay_dataset"):
            TimestralTrainer(
                cfg, train_ds, replay_config=ReplayConfig(), replay_dataset="bad", model=_fresh_tiny_model()
            )  # type: ignore[arg-type]

        trainer = TimestralTrainer(
            cfg, train_ds, replay_config=ReplayConfig(replay_fraction=0.2), model=_fresh_tiny_model()
        )
        assert isinstance(trainer.replay_dataset, SyntheticReplaySource)
        assert trainer.replay_dataset.context_length == train_ds.context_length
        assert trainer.replay_dataset.forecast_horizon == train_ds.forecast_horizon

    @pytest.mark.parametrize("mode", ["head_only", "lora_variate", "lora_temporal", "full_ft"])
    def test_all_modes_heterogeneous_layout_and_export(self, mode: str, tmp_path: Path) -> None:
        # Domain dataset has 3 rows/series (1 target + 1 past + 1 known), horizon=10 (non-patch multiple).
        # Replay dataset is univariate (1 row/series), horizon=6 (different layout AND different horizon).
        train_ds = _make_covariate_dataset(num_series=3, length=64, context_length=32, forecast_horizon=10, seed=1)
        val_ds = _make_covariate_dataset(num_series=2, length=64, context_length=32, forecast_horizon=10, seed=2)
        replay_ds = SyntheticReplaySource(num_series=5, context_length=24, forecast_horizon=6, seed=3)
        replay_cfg = ReplayConfig(replay_fraction=0.25, distill_weight=0.5, batch_size=3)

        cfg = TrainingConfig(
            model_name_or_path="tiny",
            mode=mode,  # type: ignore[arg-type]
            learning_rate=1e-3,
            batch_size=4,
            max_steps=4,
            device="cpu",
            seed=7,
            sign_flip=True,
        )
        trainer = TimestralTrainer(
            cfg,
            train_ds,
            val_ds,
            replay_config=replay_cfg,
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=11),
        )
        res = trainer.fit()
        assert res.steps == 4
        assert res.final_train_loss is not None and np.isfinite(res.final_train_loss)
        assert res.best_eval_loss is not None and np.isfinite(res.best_eval_loss)

        out_dir = trainer.export(tmp_path / f"exp_{mode}")
        meta = json.loads((out_dir / "tune_metadata.json").read_text())
        assert meta["replay_config"] == replay_cfg.to_dict()
        assert meta["replay_source"]["type"] == "SyntheticReplaySource"
        assert meta["replay_source"]["seed"] == 3
        assert meta["replay_source"]["data_fingerprint"].startswith("sha256:")
        # Official upstream loader still loads config.json + model.safetensors with strict=True.
        loaded = TimestralForecaster.from_pretrained(out_dir, device="cpu")
        assert loaded.model.config.d_model == trainer.model.config.d_model

    @pytest.mark.parametrize("mode", ["lora_temporal", "full_ft"])
    @pytest.mark.parametrize("sign_flip", [False, True])
    def test_cached_vs_live_teacher_exact_parity(self, mode: str, sign_flip: bool) -> None:
        train_ds = _make_covariate_dataset(num_series=3, length=64, context_length=32, forecast_horizon=8, seed=4)
        replay_ds = SyntheticReplaySource(num_series=4, context_length=32, forecast_horizon=6, seed=5)
        lr = 1e-3 if mode == "lora_temporal" else 3e-4
        cfg = TrainingConfig(
            model_name_or_path="tiny",
            mode=mode,  # type: ignore[arg-type]
            learning_rate=lr,
            batch_size=3,
            max_steps=5,
            device="cpu",
            seed=19,
            sign_flip=sign_flip,
        )
        t_cached = TimestralTrainer(
            cfg,
            train_ds,
            replay_config=ReplayConfig(
                replay_fraction=0.25, distill_weight=0.75, batch_size=3, cache_base_outputs=True
            ),
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=23),
        )
        t_live = TimestralTrainer(
            cfg,
            train_ds,
            replay_config=ReplayConfig(
                replay_fraction=0.25, distill_weight=0.75, batch_size=3, cache_base_outputs=False
            ),
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=23),
        )
        assert t_cached._replay_base_pos is not None
        assert (t_cached._replay_base_neg is not None) == sign_flip
        assert t_live._replay_base_pos is None
        assert (t_live._frozen_base is not None) == (mode == "full_ft")

        # Direct teacher-output parity on both +x and -x orientations (< 1e-5).
        collator = DHFDataCollator.for_model(t_cached.model.config, replay_ds.context_length)
        probe = collator([replay_ds[2], replay_ds[0], replay_ds[3]])
        with torch.no_grad(), disable_lora(t_live.model):
            ref_pos = training_forward(t_live.model, probe).quantiles
        assert torch.allclose(t_cached._replay_base_pos[[2, 0, 3]].flatten(0, 1), ref_pos, atol=1e-5)
        if sign_flip:
            assert t_cached._replay_base_neg is not None
            probe_neg = trainer_module._apply_series_flip(probe, torch.tensor([True, True, True]), 1)
            with torch.no_grad(), disable_lora(t_live.model):
                ref_neg = training_forward(t_live.model, probe_neg).quantiles
            assert torch.allclose(t_cached._replay_base_neg[[2, 0, 3]].flatten(0, 1), ref_neg, atol=1e-5)

        r_cached = t_cached.fit()
        r_live = t_live.fit()
        for (_, lc), (_, ll) in zip(r_cached.train_history, r_live.train_history, strict=True):
            assert lc == pytest.approx(ll, abs=5e-4)
        num = sum(
            float((pc.detach() - pl.detach()).pow(2).sum())
            for pc, pl in zip(t_cached._trainable, t_live._trainable, strict=True)
        )
        den = sum(float(pl.detach().pow(2).sum()) for pl in t_live._trainable)
        assert (num**0.5) / max(den**0.5, 1e-12) < 5e-4

    def test_head_only_cached_vs_streaming_parity_with_replay_and_distill(self) -> None:
        train_ds = _make_covariate_dataset(num_series=3, length=64, context_length=32, forecast_horizon=8, seed=6)
        replay_ds = SyntheticReplaySource(num_series=4, context_length=32, forecast_horizon=6, seed=7)
        replay_cfg = ReplayConfig(replay_fraction=0.3, distill_weight=0.5, batch_size=3)

        t_cached = TimestralTrainer(
            TrainingConfig(
                model_name_or_path="tiny",
                mode="head_only",
                learning_rate=3e-3,
                batch_size=3,
                max_steps=6,
                device="cpu",
                seed=29,
                sign_flip=True,
                cache_frozen_features=True,
            ),
            train_ds,
            replay_config=replay_cfg,
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=31),
        )
        t_stream = TimestralTrainer(
            TrainingConfig(
                model_name_or_path="tiny",
                mode="head_only",
                learning_rate=3e-3,
                batch_size=3,
                max_steps=6,
                device="cpu",
                seed=29,
                sign_flip=True,
                cache_frozen_features=False,
            ),
            train_ds,
            replay_config=replay_cfg,
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=31),
        )
        r_cached = t_cached.fit()
        r_stream = t_stream.fit()
        assert t_cached._replay_head_cache is not None and t_cached._replay_head_cache.negated is not None
        assert t_stream._replay_head_cache is None
        for (_, lc), (_, ls) in zip(r_cached.train_history, r_stream.train_history, strict=True):
            assert lc == pytest.approx(ls, abs=1e-4)
        for pc, ps in zip(t_cached._trainable, t_stream._trainable, strict=True):
            assert torch.allclose(pc, ps, atol=1e-4)

    def test_teacher_cache_cap_accounts_for_dual_sign_flip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        train_ds = _make_covariate_dataset(num_series=2, length=64, context_length=32, forecast_horizon=8, seed=8)
        replay_ds = SyntheticReplaySource(num_series=4, context_length=32, forecast_horizon=8, seed=9)
        model = _fresh_tiny_model(seed=1)
        single_bytes = len(replay_ds) * 1 * 8 * model.config.num_quantiles * 4
        monkeypatch.setattr(trainer_module, "_MAX_HEAD_CACHE_BYTES", int(single_bytes * 1.5))

        t_no_flip = TimestralTrainer(
            TrainingConfig(model_name_or_path="tiny", mode="full_ft", batch_size=2, max_steps=1, device="cpu"),
            train_ds,
            replay_config=ReplayConfig(replay_fraction=0.0, distill_weight=0.5, sign_flip=False),
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=1),
        )
        t_flip = TimestralTrainer(
            TrainingConfig(model_name_or_path="tiny", mode="full_ft", batch_size=2, max_steps=1, device="cpu"),
            train_ds,
            replay_config=ReplayConfig(replay_fraction=0.0, distill_weight=0.5, sign_flip=True),
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=1),
        )
        assert t_no_flip._replay_base_pos is not None and t_no_flip._frozen_base is None
        assert t_flip._replay_base_pos is None and t_flip._frozen_base is not None

    def test_step0_distill_zero_label_free_and_padding_excluded(self) -> None:
        train_ds = _make_covariate_dataset(num_series=2, length=64, context_length=32, forecast_horizon=8, seed=10)
        # forecast_horizon=6 pads to 8 (patch_size=4) so steps [6, 8) are right-padding.
        replay_ds = SyntheticReplaySource(num_series=4, context_length=32, forecast_horizon=6, seed=11)
        cfg = TrainingConfig(
            model_name_or_path="tiny",
            mode="full_ft",
            learning_rate=1e-2,
            batch_size=2,
            max_steps=1,
            device="cpu",
            sign_flip=False,
        )
        trainer = TimestralTrainer(
            cfg,
            train_ds,
            replay_config=ReplayConfig(replay_fraction=0.0, distill_weight=1.0, cache_base_outputs=False),
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=12),
        )
        collator = DHFDataCollator.for_model(trainer.model.config, replay_ds.context_length)
        raw_batch = collator([replay_ds[0], replay_ds[1]])
        # Pad with 2 dummy series (4 total, 2 real) AND zero all target labels (label-free distillation!).
        padded = pad_batch_to_series(raw_batch, 4)
        unlabelled = TrainingBatch(
            context=padded.context,
            context_mask=padded.context_mask,
            group_ids=padded.group_ids,
            role_ids=padded.role_ids,
            known_future=padded.known_future,
            known_future_mask=padded.known_future_mask,
            targets=torch.zeros_like(padded.targets),
            loss_weight=torch.zeros_like(padded.loss_weight),
        )

        # At step 0, student == frozen base so distillation loss is identically 0.
        r_pinball, r_distill = trainer._compute_replay_terms(
            unlabelled, real_series=2, teacher_quantiles=None, validate=False
        )
        assert r_pinball is None
        assert r_distill is not None and r_distill.item() == pytest.approx(0.0, abs=1e-6)

        # Perturb student weights: now unlabelled batch yields positive distill loss & finite non-zero gradients!
        with torch.no_grad():
            for p in trainer.model.head.parameters():
                p.add_(0.1)
        _, d_real = trainer._compute_replay_terms(unlabelled, real_series=2, teacher_quantiles=None, validate=False)
        assert d_real is not None and d_real.item() > 1e-3
        trainer.optimizer.zero_grad(set_to_none=True)
        d_real.backward()
        assert any(p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in trainer.model.head.parameters())

        # Mutating teacher_quantiles strictly on right-padding steps [6, 8) must change distill loss by 0.0.
        with torch.no_grad():
            base_q = training_forward(trainer._frozen_base, unlabelled, validate=False).quantiles[:2]  # type: ignore[arg-type]
        mutated_q = base_q.clone()
        mutated_q[:, 6:, :] += 100.0
        _, d_clean = trainer._compute_replay_terms(unlabelled, real_series=2, teacher_quantiles=base_q, validate=False)
        _, d_mut = trainer._compute_replay_terms(unlabelled, real_series=2, teacher_quantiles=mutated_q, validate=False)
        assert d_clean is not None and d_mut is not None
        assert d_mut.item() == pytest.approx(d_clean.item(), abs=1e-7)

    def test_replay_and_distillation_reduce_drift_and_preserve_eval_isolation(self) -> None:
        # Shifted domain dataset vs general replay dataset; aggressive head_only LR causes drift without replay.
        domain_ds = _make_covariate_dataset(
            num_series=4, length=64, context_length=32, forecast_horizon=8, seed=20, shift=50.0
        )
        val_ds = _make_covariate_dataset(
            num_series=2, length=64, context_length=32, forecast_horizon=8, seed=21, shift=50.0
        )
        general_ds = SyntheticReplaySource(num_series=16, context_length=32, forecast_horizon=8, seed=22)

        def run(replay_cfg: ReplayConfig | None) -> tuple[TimestralTrainer, float, float, float]:
            base_model = _fresh_tiny_model(seed=99)
            init_head = [p.detach().clone() for p in base_model.head.parameters()]
            collator = DHFDataCollator.for_model(base_model.config, general_ds.context_length)
            gen_batch = collator([general_ds[i] for i in range(8)])
            base_model.eval()
            with torch.no_grad():
                base_gen_q = training_forward(base_model, gen_batch).quantiles.clone()
            t = TimestralTrainer(
                TrainingConfig(
                    model_name_or_path="tiny",
                    mode="head_only",
                    learning_rate=5e-2,
                    batch_size=4,
                    max_steps=12,
                    device="cpu",
                    seed=0,
                    sign_flip=False,
                    restore_best=False,
                ),
                domain_ds,
                val_ds,
                replay_config=replay_cfg,
                replay_dataset=general_ds if replay_cfg is not None else None,
                model=base_model,
            )
            t.fit()
            drift = sum(
                float((p.detach() - p0).norm()) for p, p0 in zip(t.model.head.parameters(), init_head, strict=True)
            )
            with torch.no_grad():
                tuned_gen_q = training_forward(t.model, gen_batch).quantiles
            pred_drift = float((tuned_gen_q - base_gen_q).abs().mean())
            gen_loss = t.evaluate(general_ds).loss
            return t, drift, pred_drift, gen_loss

        t_plain, drift_plain, pred_drift_plain, gen_plain = run(None)
        t_distill, drift_distill, pred_drift_distill, _ = run(
            ReplayConfig(replay_fraction=0.0, distill_weight=5.0, batch_size=4, sign_flip=False)
        )
        t_mix, _, _, gen_mix = run(ReplayConfig(replay_fraction=0.5, distill_weight=0.0, batch_size=4, sign_flip=False))

        # Distillation keeps weights and general-domain predictions strictly closer to the frozen base.
        assert drift_distill < drift_plain
        assert pred_drift_distill < pred_drift_plain
        # Replay mixing lowers pinball loss on the replay distribution relative to domain-only training.
        assert gen_mix < gen_plain

        # Evaluation isolation: evaluate(val_ds) depends only on model weights, never on replay_config.
        t_ref = TimestralTrainer(
            TrainingConfig(model_name_or_path="tiny", mode="head_only", batch_size=4, device="cpu", sign_flip=False),
            domain_ds,
            val_ds,
            model=TuneTimestralModel.from_upstream(t_distill.model),
        )
        assert t_distill.evaluate(val_ds).loss == pytest.approx(t_ref.evaluate(val_ds).loss, abs=1e-6)
        assert t_mix.evaluate(val_ds).loss != pytest.approx(t_plain.evaluate(val_ds).loss, abs=1e-5)

    def test_replay_cycling_is_deterministic_across_short_replay_dataset(self) -> None:
        train_ds = _make_covariate_dataset(num_series=4, length=64, context_length=32, forecast_horizon=8, seed=30)
        # Only 2 replay windows while running 6 steps -> cycles 3+ times deterministically.
        tiny_replay = SyntheticReplaySource(
            num_series=2, context_length=32, forecast_horizon=8, series_length=40, seed=31
        )
        assert len(tiny_replay) == 2

        def run_once() -> tuple[list[tuple[int, float]], list[Tensor]]:
            t = TimestralTrainer(
                TrainingConfig(
                    model_name_or_path="tiny",
                    mode="lora_temporal",
                    learning_rate=1e-3,
                    batch_size=2,
                    max_steps=6,
                    device="cpu",
                    seed=42,
                    sign_flip=True,
                ),
                train_ds,
                replay_config=ReplayConfig(replay_fraction=0.25, distill_weight=0.5, batch_size=2),
                replay_dataset=tiny_replay,
                model=_fresh_tiny_model(seed=42),
            )
            res = t.fit()
            return res.train_history, [p.detach().clone() for p in t._trainable]

        h1, w1 = run_once()
        h2, w2 = run_once()
        assert h1 == h2
        for a, b in zip(w1, w2, strict=True):
            assert torch.equal(a, b)

    def test_val_dataset_rejected_as_replay_and_shared_train_slot_isolated(self) -> None:
        train_ds = _make_covariate_dataset(num_series=3, length=64, context_length=32, forecast_horizon=8, seed=40)
        val_ds = _make_covariate_dataset(num_series=2, length=64, context_length=32, forecast_horizon=8, seed=41)
        cfg = TrainingConfig(
            model_name_or_path="tiny", mode="head_only", batch_size=2, max_steps=2, device="cpu", sign_flip=True
        )
        with pytest.raises(ValueError, match="val_dataset"):
            TimestralTrainer(
                cfg, train_ds, val_ds, replay_config=ReplayConfig(), replay_dataset=val_ds, model=_fresh_tiny_model()
            )

        # Passing the same dataset object for train and replay in head_only must respect ReplayConfig.sign_flip=False.
        trainer = TimestralTrainer(
            cfg,
            train_ds,
            val_ds,
            replay_config=ReplayConfig(replay_fraction=0.2, distill_weight=0.5, sign_flip=False, batch_size=2),
            replay_dataset=train_ds,
            model=_fresh_tiny_model(),
        )
        trainer.fit()
        assert trainer._train_head_cache is not None and trainer._train_head_cache.negated is not None
        assert trainer._replay_head_cache is not None and trainer._replay_head_cache.negated is None
        assert trainer._replay_head_cache is not trainer._train_head_cache

    def test_smooth_beta_reaches_distillation_loss_and_custom_replay_export(self, tmp_path: Path) -> None:
        train_ds = _make_covariate_dataset(num_series=2, length=64, context_length=32, forecast_horizon=8, seed=42)
        custom_replay = _make_covariate_dataset(num_series=2, length=64, context_length=32, forecast_horizon=8, seed=43)
        cfg = TrainingConfig(
            model_name_or_path="tiny", mode="full_ft", batch_size=2, max_steps=1, device="cpu", sign_flip=False
        )
        t_l1 = TimestralTrainer(
            cfg,
            train_ds,
            replay_config=ReplayConfig(
                replay_fraction=0.0, distill_weight=1.0, smooth_beta=0.0, cache_base_outputs=False
            ),
            replay_dataset=custom_replay,
            model=_fresh_tiny_model(seed=44),
        )
        t_huber = TimestralTrainer(
            cfg,
            train_ds,
            replay_config=ReplayConfig(
                replay_fraction=0.0, distill_weight=1.0, smooth_beta=1.0, cache_base_outputs=False
            ),
            replay_dataset=custom_replay,
            model=_fresh_tiny_model(seed=44),
        )
        with torch.no_grad():
            for p1, p2 in zip(t_l1.model.head.parameters(), t_huber.model.head.parameters(), strict=True):
                p1.add_(0.05)
                p2.add_(0.05)
        batch = DHFDataCollator.for_model(t_l1.model.config, custom_replay.context_length)([custom_replay[0]])
        _, d_l1 = t_l1._compute_replay_terms(batch, real_series=1, teacher_quantiles=None, validate=False)
        _, d_huber = t_huber._compute_replay_terms(batch, real_series=1, teacher_quantiles=None, validate=False)
        assert d_l1 is not None and d_huber is not None
        assert d_l1.item() > d_huber.item() * 2.0  # large smooth_beta quadratically shrinks small errors

        out_dir = t_l1.export(tmp_path / "custom_replay_exp")
        meta = json.loads((out_dir / "tune_metadata.json").read_text())
        assert meta["replay_config"]["smooth_beta"] == 0.0
        assert meta["replay_source"]["type"] == "TimeSeriesDataset"
        assert "seed" not in meta["replay_source"]
        assert meta["replay_source"]["data_fingerprint"].startswith("sha256:")

    @pytest.mark.parametrize("mode", ["head_only", "lora_variate", "lora_temporal", "full_ft"])
    def test_multi_target_covariate_replay_cached_vs_live_parity(self, mode: str) -> None:
        # Multi-target (2 targets + 1 past + 1 known = 4 rows/series) replay dataset.
        rng = np.random.default_rng(50)
        length = 64
        ts = np.arange(length, dtype=np.int64)
        empty_cat = np.empty((0, length), dtype=np.float64)
        mt_series = []
        for i in range(3):
            t = np.linspace(0.0, 3.0 * np.pi, length, dtype=np.float32)
            past = np.cos(t + i)[None, :].astype(np.float32)
            known = np.sin(t + i)[None, :].astype(np.float32)
            y = np.stack(
                [np.sin(t + i) + rng.normal(0, 0.1, length), np.cos(2 * t + i) + rng.normal(0, 0.1, length)]
            ).astype(np.float32)
            mt_series.append(
                SeriesData(
                    series_id=f"mt_{i}",
                    timestamps=ts,
                    targets=y,
                    past=past,
                    known=known,
                    categorical_past=empty_cat,
                    categorical_known=empty_cat,
                )
            )
        replay_mt = TimeSeriesDataset(
            mt_series,
            context_length=32,
            forecast_horizon=6,
            stride=6,
            min_context=32,
            roles=ColumnRoles(targets=("y0", "y1"), past=("p0",), known=("k0",)),
        )
        train_ds = _make_covariate_dataset(num_series=3, length=64, context_length=32, forecast_horizon=8, seed=51)
        lr = 3e-4 if mode == "full_ft" else 1e-3
        cfg = TrainingConfig(
            model_name_or_path="tiny",
            mode=mode,  # type: ignore[arg-type]
            learning_rate=lr,
            batch_size=3,
            max_steps=4,
            device="cpu",
            seed=52,
            sign_flip=True,
        )
        t_cached = TimestralTrainer(
            cfg,
            train_ds,
            replay_config=ReplayConfig(replay_fraction=0.25, distill_weight=0.5, batch_size=2, cache_base_outputs=True),
            replay_dataset=replay_mt,
            model=_fresh_tiny_model(seed=53),
        )
        t_live = TimestralTrainer(
            TrainingConfig(
                model_name_or_path="tiny",
                mode=mode,  # type: ignore[arg-type]
                learning_rate=lr,
                batch_size=3,
                max_steps=4,
                device="cpu",
                seed=52,
                sign_flip=True,
                cache_frozen_features=False,
            ),
            train_ds,
            replay_config=ReplayConfig(
                replay_fraction=0.25, distill_weight=0.5, batch_size=2, cache_base_outputs=False
            ),
            replay_dataset=replay_mt,
            model=_fresh_tiny_model(seed=53),
        )
        rc, rl = t_cached.fit(), t_live.fit()
        for (_, lc), (_, ll) in zip(rc.train_history, rl.train_history, strict=True):
            assert lc == pytest.approx(ll, abs=5e-4)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="requires bf16 CUDA"
    )
    @pytest.mark.parametrize("mode", ["head_only", "lora_variate", "lora_temporal", "full_ft"])
    @pytest.mark.parametrize("mixed_precision", ["fp32", "bf16"])
    def test_cuda_replay_and_distillation_all_modes(self, mode: str, mixed_precision: str) -> None:
        train_ds = _make_covariate_dataset(num_series=3, length=64, context_length=32, forecast_horizon=8, seed=60)
        replay_ds = SyntheticReplaySource(num_series=4, context_length=32, forecast_horizon=6, seed=61)
        lr = 3e-4 if mode == "full_ft" else 1e-3
        cfg = TrainingConfig(
            model_name_or_path="tiny",
            mode=mode,  # type: ignore[arg-type]
            learning_rate=lr,
            batch_size=3,
            max_steps=4,
            mixed_precision=mixed_precision,  # type: ignore[arg-type]
            device="cuda",
            seed=62,
            sign_flip=True,
        )
        t_cached = TimestralTrainer(
            cfg,
            train_ds,
            replay_config=ReplayConfig(replay_fraction=0.25, distill_weight=0.5, batch_size=2, cache_base_outputs=True),
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=63),
        )
        t_live = TimestralTrainer(
            cfg,
            train_ds,
            replay_config=ReplayConfig(
                replay_fraction=0.25, distill_weight=0.5, batch_size=2, cache_base_outputs=False
            ),
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=63),
        )
        rc = t_cached.fit()
        rl = t_live.fit()
        assert rc.steps == 4 and rc.final_train_loss is not None and np.isfinite(rc.final_train_loss)
        assert rl.steps == 4 and rl.final_train_loss is not None and np.isfinite(rl.final_train_loss)
        tol = 5e-4 if mixed_precision == "fp32" else 2e-2
        for (_, lc), (_, ll) in zip(rc.train_history, rl.train_history, strict=True):
            assert lc == pytest.approx(ll, abs=tol)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="requires bf16 CUDA"
    )
    def test_cuda_bf16_replay_with_reduce_overhead_compile(self) -> None:
        # Non-divisible train (5 windows, B=4) and replay (3 windows, B=2) exercise bucket_to_max on both loaders.
        train_ds = _make_covariate_dataset(num_series=5, length=48, context_length=32, forecast_horizon=8, seed=70)
        replay_ds = SyntheticReplaySource(
            num_series=3, context_length=32, forecast_horizon=6, series_length=38, seed=71
        )
        cfg = TrainingConfig(
            model_name_or_path="tiny",
            mode="lora_temporal",
            learning_rate=1e-3,
            batch_size=4,
            max_steps=4,
            mixed_precision="bf16",
            device="cuda",
            compile_model=True,
            compile_mode="reduce-overhead",
            seed=72,
            sign_flip=True,
        )
        trainer = TimestralTrainer(
            cfg,
            train_ds,
            replay_config=ReplayConfig(replay_fraction=0.25, distill_weight=0.5, batch_size=2),
            replay_dataset=replay_ds,
            model=_fresh_tiny_model(seed=73),
        )
        res = trainer.fit()
        assert res.steps == 4 and res.final_train_loss is not None and np.isfinite(res.final_train_loss)
