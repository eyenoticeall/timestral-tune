"""Export round-trip (spec §9, T11)."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from timestral_tune import LoraConfig, TimestralTrainer, TrainingConfig
from timestral_tune._vendor.timestral import TimestralConfig, TimestralForecaster
from timestral_tune._vendor.timestral.loading import load_model as upstream_load_model
from timestral_tune.models import denormalise_quantiles, training_forward

from ..helpers import make_toy_dataset, max_rel, upstream_batch

EXPORT_RTOL = 1e-4


@pytest.mark.acceptance
@pytest.mark.parametrize("mode", ["head_only", "lora_variate", "lora_temporal", "full_ft"])
@pytest.mark.parametrize("scenario", ["covariates"], indirect=True)
def test_t11_export_round_trip(forecaster, raw, mode, tmp_path):
    """T11. Passes when: The exported directory loads in the vendored upstream loader with `strict=True`
    and reproduces the trainer's forecasts.
    """
    patch = forecaster.patch_size
    dataset = make_toy_dataset(patch, series=4, seed=11, covariates=True)
    model = copy.deepcopy(forecaster.model).cpu()
    lr = 1e-4 if mode == "full_ft" else 5e-3
    config = TrainingConfig(
        model_name_or_path="local-test-checkpoint",
        mode=mode,
        device="cpu",
        batch_size=len(dataset),
        max_steps=3,
        learning_rate=lr,
        lr_schedule="constant",
        shuffle=False,
        log_every=100,
    )
    lora_cfg = (
        None
        if mode in ("head_only", "full_ft")
        else LoraConfig(r=4, lora_alpha=8, axes=("variate" if mode == "lora_variate" else "time",))
    )
    trainer = TimestralTrainer(config, dataset, lora_config=lora_cfg, model=model)
    trainer.fit()

    out_dir = trainer.export(tmp_path / f"export-{mode}")

    # Required files exist; config.json has zero unknown keys and parses strictly in upstream TimestralConfig.
    for filename in ("config.json", "model.safetensors", "tune_metadata.json", "LICENSE", "NOTICE"):
        assert (out_dir / filename).is_file(), f"missing {filename}"
    if mode.startswith("lora_"):
        assert (out_dir / "adapter.safetensors").is_file()
        assert (out_dir / "adapter_config.json").is_file()
    else:
        assert not (out_dir / "adapter.safetensors").exists()
        assert not (out_dir / "adapter_config.json").exists()

    reloaded_config = TimestralConfig.load(out_dir / "config.json")
    assert reloaded_config == model.config
    metadata = json.loads((out_dir / "tune_metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == mode and metadata["vendored_upstream_commit"]

    # Vendored upstream loader loads with strict=True and reproduces the trainer's forecasts.
    reloaded_model = upstream_load_model(out_dir, device="cpu")
    reloaded_forecaster = TimestralForecaster(reloaded_model, quantile_real_cap_k=0)

    batch, _, _ = upstream_batch(forecaster, raw)
    trainer.model.eval()
    with torch.no_grad():
        trainer_out = training_forward(trainer.model, batch)
        trainer_raw = denormalise_quantiles(
            trainer.model, trainer_out.quantiles, trainer_out.anchor_loc, trainer_out.anchor_scale
        )
        reloaded_raw = reloaded_model.predict_block(
            batch.context,
            batch.context_mask,
            batch.group_ids,
            batch.horizon_steps // patch,
            batch.role_ids,
            batch.known_future,
            batch.known_future_mask,
        )
    assert max_rel(reloaded_raw, trainer_raw) < EXPORT_RTOL

    # Also verify full upstream `forecast()` runs cleanly on the exported directory.
    forecast = reloaded_forecaster.forecast(
        raw.context, raw.horizon, symmetric_averaging=False, clip_nonnegative=False, **raw.forecast_kwargs()
    ).quantiles
    assert torch.isfinite(forecast).all() and (forecast.diff(dim=-1) >= 0).all()

    # Exporting is non-destructive: the live trainer can keep training afterwards.
    res = trainer.fit()
    assert res.steps == 3
