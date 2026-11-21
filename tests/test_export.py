"""Unit tests for checkpoint export (spec §7.4, §8)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from timestral_tune import (
    ConformalCalibrator,
    LoraConfig,
    TimestralTrainer,
    TrainingConfig,
    export_checkpoint,
)
from timestral_tune._vendor.timestral.loading import load_model as upstream_load_model
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS, LoRASliceLinear

from .helpers import make_toy_dataset


def test_export_writes_conformal_metadata_license_and_notice(tiny_forecaster, tmp_path):
    # Create a fake local base checkpoint dir with a custom LICENSE file to test verbatim copying.
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    license_text = "Mistral AI Modified MIT License — test sentinel\n"
    (base_dir / "LICENSE").write_text(license_text, encoding="utf-8")

    dataset = make_toy_dataset(tiny_forecaster.patch_size, series=4, seed=3, covariates=True)
    model = copy.deepcopy(tiny_forecaster.model)
    cfg = TrainingConfig(
        model_name_or_path=base_dir,
        mode="lora_variate",
        device="cpu",
        batch_size=len(dataset),
        max_steps=2,
        learning_rate=1e-2,
        output_dir=tmp_path / "out",
    )
    lora_cfg = LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "v"))
    trainer = TimestralTrainer(cfg, dataset, lora_config=lora_cfg, model=model)
    trainer.fit()

    # Fit a static CQR calibrator and export alongside the model.
    rng = np.random.default_rng(0)
    preds = np.linspace(-1.0, 1.0, len(DEFAULT_QUANTILE_LEVELS))[None, None, :] + rng.normal(size=(20, 8, 1))
    actuals = rng.normal(size=(20, 8))
    scale = np.ones(20)
    cal = ConformalCalibrator(target_coverage=0.80, bucket_size=4).fit(preds, actuals, scale)

    out = trainer.export(calibrator=cal, metrics={"val_loss": 0.123})
    assert out == tmp_path / "out"

    # LICENSE copied verbatim from local base checkpoint dir; NOTICE names base model & Mistral licence.
    assert (out / "LICENSE").read_text(encoding="utf-8") == license_text
    notice = (out / "NOTICE").read_text(encoding="utf-8")
    assert str(base_dir) in notice and "Modified MIT License" in notice and "adapter.safetensors" in notice

    # conformal.json reloads cleanly and matches original calibrator.
    reloaded_cal = ConformalCalibrator.load(out / "conformal.json")
    np.testing.assert_array_equal(cal.transform(preds, scale), reloaded_cal.transform(preds, scale))

    # adapter_config.json and adapter.safetensors round-trip cleanly.
    reloaded_lora = LoraConfig.from_dict(json.loads((out / "adapter_config.json").read_text(encoding="utf-8")))
    assert reloaded_lora == lora_cfg
    adapter_tensors = load_file(str(out / "adapter.safetensors"))
    assert adapter_tensors and all(".lora_A." in k or ".lora_B." in k for k in adapter_tensors)

    # tune_metadata.json holds metrics, data fingerprint, hyperparameters, and library/vendor versions.
    meta = json.loads((out / "tune_metadata.json").read_text(encoding="utf-8"))
    assert meta["metrics"] == {"val_loss": 0.123}
    assert meta["data_fingerprint"].startswith("sha256:")
    assert meta["has_conformal"] is True and meta["mode"] == "lora_variate"

    # Live model stayed unmerged after export.
    assert all(not m.merged for m in trainer.model.modules() if isinstance(m, LoRASliceLinear))


def test_export_requires_output_dir_and_rejects_missing_license_path(tiny_forecaster, tmp_path):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    trainer = TimestralTrainer(
        TrainingConfig(model_name_or_path="x", device="cpu", batch_size=4, max_steps=1),
        dataset,
        model=copy.deepcopy(tiny_forecaster.model),
    )
    with pytest.raises(ValueError, match="output_dir"):
        trainer.export()
    with pytest.raises(FileNotFoundError, match="license_path"):
        export_checkpoint(tiny_forecaster.model, tmp_path / "out", license_path=tmp_path / "does-not-exist")


def test_export_from_compiled_model_loads_strict(tiny_forecaster, tmp_path):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, series=4, covariates=True)
    model = copy.deepcopy(tiny_forecaster.model)
    cfg = TrainingConfig(
        model_name_or_path="local",
        mode="lora_variate",
        device="cpu",
        batch_size=len(dataset),
        max_steps=2,
        learning_rate=1e-2,
        compile_model=True,
        compile_mode="default",
    )
    trainer = TimestralTrainer(cfg, dataset, model=model)
    trainer.fit()
    out = trainer.export(tmp_path / "compiled-export")
    reloaded = upstream_load_model(out, device="cpu")
    assert all(torch.isfinite(p).all() for p in reloaded.parameters())
