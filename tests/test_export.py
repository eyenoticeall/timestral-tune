"""Unit tests for checkpoint export (spec §7.4, §8)."""

from __future__ import annotations

import copy
import json
import shutil

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and shutil.which("g++") is None and shutil.which("clang++") is None:
        pytest.skip("CPU torch.compile requires a C++ compiler (g++ or clang++)")
    dataset = make_toy_dataset(tiny_forecaster.patch_size, series=4, covariates=True)
    model = copy.deepcopy(tiny_forecaster.model)
    cfg = TrainingConfig(
        model_name_or_path="local",
        mode="lora_variate",
        device=device,
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


def test_export_content_fingerprint_dirty_dir_cleanup_and_guards(tiny_forecaster, tmp_path, caplog):
    from timestral_tune.export import dataset_fingerprint

    # dataset_fingerprint hashes data content, not just shapes/lengths.
    ds_a = make_toy_dataset(tiny_forecaster.patch_size, series=4, seed=1, covariates=True)
    ds_b = make_toy_dataset(tiny_forecaster.patch_size, series=4, seed=2, covariates=True)
    assert len(ds_a) == len(ds_b) and dataset_fingerprint(ds_a) != dataset_fingerprint(ds_b)

    # Unfitted CQR calibrator is rejected at export.
    with pytest.raises(ValueError, match="unfitted"):
        export_checkpoint(tiny_forecaster.model, tmp_path / "unfitted", calibrator=ConformalCalibrator(0.80))

    # Re-exporting a non-LoRA, non-conformal model into a dirty directory removes stale conditional files.
    out_dir = tmp_path / "reused"
    lora_model = copy.deepcopy(tiny_forecaster.model)
    cfg = TrainingConfig(
        model_name_or_path="local-test",
        mode="lora_variate",
        device="cpu",
        batch_size=len(ds_a),
        max_steps=1,
        learning_rate=1e-2,
    )
    rng = np.random.default_rng(0)
    preds = np.linspace(-1.0, 1.0, len(DEFAULT_QUANTILE_LEVELS))[None, None, :] + rng.normal(size=(10, 8, 1))
    cal = ConformalCalibrator(0.80, bucket_size=4).fit(preds, rng.normal(size=(10, 8)), np.ones(10))
    TimestralTrainer(cfg, ds_a, model=lora_model).export(out_dir, calibrator=cal)
    assert (out_dir / "adapter.safetensors").is_file() and (out_dir / "conformal.json").is_file()

    with caplog.at_level("WARNING", logger="timestral_tune.export"):
        export_checkpoint(tiny_forecaster.model, out_dir, base_repo_or_path="local-missing")
    assert not (out_dir / "adapter.safetensors").exists()
    assert not (out_dir / "adapter_config.json").exists()
    assert not (out_dir / "conformal.json").exists()
    assert any("no verbatim LICENSE" in record.getMessage() for record in caplog.records)
    notice = (out_dir / "NOTICE").read_text(encoding="utf-8")
    assert "no verbatim LICENSE was found" in notice


def test_export_fingerprint_splits_mismatched_lora_atomic_metrics_and_chained_license(
    tiny_forecaster, tmp_path, caplog
):
    from timestral_tune.export import dataset_fingerprint
    from timestral_tune.models import inject_lora

    # 1. Train and validation splits sharing `series` with identical window counts get distinct fingerprints.
    full = make_toy_dataset(tiny_forecaster.patch_size, series=4, seed=3, covariates=True)
    train, val = full.split_by_time(12 * tiny_forecaster.patch_size)
    # Match window counts explicitly by slicing if needed.
    n = min(len(train), len(val))
    keep_t = np.zeros(len(train), dtype=bool)
    keep_t[:n] = True
    keep_v = np.zeros(len(val), dtype=bool)
    keep_v[:n] = True
    train_eq = train._with_windows(keep_t)
    val_eq = val._with_windows(keep_v)
    assert len(train_eq) == len(val_eq) and dataset_fingerprint(train_eq) != dataset_fingerprint(val_eq)

    # Also verify `_series_index` (not just `_origins`) enters the fingerprint: two 1-window subsets
    # sharing the exact same origin on different series must hash distinctly.
    idx_s0 = next(i for i in range(len(full)) if full._series_index[i] == 0)
    origin_s0 = full._origins[idx_s0]
    idx_s1 = next(i for i in range(len(full)) if full._series_index[i] == 1 and full._origins[i] == origin_s0)
    m0 = np.zeros(len(full), dtype=bool)
    m0[idx_s0] = True
    m1 = np.zeros(len(full), dtype=bool)
    m1[idx_s1] = True
    ds_s0, ds_s1 = full._with_windows(m0), full._with_windows(m1)
    assert ds_s0._origins.tolist() == ds_s1._origins.tolist()
    assert dataset_fingerprint(ds_s0) != dataset_fingerprint(ds_s1)

    # 2. Mismatched lora_config against model._lora_config raises ValueError before writing anything.
    adapted = copy.deepcopy(tiny_forecaster.model)
    inject_lora(adapted, LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "v")))
    bad_dir = tmp_path / "bad-lora"
    with pytest.raises(ValueError, match="does not match"):
        export_checkpoint(adapted, bad_dir, lora_config=LoraConfig(r=8, lora_alpha=16, axes=("variate",)))
    assert not bad_dir.exists()

    # 3. Non-JSON metrics fail BEFORE writing any files (no partial export); NumPy scalars serialise cleanly.
    fail_dir = tmp_path / "fail-metrics"
    with pytest.raises(TypeError):
        export_checkpoint(tiny_forecaster.model, fail_dir, metrics={"bad": object()})
    assert not fail_dir.exists(), "failed export must not leave partial weights without LICENSE/NOTICE"

    ok_dir = export_checkpoint(
        tiny_forecaster.model, tmp_path / "np-metrics", metrics={"val_loss": np.float32(0.25), "steps": np.int64(3)}
    )
    meta = json.loads((ok_dir / "tune_metadata.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["val_loss"] == pytest.approx(0.25) and meta["metrics"]["steps"] == 3

    # 4. Chained export whose base dir holds our fallback LICENSE pointer does NOT claim verbatim LICENSE.
    chained_dir = tmp_path / "chained"
    caplog.clear()
    with caplog.at_level("WARNING", logger="timestral_tune.export"):
        export_checkpoint(tiny_forecaster.model, chained_dir, base_repo_or_path=ok_dir)
    chained_notice = (chained_dir / "NOTICE").read_text(encoding="utf-8")
    assert "no verbatim LICENSE was found" in chained_notice
    assert any("fallback LICENSE pointer" in r.getMessage() for r in caplog.records)

    # 5. Explicit `license_path` pointing at a fallback pointer also warns and stays non-verbatim.
    ptr_dir = tmp_path / "from-pointer-file"
    caplog.clear()
    with caplog.at_level("WARNING", logger="timestral_tune.export"):
        export_checkpoint(tiny_forecaster.model, ptr_dir, license_path=ok_dir / "LICENSE")
    assert "no verbatim LICENSE was found" in (ptr_dir / "NOTICE").read_text(encoding="utf-8")
    assert any("fallback pointer" in r.getMessage() for r in caplog.records)


def test_validate_export_inputs_checks_errors_in_order_without_io_or_merge(tiny_forecaster, tmp_path, monkeypatch):
    from timestral_tune import export as export_module
    from timestral_tune.models import inject_lora

    def _forbid(*args, **kwargs):
        raise AssertionError("validation helper must not call Hub, merge weights, or write files")

    monkeypatch.setattr(export_module, "hf_hub_download", _forbid)
    monkeypatch.setattr(export_module, "merged_state_dict", _forbid)
    monkeypatch.setattr(export_module, "extract_lora_state_dict", _forbid)
    monkeypatch.setattr(export_module, "save_file", _forbid)

    adapted = copy.deepcopy(tiny_forecaster.model)
    lora_cfg = LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "v"))
    inject_lora(adapted, lora_cfg)
    missing_license = tmp_path / "missing-license"
    valid_license = tmp_path / "LICENSE"
    valid_license.write_text("license text\n", encoding="utf-8")

    # 1. Unfitted calibrator beats LoRA mismatch and missing license_path.
    with pytest.raises(ValueError, match="unfitted"):
        export_module._validate_export_inputs(
            adapted,
            lora_config=LoraConfig(r=8, axes=("variate",)),
            calibrator=ConformalCalibrator(0.80),
            metrics={"bad": float("nan")},
            license_path=missing_license,
        )

    # 2. LoRA mismatch beats invalid metrics and missing license_path.
    with pytest.raises(ValueError, match="does not match"):
        export_module._validate_export_inputs(
            adapted,
            lora_config=LoraConfig(r=8, axes=("variate",)),
            metrics={"bad": float("nan")},
            license_path=missing_license,
        )

    # 3. Missing LoraConfig on adapted model without `_lora_config` raises ValueError.
    no_attr_model = copy.deepcopy(adapted)
    delattr(no_attr_model, "_lora_config")
    with pytest.raises(ValueError, match="without a LoraConfig"):
        export_module._validate_export_inputs(no_attr_model, license_path=missing_license)

    # 4. Non-JSON / NaN metrics fail with allow_nan=False before missing license_path.
    with pytest.raises(ValueError, match="JSON compliant"):
        export_module._validate_export_inputs(adapted, metrics={"bad": float("nan")}, license_path=missing_license)
    with pytest.raises(TypeError):
        export_module._validate_export_inputs(adapted, metrics={"bad": object()}, license_path=missing_license)

    # 5. Calibrator JSON with NaN fails before metrics and missing license_path.
    class _BadCalibrator:
        def to_dict(self):
            return {"bad": float("inf")}

    with pytest.raises(ValueError, match="JSON compliant"):
        export_module._validate_export_inputs(
            adapted,
            calibrator=_BadCalibrator(),  # type: ignore[arg-type]
            metrics={"bad": object()},
            license_path=missing_license,
        )

    # 6. Missing explicit license_path raises FileNotFoundError once earlier checks pass.
    with pytest.raises(FileNotFoundError, match="license_path"):
        export_module._validate_export_inputs(adapted, metrics={"loss": np.float32(0.5)}, license_path=missing_license)

    # 7. Valid inputs succeed with no Hub call, weight merge, or disk write.
    validated = export_module._validate_export_inputs(
        adapted,
        lora_config=lora_cfg,
        metrics={"loss": np.float32(0.5)},
        license_path=valid_license,
    )
    assert validated.model_has_lora is True
    assert validated.effective_lora == lora_cfg
    assert validated.adapter_config_json is not None
    assert validated.conformal_json is None
    assert validated.metrics_dict == {"loss": np.float32(0.5)}


def test_export_multiprocess_metadata_keys_only_when_world_size_gt_one(tiny_forecaster, tmp_path):
    cfg = TrainingConfig(model_name_or_path="local-test", device="cpu", batch_size=8)

    # Default and world_size=1 omit world_size and effective_batch_size.
    out_none = export_checkpoint(tiny_forecaster.model, tmp_path / "ws-none", training_config=cfg)
    meta_none = json.loads((out_none / "tune_metadata.json").read_text(encoding="utf-8"))
    assert "world_size" not in meta_none and "effective_batch_size" not in meta_none
    assert meta_none["hyperparameters"]["batch_size"] == 8

    out_one = export_checkpoint(
        tiny_forecaster.model, tmp_path / "ws-1", training_config=cfg, world_size=1, effective_batch_size=8
    )
    meta_one = json.loads((out_one / "tune_metadata.json").read_text(encoding="utf-8"))
    assert "world_size" not in meta_one and "effective_batch_size" not in meta_one

    # world_size > 1 writes top-level world_size and effective_batch_size while keeping per-rank batch_size.
    out_four = export_checkpoint(
        tiny_forecaster.model, tmp_path / "ws-4", training_config=cfg, world_size=4, effective_batch_size=32
    )
    meta_four = json.loads((out_four / "tune_metadata.json").read_text(encoding="utf-8"))
    assert meta_four["world_size"] == 4
    assert meta_four["effective_batch_size"] == 32
    assert meta_four["hyperparameters"]["batch_size"] == 8

    # Passing only one of world_size / effective_batch_size is rejected (no implicit fallback).
    bad_dir = tmp_path / "bad-ws"
    with pytest.raises(ValueError, match="passed together"):
        export_checkpoint(tiny_forecaster.model, bad_dir, training_config=cfg, world_size=2)
    with pytest.raises(ValueError, match="passed together"):
        export_checkpoint(tiny_forecaster.model, bad_dir, training_config=cfg, effective_batch_size=16)

    # Invalid world_size / effective_batch_size values fail before creating output_dir.
    with pytest.raises(ValueError, match="world_size"):
        export_checkpoint(tiny_forecaster.model, bad_dir, world_size=0, effective_batch_size=8)
    with pytest.raises(ValueError, match="world_size"):
        export_checkpoint(tiny_forecaster.model, bad_dir, world_size=True, effective_batch_size=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effective_batch_size"):
        export_checkpoint(tiny_forecaster.model, bad_dir, world_size=2, effective_batch_size=0)
    assert not bad_dir.exists()
