"""Unit tests for checkpoint loading and the device shims. Network-free."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from safetensors.torch import save_file

from timestral_tune._vendor import timestral as vendored
from timestral_tune.models import (
    PINNED_REVISIONS,
    TuneTimestralForecaster,
    TuneTimestralModel,
    load_forecaster,
    load_model,
    mps_is_usable,
    resolve_device,
    resolve_revision,
    supports_float64,
)

from .conftest import available_accelerators
from .helpers import TINY_CONFIG, make_raw, max_rel

needs_mps = pytest.mark.skipif(not mps_is_usable(), reason="no Apple GPU, or PyTorch is older than its minimum for it")


@pytest.fixture
def checkpoint_dir(tiny_forecaster, tmp_path):
    """The tiny model saved in upstream's export layout: `config.json` + `model.safetensors`."""
    model = tiny_forecaster.model
    model.config.save(tmp_path / "config.json")
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    return tmp_path


def test_pinned_revisions_cover_the_four_official_checkpoints():
    assert set(PINNED_REVISIONS) == {f"mistralai/Timestral-{size}" for size in ("5M", "28M", "123M", "209M")}
    assert all(len(sha) == 40 for sha in PINNED_REVISIONS.values())


def test_resolve_revision_pins_official_repos_only():
    assert resolve_revision("mistralai/Timestral-5M", None) == PINNED_REVISIONS["mistralai/Timestral-5M"]
    assert resolve_revision("mistralai/Timestral-5M", "main") == "main"
    assert resolve_revision("someone/their-finetune", None) is None
    assert resolve_revision("/some/local/dir", None) is None


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    assert resolve_device(None) == resolve_device("auto")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="Apple GPU backend is not available"):
            resolve_device("mps")


def test_the_apple_gpu_needs_a_recent_pytorch(monkeypatch):
    """Below PyTorch 2.8 the backend lacks an operator upstream uses: `auto` skips it, `mps` raises."""
    from timestral_tune.models import MIN_TORCH_FOR_MPS, device

    assert MIN_TORCH_FOR_MPS == (2, 8)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    monkeypatch.setattr(device, "_torch_version", lambda: (2, 7))
    assert not mps_is_usable()
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="needs PyTorch 2.8 or newer"):
        resolve_device("mps")

    monkeypatch.setattr(device, "_torch_version", lambda: (2, 8))
    assert mps_is_usable()
    assert resolve_device("auto") == torch.device("mps") == resolve_device("mps")


def test_supports_float64():
    assert supports_float64("cpu") and supports_float64(torch.device("cuda"))
    assert not supports_float64("mps")


def test_load_model_round_trips_a_local_checkpoint(checkpoint_dir, tiny_forecaster):
    model = load_model(checkpoint_dir)
    assert isinstance(model, TuneTimestralModel)
    assert not model.training
    assert model.config == tiny_forecaster.model.config
    expected = tiny_forecaster.model.state_dict()
    assert all(torch.equal(value, expected[name]) for name, value in model.state_dict().items())


def test_load_model_reads_hyperparameters_from_the_checkpoint_not_from_defaults(checkpoint_dir, tiny_forecaster):
    config = dataclasses.replace(tiny_forecaster.model.config, quantile_width_factor=3.0)
    config.save(checkpoint_dir / "config.json")
    assert load_model(checkpoint_dir).config.quantile_width_factor == 3.0


def test_load_forecaster_returns_the_shimmed_forecaster(checkpoint_dir):
    forecaster = load_forecaster(checkpoint_dir, quantile_real_cap_k=0)
    assert isinstance(forecaster, TuneTimestralForecaster)
    assert isinstance(forecaster.model, TuneTimestralModel)
    assert forecaster.quantile_real_cap_k == 0
    assert isinstance(TuneTimestralForecaster.from_pretrained(checkpoint_dir), TuneTimestralForecaster)


def test_exported_weights_load_in_the_vendored_upstream_loader(checkpoint_dir):
    """Parameter names are upstream's, so a checkpoint we can load, upstream can load with strict=True."""
    upstream = vendored.load_model(checkpoint_dir)
    assert type(upstream) is vendored.TimestralModel


def test_forecaster_rejects_a_plain_upstream_model_and_from_upstream_converts_it(checkpoint_dir):
    upstream = vendored.load_model(checkpoint_dir)
    with pytest.raises(TypeError, match="TuneTimestralModel"):
        TuneTimestralForecaster(upstream)
    converted = TuneTimestralModel.from_upstream(upstream)
    assert isinstance(converted, TuneTimestralModel) and not converted.training
    assert TuneTimestralModel.from_upstream(converted) is converted
    raw = make_raw(1, context_len=37, horizon=10)
    reference = vendored.TimestralForecaster(upstream).forecast(raw.context, raw.horizon).quantiles
    assert torch.equal(TuneTimestralForecaster(converted).forecast(raw.context, raw.horizon).quantiles, reference)


def test_on_cpu_the_shims_are_exactly_upstream(checkpoint_dir):
    raw = make_raw(2, targets=1, past=1, known=1, context_len=37, horizon=10)
    reference = vendored.TimestralForecaster(vendored.load_model(checkpoint_dir)).forecast(
        raw.context, raw.horizon, **raw.forecast_kwargs()
    )
    ours = load_forecaster(checkpoint_dir).forecast(raw.context, raw.horizon, **raw.forecast_kwargs())
    assert torch.equal(ours.quantiles, reference.quantiles)


@needs_mps
def test_mps_rejects_non_float32_weights(checkpoint_dir):
    with pytest.raises(ValueError, match="float32 weights only"):
        load_model(checkpoint_dir, device="mps", dtype=torch.bfloat16)


@needs_mps
def test_plain_upstream_fails_on_mps_and_the_shims_fix_it(checkpoint_dir):
    """The reason this module exists: upstream's float64 statistics raise on the Apple GPU."""
    raw = make_raw(3, targets=1, past=1, known=1, context_len=37, horizon=10)
    upstream = vendored.TimestralForecaster(vendored.load_model(checkpoint_dir, device="mps"))
    with pytest.raises(TypeError, match="float64"):
        upstream.forecast(raw.context, raw.horizon, **raw.forecast_kwargs())

    reference = load_forecaster(checkpoint_dir).forecast(raw.context, raw.horizon, **raw.forecast_kwargs())
    on_gpu = load_forecaster(checkpoint_dir, device="mps").forecast(raw.context, raw.horizon, **raw.forecast_kwargs())
    assert on_gpu.quantiles.device.type == "mps"
    assert max_rel(on_gpu.quantiles, reference.quantiles) < 1e-4


@pytest.mark.parametrize("device", available_accelerators() or [pytest.param("cpu", id="no-accelerator")])
def test_every_float64_site_runs_on_the_device(checkpoint_dir, device):
    """Exercises all four overridden methods: categorical covariates, the output clamp, and a horizon of
    several patches (which goes through block refinement).
    """
    forecaster = load_forecaster(checkpoint_dir, device=device)  # default quantile_real_cap_k: clamp is on
    reference = load_forecaster(checkpoint_dir)
    patch = forecaster.patch_size
    raw = make_raw(4, targets=1, past=1, known=1, context_len=9 * patch + 1, horizon=3 * patch)
    codes = torch.randint(0, 4, (raw.context.shape[0], 1, raw.context.shape[-1] + raw.horizon)).double().numpy()
    kwargs = dict(raw.forecast_kwargs(), categorical_known_covariates=codes)
    ours = forecaster.forecast(raw.context, raw.horizon, **kwargs).quantiles
    expected = reference.forecast(raw.context, raw.horizon, **kwargs).quantiles
    assert max_rel(ours, expected) < 1e-4


def test_from_upstream_copies_float64_weights_exactly():
    """A float64 model must come back with the same float64 weights, not ones rounded through float32."""
    torch.manual_seed(3)
    source = vendored.TimestralModel(TINY_CONFIG).double()
    with torch.no_grad():
        for parameter in source.parameters():  # give the weights digits that float32 cannot hold
            parameter.add_(torch.randn_like(parameter) * 1e-9)
    tuned = TuneTimestralModel.from_upstream(source)
    assert next(tuned.parameters()).dtype == torch.float64
    rounded = 0
    for name, value in source.state_dict().items():
        assert torch.equal(tuned.state_dict()[name], value), name
        rounded += int((value.float().double() != value).sum()) if value.is_floating_point() else 0
    assert rounded > 0, "the source weights must not be exactly representable in float32"
    assert TuneTimestralModel.from_upstream(tuned) is tuned
