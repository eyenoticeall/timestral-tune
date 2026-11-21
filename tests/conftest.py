"""Shared fixtures.

Most acceptance tests run three times: on a tiny random model (network-free, so CI covers the
logic) and on the released 5M and 28M checkpoints (marked `network`; these are the runs the
roadmap's exit criteria ask for).
"""

from __future__ import annotations

import pytest
import torch

from timestral_tune.models import TuneTimestralForecaster, load_forecaster, mps_is_usable

from .helpers import RawSeries, make_raw, make_tiny_forecaster

MODEL_PARAMS = [
    pytest.param("tiny", id="tiny"),
    pytest.param("5M", id="5M", marks=pytest.mark.network),
    pytest.param("28M", id="28M", marks=pytest.mark.network),
]

# Scenario name -> (target rows, past-only rows, known-future rows) per series.
SCENARIOS = {
    "univariate": (1, 0, 0),
    "multivariate": (3, 0, 0),
    "covariates": (1, 2, 2),
    "multi_target_covariates": (2, 1, 1),
}


def _load(name: str) -> TuneTimestralForecaster:
    if name == "tiny":
        return make_tiny_forecaster()
    try:
        # quantile_real_cap_k=0 disables upstream's output clamp, as acceptance test T3 requires.
        return load_forecaster(f"mistralai/Timestral-{name}", quantile_real_cap_k=0)
    except Exception as error:  # noqa: BLE001 - any download or cache failure means "cannot run here"
        pytest.skip(f"released checkpoint {name} is unavailable: {type(error).__name__}: {error}")


@pytest.fixture(scope="session", params=MODEL_PARAMS)
def forecaster(request) -> TuneTimestralForecaster:
    """A CPU forecaster with upstream's output clamp disabled: tiny random, then 5M, then 28M."""
    return _load(request.param)


@pytest.fixture
def is_released_checkpoint(request) -> bool:
    """Whether the current `forecaster` is a released checkpoint rather than the tiny random model."""
    return request.node.callspec.params["forecaster"] != "tiny"


@pytest.fixture(scope="session")
def tiny_forecaster() -> TuneTimestralForecaster:
    return make_tiny_forecaster()


@pytest.fixture(params=sorted(SCENARIOS))
def scenario(request) -> str:
    return request.param


@pytest.fixture
def raw(forecaster, scenario) -> RawSeries:
    """Raw series for `scenario`, sized to the model: unaligned context, partial final horizon patch."""
    patch = forecaster.patch_size
    targets, past, known = SCENARIOS[scenario]
    return make_raw(
        seed=sum(map(ord, scenario)),
        targets=targets,
        past=past,
        known=known,
        context_len=9 * patch + patch // 2 + 1,
        horizon=2 * patch + patch // 2,
    )


def available_accelerators() -> list[str]:
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if mps_is_usable():
        devices.append("mps")
    return devices
