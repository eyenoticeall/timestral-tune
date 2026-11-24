"""Shared fixtures.

Most acceptance tests run three times: on a tiny random model (network-free, so CI covers the
logic) and on the released 5M and 28M checkpoints (marked `network`; these are the runs the
roadmap's exit criteria ask for).
"""

from __future__ import annotations

import os

import pytest
import torch

from timestral_tune.models import TuneTimestralForecaster, load_forecaster, mps_is_usable

from .distributed_harness import (
    DIST_BACKEND_ENV,
    DIST_DEVICE_MAP_ENV,
    dist_backend,
    dist_device_map,
    is_cuda_backend,
)
from .helpers import RawSeries, make_raw, make_tiny_forecaster


def pytest_configure(config: pytest.Config) -> None:
    """Fail fast on invalid opt-in distributed test environment variables (no-op when unset)."""
    del config
    if DIST_BACKEND_ENV not in os.environ and DIST_DEVICE_MAP_ENV not in os.environ:
        return
    try:
        backend = dist_backend()
        device_map = dist_device_map(backend=backend)
        if is_cuda_backend(backend) and device_map == "shared" and os.environ.get("NCCL_MULTI_RANK_GPU_ENABLE") != "1":
            raise ValueError(
                f"{DIST_DEVICE_MAP_ENV}='shared' requires NCCL_MULTI_RANK_GPU_ENABLE=1 "
                "(development-only single-GPU mode; distinct from per-rank multi-GPU verification)"
            )
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc


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
    # Covariates but none known in advance: the covariate layout with no known-future values at all.
    "past_only_covariates": (1, 2, 0),
}


def _load(name: str) -> TuneTimestralForecaster:
    if name == "tiny":
        return make_tiny_forecaster()
    # No try/except here on purpose: these tests carry the `network` marker, which is the way to opt out
    # (`-m "not network"`). A loader failure of any kind, a wrong pinned revision included, must fail them.
    # quantile_real_cap_k=0 disables upstream's output clamp, as acceptance test T3 requires.
    return load_forecaster(f"mistralai/Timestral-{name}", quantile_real_cap_k=0)


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


@pytest.fixture(autouse=True)
def _guard_torch_global_state():
    """Fail loudly if any test leaks process-global PyTorch state to later tests."""
    threads_before = torch.get_num_threads()
    grad_before = torch.is_grad_enabled()
    dtype_before = torch.get_default_dtype()
    yield
    leaks: list[str] = []
    if torch.get_num_threads() != threads_before:
        leaks.append(f"num_threads {threads_before} -> {torch.get_num_threads()}")
        torch.set_num_threads(threads_before)
    if torch.is_grad_enabled() != grad_before:
        leaks.append(f"grad_enabled {grad_before} -> {torch.is_grad_enabled()}")
        torch.set_grad_enabled(grad_before)
    if torch.get_default_dtype() != dtype_before:
        leaks.append(f"default_dtype {dtype_before} -> {torch.get_default_dtype()}")
        torch.set_default_dtype(dtype_before)
    if leaks:
        pytest.fail("test leaked process-global PyTorch state: " + ", ".join(leaks))
