"""Network-free checks that the scaffold is sound.

These are not acceptance tests. They verify the package imports, the vendored upstream copy is
intact and isolated, and a tiny randomly initialised model runs through the vendored code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import timestral_tune
from timestral_tune._vendor import timestral as vendored
from timestral_tune._vendor.timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET

VENDOR_DIR = Path(vendored.__file__).parent
QUANTILE_LEVELS = (0.01, *(round(0.05 * i, 2) for i in range(1, 20)), 0.99)


def test_version_is_exported():
    assert isinstance(timestral_tune.__version__, str) and timestral_tune.__version__


def test_vendored_files_match_manifest():
    """Catches hand edits to vendored files. Regenerate with scripts/vendor_upstream.py instead."""
    manifest = json.loads((VENDOR_DIR / "VENDOR_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["commit"]) == 40
    for name, meta in manifest["files"].items():
        digest = hashlib.sha256((VENDOR_DIR / name).read_bytes()).hexdigest()
        assert digest == meta["vendored_sha256"], f"{name} differs from the vendored manifest"


def test_pinned_revisions_agree_between_the_loader_and_the_documents():
    """The Hub pins live in `models/loader.py`; `docs/UPSTREAM.md` and the spec header repeat them."""
    from timestral_tune.models import PINNED_REVISIONS

    root = Path(__file__).resolve().parents[1]
    upstream = (root / "docs" / "UPSTREAM.md").read_text(encoding="utf-8")
    manifest = json.loads((Path(vendored.__file__).parent / "VENDOR_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(PINNED_REVISIONS) == 4
    for repo_id, revision in PINNED_REVISIONS.items():
        assert f"| `{repo_id}` | `{revision}` |" in upstream, f"{repo_id} is pinned differently in docs/UPSTREAM.md"
    assert manifest["commit"] in upstream, "the vendored commit is not the one docs/UPSTREAM.md names"


def test_vendored_license_is_shipped():
    assert "Apache License" in (VENDOR_DIR / "LICENSE").read_text(encoding="utf-8")


def test_vendored_import_is_isolated():
    """Importing the vendored copy must not import (or require) a top-level `timestral`."""
    code = (
        "import sys; import timestral_tune._vendor.timestral as v; "
        "assert 'timestral' not in sys.modules, 'vendored copy pulled in a top-level timestral'; "
        "assert v.TimestralModel.__module__.startswith('timestral_tune._vendor.timestral')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def tiny_model():
    """A small randomly initialised model with one variate-attention layer. Not a released checkpoint."""
    config = vendored.TimestralConfig(
        patch_size=4,
        d_model=32,
        num_heads=2,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate"),
        quantile_knots=QUANTILE_LEVELS,
        block_size=4,
        context_length=64,
    )
    torch.manual_seed(0)
    model = vendored.TimestralModel(config)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return model.eval()


def test_tiny_model_forecasts_without_covariates(tiny_model):
    patch = tiny_model.config.patch_size
    rows, context_patches, horizon_patches = 4, 6, 2
    inputs = torch.randn(rows, context_patches * patch).cumsum(-1)
    observed = torch.ones_like(inputs, dtype=torch.bool)
    group_ids = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        quantiles = tiny_model.predict_block(inputs, observed, group_ids, horizon_patches)
    assert quantiles.shape == (rows, horizon_patches * patch, len(QUANTILE_LEVELS))
    assert torch.isfinite(quantiles).all()
    assert (quantiles.diff(dim=-1) >= 0).all(), "quantiles must be non-decreasing"


def test_tiny_model_forecasts_with_covariates(tiny_model):
    assert tiny_model.config.supports_known_covariates
    patch = tiny_model.config.patch_size
    context_patches, horizon_patches = 6, 2
    roles = torch.tensor([ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV])
    rows = roles.numel()
    inputs = torch.randn(rows, context_patches * patch).cumsum(-1)
    observed = torch.ones_like(inputs, dtype=torch.bool)
    group_ids = torch.zeros(rows, dtype=torch.long)
    known_future = torch.zeros(rows, horizon_patches * patch)
    known_future[2] = torch.randn(horizon_patches * patch)
    known_future_mask = torch.zeros(rows, horizon_patches * patch, dtype=torch.bool)
    known_future_mask[2] = True
    with torch.no_grad():
        quantiles = tiny_model.predict_block(
            inputs, observed, group_ids, horizon_patches, roles, known_future, known_future_mask
        )
    assert quantiles.shape == (rows, horizon_patches * patch, len(QUANTILE_LEVELS))
    assert torch.isfinite(quantiles).all()
