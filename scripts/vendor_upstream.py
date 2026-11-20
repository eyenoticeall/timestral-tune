"""Vendor the official `mistralai/timestral` package at a pinned commit.

Usage (from the repo root):

    python scripts/vendor_upstream.py            # fetch, rewrite imports, write files and manifest
    python scripts/vendor_upstream.py --commit <sha>   # re-vendor at another commit

What it does, for each upstream source file:

1. Downloads the file at the pinned commit and records the sha256 of the pristine bytes.
2. Rewrites absolute sibling imports (`from timestral.x import y`) to relative ones
   (`from .x import y`). This is the only modification made. Without it the vendored
   copy would either fail to import or silently import an installed official `timestral`.
3. Prepends a modification notice to every file it changed (Apache-2.0 section 4b).
4. Writes `VENDOR_MANIFEST.json` with the commit, and per file: the pristine sha256,
   the vendored sha256 and whether it was modified.

Standard library only. See docs/UPSTREAM.md for the re-vendoring procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

REPO = "mistralai/timestral"
PINNED_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
UPSTREAM_PACKAGE_DIR = "src/timestral"
SOURCE_FILES = (
    "__init__.py",
    "config.py",
    "covariate_encoding.py",
    "forecaster.py",
    "layers.py",
    "loading.py",
    "masks.py",
    "model.py",
    "scaler.py",
)
LICENSE_FILE = "LICENSE"  # at the upstream repo root

DEST = Path(__file__).resolve().parent.parent / "src" / "timestral_tune" / "_vendor" / "timestral"
MANIFEST_NAME = "VENDOR_MANIFEST.json"

_IMPORT_RE = re.compile(r"^from timestral\.(\w+) import", flags=re.MULTILINE)

_HEADER = """\
# Vendored from https://github.com/{repo} at commit {commit} (Apache-2.0).
# Modified by timestral-tune: absolute `timestral.*` imports were rewritten to relative
# imports so that this copy never resolves to an installed `timestral`. No other changes.
# Regenerate with scripts/vendor_upstream.py; do not edit by hand.
"""


def fetch(repo: str, commit: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "generic-client"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https host
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rewrite(source: str, repo: str, commit: str) -> tuple[str, bool]:
    """Return `(vendored_source, modified)`."""
    rewritten = _IMPORT_RE.sub(r"from .\1 import", source)
    if rewritten == source:
        return source, False
    return _HEADER.format(repo=repo, commit=commit) + rewritten, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", default=PINNED_COMMIT, help="upstream commit to vendor")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}

    for name in SOURCE_FILES:
        pristine = fetch(REPO, args.commit, f"{UPSTREAM_PACKAGE_DIR}/{name}")
        vendored, modified = rewrite(pristine.decode("utf-8"), REPO, args.commit)
        if re.search(r"^\s*(from|import) timestral\b", vendored, flags=re.MULTILINE):
            raise SystemExit(f"{name}: an absolute `timestral` import survived the rewrite")
        data = vendored.encode("utf-8")
        (DEST / name).write_bytes(data)
        files[name] = {"upstream_sha256": sha256(pristine), "vendored_sha256": sha256(data), "modified": modified}
        print(f"{'modified ' if modified else 'verbatim '} {name}")

    license_bytes = fetch(REPO, args.commit, LICENSE_FILE)
    (DEST / LICENSE_FILE).write_bytes(license_bytes)
    files[LICENSE_FILE] = {
        "upstream_sha256": sha256(license_bytes),
        "vendored_sha256": sha256(license_bytes),
        "modified": False,
    }

    manifest = {
        "repo": REPO,
        "commit": args.commit,
        "upstream_package_dir": UPSTREAM_PACKAGE_DIR,
        "modification": "absolute `timestral.*` imports rewritten to relative imports; notice header added",
        "files": files,
    }
    (DEST / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(files)} files and {MANIFEST_NAME} to {DEST}")


if __name__ == "__main__":
    main()
