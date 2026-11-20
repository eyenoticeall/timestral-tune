"""Report whether upstream `mistralai/timestral` has moved away from the vendored copy.

Usage (from the repo root):

    python scripts/check_upstream_drift.py              # compare against upstream `main`
    python scripts/check_upstream_drift.py --ref <ref>  # compare against another branch, tag or commit

Compares the sha256 of each upstream source file at `--ref` with the pristine hashes recorded
in `VENDOR_MANIFEST.json`, and reports files added to or removed from the upstream package.

Exit code 0: no drift. Exit code 1: drift found (details printed). Standard library only.
See docs/UPSTREAM.md for what to do when this fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MANIFEST = (
    Path(__file__).resolve().parent.parent / "src" / "timestral_tune" / "_vendor" / "timestral" / "VENDOR_MANIFEST.json"
)


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "generic-client"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https hosts
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="main", help="upstream branch, tag or commit to compare against")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    repo, package_dir = manifest["repo"], manifest["upstream_package_dir"]
    recorded = {name: meta["upstream_sha256"] for name, meta in manifest["files"].items() if name != "LICENSE"}

    tree = json.loads(_get(f"https://api.github.com/repos/{repo}/git/trees/{args.ref}?recursive=1"))
    prefix = package_dir + "/"
    upstream_files = sorted(
        entry["path"][len(prefix) :]
        for entry in tree.get("tree", [])
        if entry["type"] == "blob" and entry["path"].startswith(prefix)
    )

    problems: list[str] = []
    for name in sorted(set(upstream_files) - set(recorded)):
        problems.append(f"new upstream file: {name}")
    for name in sorted(set(recorded) - set(upstream_files)):
        problems.append(f"removed upstream: {name}")
    for name in sorted(set(recorded) & set(upstream_files)):
        current = hashlib.sha256(_get(f"https://raw.githubusercontent.com/{repo}/{args.ref}/{package_dir}/{name}"))
        if current.hexdigest() != recorded[name]:
            problems.append(f"changed upstream: {name}")

    print(f"vendored commit: {manifest['commit']}")
    print(f"compared against: {repo}@{args.ref} ({tree.get('sha', 'unknown tree')})")
    if problems:
        print("DRIFT:")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("no drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
