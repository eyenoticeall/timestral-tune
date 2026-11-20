# Upstream: Pins and Vendoring

`timestral-tune` builds on Mistral AI's official inference package, which is vendored into this repo.
The decision and its reasons are in the handoff, §6.1 and decision D3.

## Pins

| Artifact | Revision |
|---|---|
| `mistralai/timestral` (code, Apache 2.0) | commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` (2026-10-15) |
| `mistralai/Timestral-5M` | `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` |
| `mistralai/Timestral-28M` | `49b26409ac87b5935539a5d8ffe7f397b720db47` |
| `mistralai/Timestral-123M` | `5a147287addda85f6acd8da3fb8858fe9af7e8f7` |
| `mistralai/Timestral-209M` | `9874d252ab0aae16f7f7c999f98f98d6ad7d8365` |

The code pin is also recorded in `src/timestral_tune/_vendor/timestral/VENDOR_MANIFEST.json` and in
`scripts/vendor_upstream.py`. The Hub revisions are also in `scripts/probes/_common.py`.

## What is vendored and how

`scripts/vendor_upstream.py` downloads the nine source files of upstream's `src/timestral/` package and
its `LICENSE` at the pinned commit into `src/timestral_tune/_vendor/timestral/`.

**One modification is made, by the script:** absolute sibling imports (`from timestral.config import …`)
are rewritten to relative ones (`from .config import …`). Without this, the vendored copy would fail to
import, or would silently import an installed official `timestral` instead of itself. Each file changed
gets a four-line notice header, as Apache 2.0 requires for modified files.

| Vendored file | Modified |
|---|---|
| `__init__.py`, `forecaster.py`, `layers.py`, `loading.py`, `model.py` | Yes: imports rewritten, header added |
| `config.py`, `covariate_encoding.py`, `masks.py`, `scaler.py`, `LICENSE` | No: byte-identical to upstream |

The manifest records, per file, the sha256 of the pristine upstream bytes, the sha256 of the vendored
bytes, and whether it was modified.

**Line numbers.** The handoff cites upstream as `[U file:line]` at the pinned commit. In the five modified
vendored files, every line sits **four lines lower** than upstream because of the header. Unmodified files
match exactly.

## Rules

- Never edit a vendored file by hand. `tests/test_scaffold.py::test_vendored_files_match_manifest` fails
  if one changes.
- New behaviour goes outside `_vendor/`: subclass, wrap, or call the vendored functions.
- Never rename upstream parameters. Exported weights must load in the official package with `strict=True`.
- If a real change to vendored code ever becomes unavoidable, make it in the script (so it is repeatable),
  extend the notice header to describe it, and record the decision in handoff §11.

## Checking for drift

```bash
python scripts/check_upstream_drift.py            # against upstream main
python scripts/check_upstream_drift.py --ref <ref>
```

It compares upstream file hashes with the pristine hashes in the manifest and reports added, removed or
changed files. Exit code 1 means upstream moved. `.github/workflows/upstream-drift.yml` runs it weekly.

Last checked 2026-11-21: no drift (upstream `main` was still the pinned commit).

## Re-vendoring procedure

1. Read the upstream diff between the old and new commit. Look for changed function signatures, changed
   private functions used by our code (`_run_two_stream`, `_assemble_rows`, `_refine_block`,
   `_cap_quantiles`, `_encode_categorical`, `advancing_block_frames`, the mask builders), and new files
   (add them to `SOURCE_FILES` in the script).
2. Update `PINNED_COMMIT` in `scripts/vendor_upstream.py` and run it.
3. Run the full test suite. Forward-parity tests (T1–T3) and the export round-trip (T11) are the ones
   that prove nothing we depend on changed.
4. Re-run the probes and update `docs/VERIFICATION.md` if any number moves.
5. Update the pins table above and in the handoff header, and fix any `[U file:line]` references that moved.
6. Commit as its own commit: `build(vendor): re-vendor timestral at <short sha>`.

If Mistral publishes new checkpoint revisions, update the Hub pins the same way and re-run the probes.

## Notes on upstream

- Upstream does not accept pull requests and makes no stability promise for private functions.
- `timestral` is not on PyPI, and the name is unclaimed there. Never add a bare `timestral` requirement.
- `TimestralConfig.from_dict` rejects unknown keys, so nothing may be added to a checkpoint's `config.json`.
- Always read hyperparameters from the checkpoint's `config.json`; code defaults differ from the released
  values (for example `quantile_width_factor`).
