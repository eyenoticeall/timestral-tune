"""Regression tests for `scripts/multi_gpu_check.py` and its thin `scripts/multi_gpu_check.sh` wrapper.

Runs entirely on lightweight stub stages (no `torch`, no `torchrun`, no network) on both macOS and
Linux. Every subprocess test uses deterministic ready-file handshakes, generous CI timeouts, and
`try/finally` process-group cleanup with explicit leftover-process checks.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_SH = REPO_ROOT / "scripts" / "multi_gpu_check.sh"
SCRIPT_PY = REPO_ROOT / "scripts" / "multi_gpu_check.py"


def _load_driver_module() -> ModuleType:
    """Import `scripts/multi_gpu_check.py` without executing `main()`."""
    spec = importlib.util.spec_from_file_location("multi_gpu_check_driver", SCRIPT_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


DRIVER = _load_driver_module()


# --------------------------------------------------------------------------- test helpers


def _make_valid_ref_zip(path: Path, payload: bytes = b"\x00" * 32) -> Path:
    """Write a minimal valid reference zip archive passing `verify_and_hash_ref_archive`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    spec_doc = {"spec": {"schema_version": 5}, "environment": {}}
    entries_doc = {
        "tiny_random:domain_head_only_f32": {
            "tensor_member": "tensors/0000.bin",
            "tensor_dtype": "float32",
            "tensor_shape": [len(payload) // 4],
        }
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", payload)
        zf.writestr("spec.json", json.dumps(spec_doc).encode())
        zf.writestr("entries.json", json.dumps(entries_doc).encode())
    return path


def _flip_zip_member_byte(src: Path, dst: Path, member: str = "tensors/0000.bin") -> Path:
    """Copy `src` to `dst` and flip one byte in `member`'s raw data (preserving size, breaking CRC-32)."""
    raw = bytearray(src.read_bytes())
    with zipfile.ZipFile(src, "r") as zf:
        info = zf.getinfo(member)
        lh_off = info.header_offset
        fn_len = int.from_bytes(raw[lh_off + 26 : lh_off + 28], "little")
        ex_len = int.from_bytes(raw[lh_off + 28 : lh_off + 30], "little")
        data_off = lh_off + 30 + fn_len + ex_len
        raw[data_off] ^= 0xFF
    dst.write_bytes(bytes(raw))
    return dst


def _list_matching_pids(tag: str) -> list[str]:
    """Return live non-zombie process entries matching `tag` in argv or `_TT_TEST_LEFTOVER_TAG` env."""
    self_pid = os.getpid()
    env_needle = f"_TT_TEST_LEFTOVER_TAG={tag}"
    hits_by_pid: dict[int, str] = {}

    if os.path.isdir("/proc"):
        saw_proc_pid = False
        try:
            proc_entries = os.listdir("/proc")
        except OSError:
            proc_entries = []
        env_needle_b = env_needle.encode("utf-8")
        for entry in proc_entries:
            if not (entry.isascii() and entry.isdigit()):
                continue
            saw_proc_pid = True
            pid = int(entry)
            if pid == self_pid:
                continue
            try:
                stat = Path(f"/proc/{pid}/stat").read_bytes()
                rparen = stat.rfind(b")")
                if rparen >= 0:
                    fields = stat[rparen + 2 :].split()
                    if fields and fields[0] == b"Z":
                        continue
                cmd_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
                cmd = cmd_bytes.replace(b"\0", b" ").decode(errors="replace").strip()
                matched = tag in cmd
                if not matched:
                    try:
                        env_bytes = Path(f"/proc/{pid}/environ").read_bytes()
                        matched = env_needle_b in env_bytes.split(b"\0")
                    except OSError:
                        pass
                if matched:
                    hits_by_pid[pid] = f"{pid} {cmd}"
            except OSError:
                continue
        if saw_proc_pid:
            return [hits_by_pid[p] for p in sorted(hits_by_pid)]

    ps_bin = DRIVER._resolve_ps_bin()
    if ps_bin is not None:
        # Pass 1: untruncated argv only (no unrelated env noise).
        out_cmd = subprocess.run(
            [ps_bin, "-ww", "-axo", "pid=,stat=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if out_cmd.returncode == 0:
            for line in out_cmd.stdout.splitlines():
                parts = line.strip().split(None, 2)
                if len(parts) < 3 or not (parts[0].isascii() and parts[0].isdigit()):
                    continue
                pid = int(parts[0])
                if pid == self_pid or parts[1].startswith("Z"):
                    continue
                if tag in parts[2] and "pid=,stat=,command=" not in parts[2]:
                    hits_by_pid[pid] = f"{pid} {parts[2]}"
        # Pass 2: environment tokens via `ps eww`, matching exact `_TT_TEST_LEFTOVER_TAG=<tag>` assignment.
        out_env = subprocess.run(
            [ps_bin, "eww", "-axo", "pid=,stat=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if out_env.returncode == 0:
            for line in out_env.stdout.splitlines():
                parts = line.strip().split(None, 2)
                if len(parts) < 3 or not (parts[0].isascii() and parts[0].isdigit()):
                    continue
                pid = int(parts[0])
                if pid == self_pid or pid in hits_by_pid or parts[1].startswith("Z"):
                    continue
                if env_needle in parts[2].split():
                    hits_by_pid[pid] = f"{pid} {parts[2]}"
    return [hits_by_pid[p] for p in sorted(hits_by_pid)]


def _cleanup_tagged_processes(tag: str) -> None:
    """Best-effort SIGKILL sweep for any live processes matching `tag` in argv or `_TT_TEST_LEFTOVER_TAG`."""
    deadline = time.monotonic() + 1.0
    while True:
        hits = _list_matching_pids(tag)
        if not hits:
            return
        for entry in hits:
            pid_str = entry.split(None, 1)[0]
            if pid_str.isascii() and pid_str.isdigit():
                try:
                    os.kill(int(pid_str), signal.SIGKILL)
                except OSError:
                    pass
        if time.monotonic() >= deadline:
            return
        time.sleep(0.02)


def _wait_for_path(path: Path, proc: subprocess.Popen[Any] | None = None, *, timeout: float = 20.0) -> None:
    """Wait until `path` exists or fail fast if `proc` exits prematurely or `timeout` elapses."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if proc is not None and proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout is not None else ""
            raise AssertionError(f"process exited early (rc={proc.returncode}) before {path.name} appeared:\n{out}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {path}")
        time.sleep(0.01)


def _kill_process_group_quiet(proc: subprocess.Popen[Any]) -> None:
    """Ensure `proc` and its process group are fully dead and reaped during test cleanup."""
    if proc.poll() is None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except OSError:
                pass
            try:
                proc.wait(timeout=3.0)
                break
            except subprocess.TimeoutExpired:
                continue
    if proc.stdout is not None:
        try:
            proc.stdout.close()
        except OSError:
            pass


def _make_stub_repo(base_dir: Path, *, with_git: bool = False) -> Path:
    """Create a minimal standalone repo containing the real driver scripts for isolated subprocess tests."""
    repo = base_dir / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "probes").mkdir(parents=True)
    (repo / "src" / "timestral_tune").mkdir(parents=True)
    shutil.copy2(SCRIPT_SH, repo / "scripts" / "multi_gpu_check.sh")
    shutil.copy2(SCRIPT_PY, repo / "scripts" / "multi_gpu_check.py")
    (repo / "scripts" / "probes" / "_dist_common.py").write_text("# stub helper\n", encoding="utf-8")
    (repo / "src" / "timestral_tune" / "config.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text('[project]\nname = "stub"\n', encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    if with_git:
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


def _write_stub_uv(bin_dir: Path, body: str) -> Path:
    """Create an executable stub `uv` script in `bin_dir` running with the current Python interpreter."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    uv_path = bin_dir / "uv"
    uv_path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    uv_path.chmod(0o755)
    return uv_path


def _standard_stub_uv_body(
    *,
    env_sync_rc: int = 0,
    gpu_count: int = 0,
    cuda_available: bool = False,
    fail_ref_n: set[int] | None = None,
    corrupt_ref_n: set[int] | None = None,
    call_log: Path | None = None,
) -> str:
    """Return a Python script body for a fast, deterministic stub `uv` handling every session stage."""
    return f"""\
import json, pathlib, sys, uuid, zipfile

argv = sys.argv[1:]
joined = " ".join(argv)
call_log = {str(call_log)!r}
if call_log is not None:
    with open(call_log, "a", encoding="utf-8") as cf:
        cf.write(joined + "\\n")

if argv and argv[0] == "sync":
    sys.exit({env_sync_rc})

if "sys.executable" in joined and "torch.version.cuda" in joined:
    print("stub-runtime-env-sig-v1")
    sys.exit(0)

if "torch.cuda.device_count()" in joined and "tree_state_method=" not in joined:
    print({gpu_count!r})
    sys.exit(0)

if "tree_state_method=" in joined:
    backend, tree_method, tree_sig_short = argv[-3], argv[-2], argv[-1]
    print("os=Stub arch=stub python=3.12.0")
    print(f"tree_state_method={{tree_method}} tree_state_sig={{tree_sig_short}}")
    print("torch=2.0.0 cuda_available={cuda_available} cuda_runtime=None")
    if backend == "nccl" and not {cuda_available}:
        print("error: --backend nccl requested, but this PyTorch build cannot use CUDA", file=sys.stderr)
        sys.exit(3)
    sys.exit(0)

if "--write-reference" in argv:
    ref_path = pathlib.Path(argv[argv.index("--write-reference") + 1])
    ws = int(argv[argv.index("--world-size") + 1])
    if ws in {sorted(fail_ref_n or ())!r}:
        print(f"simulated ref failure for N={{ws}}", file=sys.stderr)
        sys.exit(1)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    if ws in {sorted(corrupt_ref_n or ())!r}:
        ref_path.write_bytes(b"not-a-valid-zip")
        sys.exit(0)
    spec_doc = {{"spec": {{"schema_version": 5}}, "environment": {{"archive_id": uuid.uuid4().hex}}}}
    entries_doc = {{
        "tiny_random:s": {{
            "tensor_member": "tensors/0000.bin",
            "tensor_dtype": "float32",
            "tensor_shape": [4],
        }}
    }}
    with zipfile.ZipFile(ref_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", b"\\x00" * 16)
        zf.writestr("spec.json", json.dumps(spec_doc).encode())
        zf.writestr("entries.json", json.dumps(entries_doc).encode())
    print(f"wrote stub ref for N={{ws}}")
    sys.exit(0)

if "torchrun" in argv:
    print("stub torchrun ok")
    sys.exit(0)

if "distributed_rank_failure_check.py" in joined:
    print("result: PASS (no hang) | parent_killed=none")
    sys.exit(0)

if "distributed_parity.py" in joined:
    print("stub prototype parity ok")
    sys.exit(0)

sys.exit(0)
"""


def _clean_tracked_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of `os.environ` stripped of tracked prefixes plus optional `extra` overrides."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(DRIVER.TRACKED_ENV_PREFIXES)}
    env["TMPDIR"] = "/tmp"
    if extra:
        env.update(extra)
    return env


def _symlink_python_dir(bin_dir: Path) -> None:
    """Ensure `python3` in `bin_dir` resolves to `sys.executable` (Python >= 3.10) for wrapper tests."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    py_link = bin_dir / "python3"
    if not py_link.exists():
        os.symlink(sys.executable, py_link)


# --------------------------------------------------------------------------- 1. CLI & thin wrapper


def test_thin_wrapper_bash_32_syntax_and_help() -> None:
    bash_bin = "/bin/bash" if os.path.isfile("/bin/bash") else shutil.which("bash")
    assert bash_bin is not None
    chk = subprocess.run([bash_bin, "-n", str(SCRIPT_SH)], check=False, capture_output=True, text=True, timeout=15)
    assert chk.returncode == 0, chk.stderr

    hlp = subprocess.run([bash_bin, str(SCRIPT_SH), "--help"], check=False, capture_output=True, text=True, timeout=15)
    assert hlp.returncode == 0, hlp.stderr
    assert "Usage: bash scripts/multi_gpu_check.sh [options]" in hlp.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["--unknown-flag"],
        ["--backend"],
        ["--backend", "mpi"],
        ["--backend", "gloo", "--shared-gpu"],
        ["--reference-mode", "bad"],
        ["--pg-timeout", "0"],
        ["--pg-timeout", "-1"],
        ["--pg-timeout", "nan"],
        ["--pg-timeout", "5\n"],
        ["--step-timeout", "inf"],
        ["--step-timeout", "5\n"],
        ["--stage-timeout", "-inf"],
        ["--stage-timeout", "5\n"],
        ["--stage-timeout", "1" + "0" * 400],
        ["--ranks"],
        ["--ranks", ","],
        ["--ranks", "1"],
        ["--ranks", "abc"],
        ["--ranks", "\u00b2"],
        ["--ranks", "\u0662"],
        ["--sizes"],
        ["--sizes", ","],
        ["--sizes", "*"],
        ["--sizes", "999M"],
    ],
)
def test_parse_args_rejects_invalid_inputs(argv: list[str]) -> None:
    with pytest.raises(DRIVER.CliError):
        DRIVER.parse_args(argv)


def test_parse_args_accepts_frozen_cli_forms() -> None:
    cfg = DRIVER.parse_args(
        [
            "--backend",
            "nccl",
            "--shared-gpu",
            "--ranks",
            "2",
            "4",
            "--sizes",
            "5M,28M",
            "--python",
            "3.12",
            "--log-dir",
            "/tmp/custom_logs",
            "--force",
            "--reference-mode",
            "sketch",
            "--pg-timeout",
            "15",
            "--step-timeout",
            "120",
            "--stage-timeout",
            "45.5",
            "--only",
            "domain_head_only_f32",
            "--skip",
            "workers_uneven_f32",
            "--only",
            "export_multirank_f32",
        ]
    )
    assert cfg.backend == "nccl"
    assert cfg.shared_gpu is True
    assert cfg.device_map == "shared"
    assert cfg.ranks == (2, 4)
    assert cfg.ranks_specified is True
    assert cfg.sizes == ("5M", "28M")
    assert cfg.uv_py_args == ("--python", "3.12")
    assert cfg.force is True
    assert cfg.reference_mode == "sketch"
    assert cfg.stage_timeout == pytest.approx(45.5)
    assert cfg.extra_driver_args == (
        "--only",
        "domain_head_only_f32",
        "--skip",
        "workers_uneven_f32",
        "--only",
        "export_multirank_f32",
    )

    cfg_skip_sizes = DRIVER.parse_args(["--backend", "nccl", "--sizes", ""])
    assert cfg_skip_sizes.sizes == ()

    cfg_gloo_default = DRIVER.parse_args(["--backend", "gloo"])
    assert cfg_gloo_default.sizes == ()
    assert cfg_gloo_default.ranks_specified is False


# --------------------------------------------------------------------------- 2. stage signature


@pytest.mark.parametrize("prefix", DRIVER.TRACKED_ENV_PREFIXES)
def test_stage_signature_invalidates_every_tracked_env_prefix(prefix: str) -> None:
    base_env: dict[str, str] = {}
    base = DRIVER.stage_signature("git", "a" * 64, "rt", ["04_gloo_tiny_n2", "arg1"], env=base_env)
    alt = DRIVER.stage_signature("git", "a" * 64, "rt", ["04_gloo_tiny_n2", "arg1"], env={f"{prefix}TEST": "1"})
    assert alt != base


def test_stage_signature_ignores_unrelated_env_and_separates_args_from_env() -> None:
    base = DRIVER.stage_signature("git", "a" * 64, "rt", ["st1"], env={})
    unrelated = DRIVER.stage_signature("git", "a" * 64, "rt", ["st1"], env={"UNRELATED_VAR": "xyz"})
    assert unrelated == base

    as_arg = DRIVER.stage_signature("git", "a" * 64, "rt", ["st1", "NCCL_DEBUG=INFO"], env={})
    as_env = DRIVER.stage_signature("git", "a" * 64, "rt", ["st1"], env={"NCCL_DEBUG": "INFO"})
    with_sep_arg = DRIVER.stage_signature("git", "a" * 64, "rt", ["st1", "--env--", "NCCL_DEBUG=INFO"], env={})
    assert len({as_arg, as_env, with_sep_arg}) == 3


# --------------------------------------------------------------------------- 3. tree-state signature


def test_tree_state_sig_git_checkout_edits_untracked_and_exclusions(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=True)
    log_dir = repo / "multi_gpu_check_logs"
    log_dir.mkdir()

    method1, sig1 = DRIVER.compute_tree_state_sig(repo, log_dir)
    assert method1 == "git"
    assert len(sig1) == 64

    # Logs inside LOG_DIR and bytecode caches under scripts/ must not invalidate.
    (log_dir / "01_env.log").write_text("log data\n", encoding="utf-8")
    cache_dir = repo / "scripts" / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "multi_gpu_check.cpython-312.pyc").write_bytes(b"\x00\x01")
    method_same, sig_same = DRIVER.compute_tree_state_sig(repo, log_dir)
    assert (method_same, sig_same) == (method1, sig1)

    # Uncommitted edit to tracked file invalidates.
    helper = repo / "scripts" / "probes" / "_dist_common.py"
    helper.write_text("# modified\n", encoding="utf-8")
    method2, sig2 = DRIVER.compute_tree_state_sig(repo, log_dir)
    assert method2 == "git" and sig2 != sig1

    # Untracked file under scripts/ also invalidates.
    (repo / "scripts" / "new_probe.py").write_text("x = 1\n", encoding="utf-8")
    method3, sig3 = DRIVER.compute_tree_state_sig(repo, log_dir)
    assert method3 == "git" and sig3 != sig2 and sig3 != sig1


def test_tree_state_sig_no_git_tarball_nested_parent_and_missing_git_binary(tmp_path: Path) -> None:
    # Exported tarball with no .git directory at all.
    src_repo = _make_stub_repo(tmp_path / "src_repo", with_git=True)
    tar_path = tmp_path / "export.tar"
    with tarfile.open(tar_path, "w") as tf:
        for item in ("scripts", "src", "pyproject.toml", "uv.lock"):
            tf.add(src_repo / item, arcname=item)
    export_dir = tmp_path / "exported"
    export_dir.mkdir()
    with tarfile.open(tar_path, "r") as tf:
        if sys.version_info >= (3, 12):
            tf.extractall(export_dir, filter="data")
        else:
            tf.extractall(export_dir)
    assert not (export_dir / ".git").exists()

    log_dir = export_dir / "logs"
    log_dir.mkdir()
    m1, s1 = DRIVER.compute_tree_state_sig(export_dir, log_dir)
    assert m1 == "content_hash" and len(s1) == 64

    # Writing inside log_dir or __pycache__ does not change the hash.
    (log_dir / "summary.txt").write_text("hi\n", encoding="utf-8")
    (export_dir / "src" / "__pycache__").mkdir()
    (export_dir / "src" / "__pycache__" / "x.pyc").write_bytes(b"abc")
    assert DRIVER.compute_tree_state_sig(export_dir, log_dir) == (m1, s1)

    # Editing scripts/ and src/ each change the hash.
    (export_dir / "scripts" / "probes" / "_dist_common.py").write_text("# edited\n", encoding="utf-8")
    m2, s2 = DRIVER.compute_tree_state_sig(export_dir, log_dir)
    assert m2 == "content_hash" and s2 != s1
    (export_dir / "src" / "timestral_tune" / "config.py").write_text("VALUE = 2\n", encoding="utf-8")
    m3, s3 = DRIVER.compute_tree_state_sig(export_dir, log_dir)
    assert m3 == "content_hash" and s3 != s2 and s3 != s1

    # Exported tree nested inside an unrelated parent git repository must use content_hash, not the parent repo.
    parent_git = tmp_path / "parent_git"
    parent_git.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=parent_git, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=parent_git, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=parent_git, check=True)
    (parent_git / "README.md").write_text("parent\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=parent_git, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "parent"], cwd=parent_git, check=True)
    nested = parent_git / "nested_export"
    shutil.copytree(export_dir, nested)
    nested_log = nested / "logs"
    mn, sn = DRIVER.compute_tree_state_sig(nested, nested_log)
    assert mn == "content_hash" and sn == s3

    # `git` absent from PATH even inside a real git checkout must fall back cleanly to content_hash.
    m_nogit, s_nogit = DRIVER.compute_tree_state_sig(src_repo, src_repo / "logs", env={"PATH": ""})
    assert m_nogit == "content_hash" and len(s_nogit) == 64

    # Empty directory with no source files must fail fast (never return empty/constant signature).
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match="no source files"):
        DRIVER.compute_tree_state_sig(empty_dir, empty_dir / "logs", env={"PATH": ""})


# --------------------------------------------------------------------------- 4. reference archive verification


def test_verify_and_hash_ref_archive_valid_and_corrupted(tmp_path: Path) -> None:
    valid_zip = _make_valid_ref_zip(tmp_path / "valid.zip")
    sha = DRIVER.verify_and_hash_ref_archive(valid_zip)
    assert sha is not None and len(sha) == 64

    # Single-byte flip inside stored tensor member (same byte length, bad CRC-32).
    flipped = _flip_zip_member_byte(valid_zip, tmp_path / "flipped.zip")
    assert DRIVER.verify_and_hash_ref_archive(flipped) is None

    # Empty file and non-zip file.
    empty = tmp_path / "empty.zip"
    empty.write_bytes(b"")
    assert DRIVER.verify_and_hash_ref_archive(empty) is None
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip file")
    assert DRIVER.verify_and_hash_ref_archive(junk) is None

    # Wrong tensor byte length in metadata.
    bad_len = tmp_path / "bad_len.zip"
    with zipfile.ZipFile(bad_len, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", b"\x00" * 8)
        zf.writestr("spec.json", json.dumps({"spec": {"schema_version": 5}}).encode())
        zf.writestr(
            "entries.json",
            json.dumps(
                {"k": {"tensor_member": "tensors/0000.bin", "tensor_dtype": "float32", "tensor_shape": [10]}}
            ).encode(),
        )
    assert DRIVER.verify_and_hash_ref_archive(bad_len) is None


# --------------------------------------------------------------------------- 5. resume & failed-ref skip


def test_resume_invalidation_and_failed_ref_skip(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=True)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    call_log = tmp_path / "uv_calls.log"
    _write_stub_uv(bin_dir, _standard_stub_uv_body(call_log=call_log))
    log_dir = tmp_path / "logs"

    def _run(
        extra_args: list[str] | None = None, extra_env: Mapping[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = _clean_tracked_env(extra_env)
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        cmd = [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            "--ranks",
            "2",
            "--log-dir",
            str(log_dir),
            *(extra_args or []),
        ]
        return subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=30)

    # 1. Fresh run executes all 4 runnable stages.
    r1 = _run()
    assert r1.returncode == 0, r1.stderr
    s1 = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "01_env" in s1 and "(resumed)" not in s1
    assert "device_map=" not in s1
    assert "tree_state=git:" in (log_dir / "01_env.log").read_text(encoding="utf-8").splitlines()[0]

    # 2. Immediate resume with unrelated env var repeats zero stages.
    call_log.unlink()
    r2 = _run(extra_env={"UNRELATED_FOO": "bar"})
    assert r2.returncode == 0, r2.stderr
    s2 = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert s2.count("0 (resumed)") == 4
    # Only compute_runtime_env_sig runs; no stage execution (`sync`, `--write-reference`, `torchrun`, rank_failure).
    calls_text = call_log.read_text(encoding="utf-8") if call_log.exists() else ""
    assert "--write-reference" not in calls_text and "torchrun" not in calls_text

    # 3. Changing a tracked GLOO_* env variable invalidates every stage.
    r3 = _run(extra_env={"GLOO_TEST_FLAG": "1"})
    assert r3.returncode == 0, r3.stderr
    assert "(resumed)" not in (log_dir / "summary.txt").read_text(encoding="utf-8")

    # 4. Re-establish clean resume without GLOO_TEST_FLAG, then edit _dist_common.py -> invalidates all stages.
    assert _run().returncode == 0
    assert (log_dir / "summary.txt").read_text(encoding="utf-8").count("0 (resumed)") == 0
    assert _run().returncode == 0
    assert (log_dir / "summary.txt").read_text(encoding="utf-8").count("0 (resumed)") == 4
    (repo / "scripts" / "probes" / "_dist_common.py").write_text("# uncommitted edit\n", encoding="utf-8")
    r4 = _run()
    assert r4.returncode == 0, r4.stderr
    assert "(resumed)" not in (log_dir / "summary.txt").read_text(encoding="utf-8")

    # 5. Corrupting the reference zip invalidates 03_ref and 04 while 01_env and 09 resume.
    ref_zip = log_dir / "refs" / "ref_gloo_tiny_n2.zip"
    _flip_zip_member_byte(ref_zip, ref_zip)
    r5 = _run()
    assert r5.returncode == 0, r5.stderr
    s5 = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert s5.count("0 (resumed)") == 2
    assert "03_ref_gloo_tiny_n2                  0 " in s5
    assert "04_gloo_tiny_n2                      0 " in s5

    # 6. Failed reference stage deletes stale artifact, skips dependent 04, continues to 09, exits 1.
    _write_stub_uv(bin_dir, _standard_stub_uv_body(fail_ref_n={2}))
    r6 = _run(extra_args=["--force"])
    assert r6.returncode == 1, r6.stderr
    s6 = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "03_ref_gloo_tiny_n2                  1 " in s6
    assert "04_gloo_tiny_n2                      SKIPPED (ref failed)" in s6
    assert "09_rank_failure_gloo_n2              0 " in s6
    assert not ref_zip.exists()

    # 7. Reference stage exiting 0 with a corrupt artifact records `1 (corrupt artifact)` and skips 04.
    _write_stub_uv(bin_dir, _standard_stub_uv_body(corrupt_ref_n={2}))
    r7 = _run(extra_args=["--force"])
    assert r7.returncode == 1, r7.stderr
    s7 = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "03_ref_gloo_tiny_n2                  1 (corrupt artifact)" in s7
    assert "04_gloo_tiny_n2                      SKIPPED (ref failed)" in s7
    assert not ref_zip.exists()


def test_session_env_defaults_and_nccl_stage_ordering(tmp_path: Path) -> None:
    # TMPDIR > 60 chars collapses to /tmp; shared_gpu exports NCCL_MULTI_RANK_GPU_ENABLE=1.
    long_tmp = "/" + "a" * 70
    env_shared = DRIVER._prepare_session_env({"TMPDIR": long_tmp}, shared_gpu=True)
    assert env_shared["TMPDIR"] == "/tmp"
    assert env_shared["PYTHONFAULTHANDLER"] == "1"
    assert env_shared["NCCL_MULTI_RANK_GPU_ENABLE"] == "1"

    env_short = DRIVER._prepare_session_env({"TMPDIR": "/tmp/ok"}, shared_gpu=False)
    assert env_short["TMPDIR"] == "/tmp/ok"
    assert "NCCL_MULTI_RANK_GPU_ENABLE" not in env_short

    # Verify full 4-GPU NCCL stage names/ordering plus prototype skip reasons.
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    _write_stub_uv(bin_dir, _standard_stub_uv_body(gpu_count=4, cuda_available=True))

    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})
    ld_4gpu = tmp_path / "logs_4gpu"
    r_full = subprocess.run(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "nccl",
            "--sizes",
            "5M",
            "--log-dir",
            str(ld_4gpu),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r_full.returncode == 0, r_full.stderr
    s_full = (ld_4gpu / "summary.txt").read_text(encoding="utf-8")
    assert "backend=nccl, device_map=per-rank, ranks=2 4" in s_full
    expected_stages = [
        "01_env",
        "02_prototype_parity_cuda",
        "03_ref_nccl_tiny_n2",
        "04_nccl_tiny_n2",
        "03_ref_nccl_tiny_n4",
        "04_nccl_tiny_n4",
        "05_ref_nccl_released_n2",
        "06_nccl_released_n2",
        "05_ref_nccl_released_n4",
        "06_nccl_released_n4",
        "07_ref_gloo_tiny_n2",
        "08_gloo_tiny_n2",
        "07_ref_gloo_tiny_n4",
        "08_gloo_tiny_n4",
        "09_rank_failure_gloo_n2",
        "09_rank_failure_gloo_n4",
        "10_rank_failure_nccl_n2",
        "10_rank_failure_nccl_n4",
    ]
    observed = [ln.split()[0] for ln in s_full.splitlines() if ln and ln[0].isdigit()]
    assert observed == expected_stages

    # Prototype skip reasons: shared GPU, needs 4 GPUs (2 GPUs), rank 4 not requested (4 GPUs, --ranks 2).
    for label, gpus, extra, reason in [
        ("shared", 1, ["--shared-gpu", "--sizes", ""], "shared GPU"),
        ("2gpu", 2, ["--sizes", ""], "needs 4 GPUs"),
        ("no_r4", 4, ["--ranks", "2", "--sizes", ""], "rank 4 not requested"),
    ]:
        _write_stub_uv(bin_dir, _standard_stub_uv_body(gpu_count=gpus, cuda_available=True))
        ld_case = tmp_path / f"logs_{label}"
        r_case = subprocess.run(
            [
                "/bin/bash",
                str(repo / "scripts" / "multi_gpu_check.sh"),
                "--backend",
                "nccl",
                "--log-dir",
                str(ld_case),
                *extra,
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert r_case.returncode == 0, (label, r_case.stderr)
        assert f"02_prototype_parity_cuda             SKIPPED ({reason})" in (ld_case / "summary.txt").read_text(
            encoding="utf-8"
        )


# --------------------------------------------------------------------------- 6. stage timeout & tree kill


def test_stage_timeout_kills_sigterm_ignoring_child_and_new_session_grandchild(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    tag = f"tt_pytest_timeout_{os.getpid()}"
    grandchild_ready = tmp_path / "grandchild.ready"
    reparented_ready = tmp_path / "reparented.ready"
    reparented_pidfile = tmp_path / "reparented.pid"
    dump_sentinel = "SHUTDOWN_DUMP_COMPLETE_SENTINEL"

    stub_body = f"""\
import os, pathlib, signal, subprocess, sys, time

argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync":
    sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined:
    print("stub-rt")
    sys.exit(0)
if "tree_state_method=" in joined:
    print("tree_state_method=content_hash tree_state_sig=000000000000")
    sys.exit(0)
if "--write-reference" in argv:
    # 1. New-session grandchild still parented by the stage process, ignoring SIGTERM.
    gc_code = (
        "import pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(grandchild_ready)!r}).write_text('ready'); "
        "time.sleep(60)"
    )
    subprocess.Popen([sys.executable, "-c", gc_code, {tag!r}], start_new_session=True)
    # 2. Truly reparented new-session worker (intermediate parent exits immediately so PPID leaves stage tree),
    #    reachable only via the inherited `_TIMESTRAL_STAGE_TOKEN` scan.
    rep_worker_code = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(reparented_pidfile)!r}).write_text(str(os.getpid())); "
        f"pathlib.Path({str(reparented_ready)!r}).write_text('ready'); "
        "time.sleep(60)"
    )
    helper_code = (
        "import os, subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {{rep_worker_code!r}}], start_new_session=True); "
        "os._exit(0)"
    )
    helper = subprocess.Popen([sys.executable, "-c", helper_code])
    helper.wait()
    # Parent ignores SIGTERM after dumping >128 KiB to stdout (larger than OS pipe buffer).
    def _on_term(_sig, _frame):
        sys.stdout.write(("X" * 1024 + "\\n") * 160 + {dump_sentinel!r} + "\\n")
        sys.stdout.flush()
    signal.signal(signal.SIGTERM, _on_term)
    while not (pathlib.Path({str(grandchild_ready)!r}).exists() and pathlib.Path({str(reparented_ready)!r}).exists()):
        time.sleep(0.01)
    time.sleep(60)
    sys.exit(0)
if "distributed_rank_failure_check.py" in joined:
    print("result: PASS (no hang)")
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    log_dir = tmp_path / "logs"
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "_TT_TEST_LEFTOVER_TAG": tag})

    proc = subprocess.Popen(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            "--ranks",
            "2",
            "--stage-timeout",
            "1",
            "--log-dir",
            str(log_dir),
        ],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=30)
        assert proc.returncode == 1, out
        assert grandchild_ready.exists(), "grandchild never started"
        assert reparented_ready.exists(), "reparented worker never started"
        rep_pid = int(reparented_pidfile.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(rep_pid, 0)
        assert _list_matching_pids(tag) == []
    finally:
        _kill_process_group_quiet(proc)
        if reparented_pidfile.exists():
            try:
                os.kill(int(reparented_pidfile.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
        _cleanup_tagged_processes(tag)

    summary = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "03_ref_gloo_tiny_n2                  124 " in summary
    assert "04_gloo_tiny_n2                      SKIPPED (ref failed)" in summary
    assert "09_rank_failure_gloo_n2              0 " in summary
    ref_log = (log_dir / "03_ref_gloo_tiny_n2.log").read_text(encoding="utf-8")
    assert dump_sentinel in ref_log
    assert "timed out after 1s" in ref_log
    assert not (log_dir / "03_ref_gloo_tiny_n2.done").exists()
    assert not (log_dir / "refs" / "ref_gloo_tiny_n2.zip").exists()


# --------------------------------------------------------------------------- 7. early exits write summary


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_failed_env_stage_aborts_and_writes_summary(tmp_path: Path, backend: str) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    _write_stub_uv(bin_dir, _standard_stub_uv_body(env_sync_rc=7))
    log_dir = tmp_path / f"logs_{backend}"
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})

    r = subprocess.run(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            backend,
            "--ranks",
            "2",
            "--log-dir",
            str(log_dir),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 7, r.stderr
    summary = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "01_env                               7 " in summary
    assert "03_ref" not in summary and "09_rank_failure" not in summary


@pytest.mark.parametrize(
    ("gpu_count", "extra_args", "expected_err"),
    [
        (0, [], "reported 0 visible CUDA GPUs"),
        (1, [], "only 1 visible GPU detected"),
        (1, ["--ranks", "2"], "requested --ranks 2 with per-rank NCCL"),
    ],
)
def test_gpu_count_early_exits_write_summary(
    tmp_path: Path,
    gpu_count: int,
    extra_args: list[str],
    expected_err: str,
) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    _write_stub_uv(bin_dir, _standard_stub_uv_body(gpu_count=gpu_count, cuda_available=True))
    log_dir = tmp_path / "logs"
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})

    r = subprocess.run(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "nccl",
            "--log-dir",
            str(log_dir),
            *extra_args,
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert expected_err in r.stderr
    summary_file = log_dir / "summary.txt"
    assert summary_file.is_file()
    summary = summary_file.read_text(encoding="utf-8")
    assert "device_map=per-rank" in summary
    assert "01_env                               0 " in summary
    assert "03_ref" not in summary


# --------------------------------------------------------------------------- 8. signal matrix (default disposition)


def _run_signal_scenario(
    tmp_path: Path,
    *,
    sig: signal.Signals,
    ignored_on_entry: bool,
    moment: str,
) -> tuple[int, Path, Path, str]:
    """Run one deterministic signal-matrix cell through `bash scripts/multi_gpu_check.sh`."""
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    tag = f"tt_sig_{moment}_{sig.name}_{os.getpid()}"
    log_dir = tmp_path / f"logs_{tag}"
    ready_file = tmp_path / "moment.ready"
    go_file = tmp_path / "signal_sent.go"
    later_stage_file = tmp_path / "later_stage.started"
    rt_counter = tmp_path / "rt_calls.txt"

    stub_body = f"""\
import json, pathlib, signal, sys, time, uuid, zipfile

argv = sys.argv[1:]
joined = " ".join(argv)
moment = {moment!r}
ready = pathlib.Path({str(ready_file)!r})
go = pathlib.Path({str(go_file)!r})
later = pathlib.Path({str(later_stage_file)!r})
rt_counter = pathlib.Path({str(rt_counter)!r})

def _wait_go():
    deadline = time.monotonic() + 20.0
    while not go.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

def _write_ref(ref_path):
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    spec_doc = {{"spec": {{"schema_version": 5}}, "environment": {{"archive_id": uuid.uuid4().hex}}}}
    entries_doc = {{
        "tiny_random:s": {{
            "tensor_member": "tensors/0000.bin",
            "tensor_dtype": "float32",
            "tensor_shape": [4],
        }}
    }}
    with zipfile.ZipFile(ref_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", b"\\x00" * 16)
        zf.writestr("spec.json", json.dumps(spec_doc).encode())
        zf.writestr("entries.json", json.dumps(entries_doc).encode())

if argv and argv[0] == "sync":
    sys.exit(0)

if "sys.executable" in joined and "torch.version.cuda" in joined:
    n = int(rt_counter.read_text()) + 1 if rt_counter.exists() else 1
    rt_counter.write_text(str(n))
    if (moment == "startup" and n == 1) or (moment == "between_stages" and n == 2):
        ready.write_text("ready")
        _wait_go()
    print("stub-rt-sig")
    sys.exit(0)

if "tree_state_method=" in joined:
    print("tree_state_method=content_hash tree_state_sig=000000000000")
    sys.exit(0)

if "--write-reference" in argv:
    ref_path = pathlib.Path(argv[argv.index("--write-reference") + 1])
    ws = int(argv[argv.index("--world-size") + 1])
    if ws == 2:
        if moment == "mid_stage":
            # Verify POSIX SIG_IGN inheritance inside the child stage itself when ignored on entry.
            if {ignored_on_entry!r}:
                assert signal.getsignal(signal.{sig.name}) == signal.SIG_IGN
            # Write both a partial archive and a partial .tmp file so interrupt cleanup is verified on both.
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_bytes(b"partial-zip")
            pathlib.Path(f"{{ref_path}}.tmp").write_bytes(b"partial")
            ready.write_text("ready")
            _wait_go()
            pathlib.Path(f"{{ref_path}}.tmp").unlink(missing_ok=True)
            _write_ref(ref_path)
            sys.exit(0)
        if moment == "timeout_teardown":
            # Catch SIGTERM so the stub announces entry into timeout teardown and stays alive until SIGKILL.
            def _on_term(_s, _f):
                ready.write_text("teardown")
            signal.signal(signal.SIGTERM, _on_term)
            time.sleep(30)
            sys.exit(0)
    if ws == 4:
        later.write_text("started")
    _write_ref(ref_path)
    sys.exit(0)

if "torchrun" in argv:
    if moment in ("between_stages", "mid_stage", "startup"):
        later.write_text("started")
    sys.exit(0)

if "distributed_rank_failure_check.py" in joined:
    later.write_text("started")
    print("result: PASS (no hang)")
    sys.exit(0)

sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})
    # Tag environment so leftover-process check matches any orphaned child from this test.
    env["_TT_TEST_LEFTOVER_TAG"] = tag

    stage_timeout = "0.3" if moment == "timeout_teardown" else "30"
    ranks_args = ["--ranks", "2", "4"] if moment == "timeout_teardown" else ["--ranks", "2"]

    def _preexec() -> None:
        os.setsid()
        for managed in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            if managed == sig and ignored_on_entry:
                signal.signal(managed, signal.SIG_IGN)
            else:
                signal.signal(managed, signal.SIG_DFL)

    proc = subprocess.Popen(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            *ranks_args,
            "--stage-timeout",
            stage_timeout,
            "--log-dir",
            str(log_dir),
        ],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=_preexec,
    )
    try:
        _wait_for_path(ready_file, proc, timeout=20.0)
        os.killpg(proc.pid, sig)
        if ignored_on_entry and moment != "timeout_teardown":
            # Give the signal time to be delivered before releasing the stub stage.
            time.sleep(0.1)
            go_file.write_text("go", encoding="utf-8")
        out, _ = proc.communicate(timeout=30.0)
        rc = proc.returncode
        leftovers = _list_matching_pids(tag)
        assert leftovers == [], f"leftover processes after {moment}/{sig.name}: {leftovers}"
    finally:
        _kill_process_group_quiet(proc)
        _cleanup_tagged_processes(tag)

    return rc, log_dir, later_stage_file, out


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP], ids=["INT", "TERM", "HUP"])
@pytest.mark.parametrize("moment", ["startup", "mid_stage", "timeout_teardown", "between_stages"])
def test_signal_default_disposition_interrupts_cleanly(
    tmp_path: Path,
    sig: signal.Signals,
    moment: str,
) -> None:
    rc, log_dir, later_stage, out = _run_signal_scenario(tmp_path, sig=sig, ignored_on_entry=False, moment=moment)
    expected_rc = 128 + int(sig)
    assert rc == expected_rc, f"expected {expected_rc}, got {rc}\n{out}"

    summary = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert f"INTERRUPTED by {sig.name} (exit {expected_rc})" in summary
    assert f"Session interrupted by {sig.name} (exit {expected_rc}); remaining stages were not started." in summary
    assert not later_stage.exists(), "a subsequent stage ran after interrupt"

    if moment == "startup":
        assert "01_env" not in summary
    elif moment == "between_stages":
        assert "01_env                               0 " in summary
        assert "03_ref" not in summary
    else:
        assert f"{expected_rc} (interrupted:{sig.name})" in summary
        assert not (log_dir / "03_ref_gloo_tiny_n2.done").exists()
        assert not (log_dir / "refs" / "ref_gloo_tiny_n2.zip").exists()
        assert not (log_dir / "refs" / "ref_gloo_tiny_n2.zip.tmp").exists()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP], ids=["INT", "TERM", "HUP"])
@pytest.mark.parametrize("moment", ["startup", "mid_stage", "timeout_teardown", "between_stages"])
def test_signal_ignored_on_entry_is_preserved(
    tmp_path: Path,
    sig: signal.Signals,
    moment: str,
) -> None:
    rc, log_dir, later_stage, out = _run_signal_scenario(tmp_path, sig=sig, ignored_on_entry=True, moment=moment)
    summary = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "INTERRUPTED" not in summary, summary
    assert later_stage.exists(), f"subsequent stage did not run under ignored {sig.name}:\n{out}"

    if moment == "timeout_teardown":
        # Ignored signal must not turn a stage timeout into an interrupt; session continues to N=4 and exits 1.
        assert rc == 1, f"expected 1 (stage timeout only), got {rc}\n{out}"
        assert "03_ref_gloo_tiny_n2                  124 " in summary
        assert "04_gloo_tiny_n2                      SKIPPED (ref failed)" in summary
        assert "03_ref_gloo_tiny_n4                  0 " in summary
        assert "04_gloo_tiny_n4                      0 " in summary
        assert "09_rank_failure_gloo_n2              0 " in summary
        assert "09_rank_failure_gloo_n4              0 " in summary
    else:
        assert rc == 0, f"expected 0 under ignored {sig.name}, got {rc}\n{out}"
        assert "01_env                               0 " in summary
        assert "03_ref_gloo_tiny_n2                  0 " in summary
        assert "04_gloo_tiny_n2                      0 " in summary
        assert "09_rank_failure_gloo_n2              0 " in summary
        assert (log_dir / "03_ref_gloo_tiny_n2.done").is_file()
        assert (log_dir / "refs" / "ref_gloo_tiny_n2.zip").is_file()


# --------------------------------------------------------------------------- 9. stage runner & process-tree hardening


@pytest.mark.parametrize("fail_step", ["sync", "probe"])
def test_env_stage_bootstrap_maps_signal_kill_to_posix_exit_code(tmp_path: Path, fail_step: str) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    stub_body = f"""\
import os, signal, sys

argv = sys.argv[1:]
joined = " ".join(argv)
if "sys.executable" in joined and "torch.version.cuda" in joined:
    print("stub-rt")
    sys.exit(0)
if argv and argv[0] == "sync":
    if {fail_step!r} == "sync":
        os.kill(os.getpid(), signal.SIGSEGV)
    sys.exit(0)
if "tree_state_method=" in joined:
    if {fail_step!r} == "probe":
        os.kill(os.getpid(), signal.SIGSEGV)
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    log_dir = tmp_path / f"logs_{fail_step}"
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})
    r = subprocess.run(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            "--ranks",
            "2",
            "--log-dir",
            str(log_dir),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 139, (r.returncode, r.stderr)
    summary = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "01_env                               139 " in summary


def test_stage_process_closed_stdio_does_not_busy_spin(tmp_path: Path) -> None:
    cfg = DRIVER.SessionConfig(backend="gloo", log_dir=tmp_path / "logs", stage_timeout=5.0)
    runner = DRIVER.SessionRunner(config=cfg, repo_root=tmp_path, env=_clean_tracked_env())
    runner.log_dir.mkdir(parents=True, exist_ok=True)
    runner.tree_method, runner.tree_sig = "content_hash", "0" * 64

    cpu0 = time.process_time()
    rc, int_sig = runner.run_stage_process(
        "closed_stdio",
        runner.log_dir / "closed_stdio.log",
        [sys.executable, "-c", "import os, time; os.close(1); os.close(2); time.sleep(0.45)"],
    )
    cpu_dt = time.process_time() - cpu0
    assert (rc, int_sig) == (0, None)
    assert cpu_dt < 0.20, f"busy-spin detected on closed stdio: cpu_dt={cpu_dt:.3f}s"


def test_post_exit_survivor_descendant_is_killed_and_logged(tmp_path: Path) -> None:
    cfg = DRIVER.SessionConfig(backend="gloo", log_dir=tmp_path / "logs", stage_timeout=5.0)
    runner = DRIVER.SessionRunner(
        config=cfg, repo_root=tmp_path, env=_clean_tracked_env(), teardown_grace_s=0.4, teardown_reap_s=1.0
    )
    runner.log_dir.mkdir(parents=True, exist_ok=True)
    runner.tree_method, runner.tree_sig = "content_hash", "0" * 64

    pidfile = tmp_path / "survivor.pid"
    stage_code = (
        "import subprocess, sys; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True); "
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))"
    )
    log_file = runner.log_dir / "survivor_stage.log"
    rc, int_sig = runner.run_stage_process("survivor_stage", log_file, [sys.executable, "-c", stage_code])
    assert (rc, int_sig) == (0, None)

    survivor_pid = int(pidfile.read_text(encoding="utf-8").strip())
    alive = True
    try:
        os.kill(survivor_pid, 0)
    except OSError:
        alive = False
    finally:
        if alive:
            try:
                os.kill(survivor_pid, signal.SIGKILL)
            except OSError:
                pass
    assert not alive, f"surviving descendant pid={survivor_pid} outlived finished stage"
    log_text = log_file.read_text(encoding="utf-8")
    assert "surviving descendants" in log_text
    assert "signalled 1 pid(s)" in log_text


@pytest.mark.parametrize("leader_rc", [0, 5])
def test_post_exit_open_pipe_survivor_returns_leader_exit_code_with_warning(tmp_path: Path, leader_rc: int) -> None:
    cfg = DRIVER.SessionConfig(backend="gloo", sizes=(), log_dir=tmp_path / "logs", stage_timeout=20.0)
    runner = DRIVER.SessionRunner(
        config=cfg,
        repo_root=tmp_path,
        env=_clean_tracked_env(),
        post_exit_eof_grace_s=0.3,
        teardown_grace_s=0.3,
        teardown_reap_s=0.5,
    )
    runner.log_dir.mkdir(parents=True, exist_ok=True)
    runner.tree_method, runner.tree_sig = "content_hash", "0" * 64

    pidfile = tmp_path / f"open_pipe_{leader_rc}.pid"
    ready = tmp_path / f"open_pipe_{leader_rc}.ready"
    child_code = f"import pathlib, time; pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(60)"
    stage_code = (
        "import pathlib, subprocess, sys, time; "
        f"p = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); "
        f"deadline = time.monotonic() + 10.0\n"
        f"while not pathlib.Path({str(ready)!r}).exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        "print('leader done', flush=True); "
        f"sys.exit({leader_rc})"
    )
    stage_name = f"open_pipe_rc{leader_rc}"
    t0 = time.monotonic()
    try:
        outcome = runner.run_stage_with_sig(stage_name, "sig_open_pipe", None, [sys.executable, "-c", stage_code])
        dt = time.monotonic() - t0
        assert dt < 3.0, f"finished stage stalled waiting for stage_timeout: dt={dt:.2f}s"
        assert outcome.returncode == leader_rc
        assert outcome.ok is (leader_rc == 0)
        assert outcome.interrupted_sig is None
        assert ready.exists()
        child_pid = int(pidfile.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass

    log_text = (runner.log_dir / f"{stage_name}.log").read_text(encoding="utf-8")
    assert "leader done" in log_text
    assert "WARNING:" in log_text
    assert "surviving descendants" in log_text
    assert "signalled 1 pid(s)" in log_text
    assert "timed out" not in log_text
    assert (runner.log_dir / f"{stage_name}.done").exists() is (leader_rc == 0)


def test_post_exit_open_pipe_flushing_child_completes_without_warning_and_interrupt_wins(tmp_path: Path) -> None:
    # 1. A child that legitimately finishes flushing output within post_exit_eof_grace_s completes
    #    cleanly with all output captured and no spurious WARNING.
    cfg1 = DRIVER.SessionConfig(backend="gloo", sizes=(), log_dir=tmp_path / "logs_flush", stage_timeout=20.0)
    runner1 = DRIVER.SessionRunner(
        config=cfg1,
        repo_root=tmp_path,
        env=_clean_tracked_env(),
        post_exit_eof_grace_s=1.5,
        teardown_grace_s=0.3,
        teardown_reap_s=0.5,
    )
    runner1.log_dir.mkdir(parents=True, exist_ok=True)
    runner1.tree_method, runner1.tree_sig = "content_hash", "0" * 64

    flush_child = "import sys, time; time.sleep(0.1); print('POST_EXIT_FLUSH_SENTINEL', flush=True)"
    flush_stage = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {flush_child!r}]); sys.exit(0)"
    log1 = runner1.log_dir / "flush_stage.log"
    rc1, int_sig1 = runner1.run_stage_process("flush_stage", log1, [sys.executable, "-c", flush_stage])
    assert (rc1, int_sig1) == (0, None)
    text1 = log1.read_text(encoding="utf-8")
    assert "POST_EXIT_FLUSH_SENTINEL" in text1
    assert "WARNING:" not in text1

    # 2. A session interrupt arriving during the post-exit EOF grace window still wins over the
    #    leader's exit code and terminates the surviving descendant promptly.
    cfg2 = DRIVER.SessionConfig(backend="gloo", sizes=(), log_dir=tmp_path / "logs_int", stage_timeout=20.0)
    runner2 = DRIVER.SessionRunner(
        config=cfg2,
        repo_root=tmp_path,
        env=_clean_tracked_env(),
        post_exit_eof_grace_s=5.0,
        teardown_grace_s=0.3,
        teardown_reap_s=0.5,
    )
    runner2.log_dir.mkdir(parents=True, exist_ok=True)
    runner2.tree_method, runner2.tree_sig = "content_hash", "0" * 64

    child_pidfile = tmp_path / "int_grace_child.pid"
    child_ready = tmp_path / "int_grace_child.ready"
    leader_pidfile = tmp_path / "int_grace_leader.pid"
    child_code = f"import pathlib, time; pathlib.Path({str(child_ready)!r}).write_text('ready'); time.sleep(60)"
    stage_code = (
        "import os, pathlib, subprocess, sys, time; "
        f"pathlib.Path({str(leader_pidfile)!r}).write_text(str(os.getpid())); "
        f"p = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pidfile)!r}).write_text(str(p.pid)); "
        f"deadline = time.monotonic() + 10.0\n"
        f"while not pathlib.Path({str(child_ready)!r}).exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        "sys.exit(0)"
    )

    def _interrupt_during_grace() -> None:
        deadline = time.monotonic() + 10.0
        while not (child_ready.exists() and leader_pidfile.exists()) and time.monotonic() < deadline:
            time.sleep(0.01)
        lpid = int(leader_pidfile.read_text(encoding="utf-8").strip())
        # Wait until StageProcess._wait_until_exit_or_deadline() has reaped the exited leader
        # via proc.poll(), guaranteeing we are inside the post_exit_eof_grace_s window.
        while time.monotonic() < deadline:
            try:
                os.kill(lpid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        runner2.interrupted_sig_num = int(signal.SIGTERM)

    th = threading.Thread(target=_interrupt_during_grace, daemon=True)
    th.start()
    log2 = runner2.log_dir / "int_grace_stage.log"
    t0 = time.monotonic()
    try:
        rc2, int_sig2 = runner2.run_stage_process("int_grace_stage", log2, [sys.executable, "-c", stage_code])
        dt2 = time.monotonic() - t0
        th.join(timeout=5.0)
        assert (rc2, int_sig2) == (143, int(signal.SIGTERM))
        assert dt2 < 2.5, f"interrupt during post-exit EOF grace waited too long: dt={dt2:.2f}s"
        cpid = int(child_pidfile.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(cpid, 0)
    finally:
        th.join(timeout=2.0)
        if child_pidfile.exists():
            try:
                os.kill(int(child_pidfile.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
    assert "interrupted by SIGTERM" in log2.read_text(encoding="utf-8")


def test_find_token_pids_exact_matching_empty_proc_fallback_and_scan_order(tmp_path: Path) -> None:
    ps_bin = DRIVER._resolve_ps_bin()
    base_token = f"tt_exact_{os.getpid()}"
    exact_ready = tmp_path / "exact.ready"
    ext_ready = tmp_path / "ext.ready"
    p_exact = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import pathlib, time; pathlib.Path({str(exact_ready)!r}).write_text('ready'); time.sleep(15)",
        ],
        env=dict(os.environ, _TIMESTRAL_STAGE_TOKEN=base_token),
    )
    p_extended = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import pathlib, time; pathlib.Path({str(ext_ready)!r}).write_text('ready'); time.sleep(15)",
        ],
        env=dict(os.environ, _TIMESTRAL_STAGE_TOKEN=f"{base_token}_suffix"),
    )
    try:
        _wait_for_path(exact_ready, p_exact, timeout=10.0)
        _wait_for_path(ext_ready, p_extended, timeout=10.0)
        needle = f"_TIMESTRAL_STAGE_TOKEN={base_token}"
        hits = DRIVER.find_token_pids(needle, self_pid=os.getpid(), ps_bin=ps_bin)
        assert p_exact.pid in hits
        assert p_extended.pid not in hits

        # Simulate an empty /proc directory and ensure fallback to ps still finds the exact token.
        real_isdir = os.path.isdir
        real_listdir = os.listdir
        with (
            mock.patch.object(os.path, "isdir", side_effect=lambda p: True if str(p) == "/proc" else real_isdir(p)),
            mock.patch.object(os, "listdir", side_effect=lambda p: [] if str(p) == "/proc" else real_listdir(p)),
        ):
            fallback_hits = DRIVER.find_token_pids(needle, self_pid=os.getpid(), ps_bin=ps_bin)
            assert p_exact.pid in fallback_hits
            assert p_extended.pid not in fallback_hits
    finally:
        for p in (p_exact, p_extended):
            p.kill()
            p.wait()

    # Verify collect_live_targets scans tokens before reading the process table so a child forked in
    # between is still discovered via its parent in `children`.
    call_order: list[str] = []

    def _fake_find_tokens(*_a: Any, **_kw: Any) -> set[int]:
        call_order.append("token")
        return {1001}

    def _fake_proc_table(*_a: Any, **_kw: Any) -> tuple[dict[int, list[int]], dict[int, int]]:
        call_order.append("table")
        # Child 1002 was forked by 1001 between the token scan and the table read.
        return {1001: [1002]}, {1001: 1001, 1002: 1001}

    fake_proc = mock.MagicMock()
    fake_proc.poll.return_value = 0
    fake_proc.pid = 999
    with (
        mock.patch.object(DRIVER, "find_token_pids", side_effect=_fake_find_tokens),
        mock.patch.object(DRIVER, "read_proc_table", side_effect=_fake_proc_table),
    ):
        live_pids, live_pgids = DRIVER.collect_live_targets(
            fake_proc, "_TIMESTRAL_STAGE_TOKEN=t", self_pid=1, self_pgid=1, ps_bin=None
        )
    assert call_order == ["token", "table"]
    assert live_pids == {1001, 1002}
    assert live_pgids == {1001}


def test_missing_stage_binary_records_127_and_writes_summary(tmp_path: Path) -> None:
    cfg = DRIVER.SessionConfig(backend="gloo", log_dir=tmp_path / "logs", ranks=(2,), ranks_specified=True)
    runner = DRIVER.SessionRunner(config=cfg, repo_root=tmp_path, env=_clean_tracked_env())
    runner.log_dir.mkdir(parents=True, exist_ok=True)
    runner.tree_method, runner.tree_sig = "content_hash", "0" * 64

    outcome = runner.run_stage_with_sig("missing_bin_stage", "sig", None, ["/nonexistent_binary_for_stage_test"])
    assert not outcome.ok
    assert outcome.returncode == 127
    assert runner.any_failed == 1
    summary = runner.write_summary()
    assert "missing_bin_stage                    127 " in summary
    log_text = (runner.log_dir / "missing_bin_stage.log").read_text(encoding="utf-8")
    assert "failed to launch stage missing_bin_stage" in log_text


def test_unexpected_log_write_error_kills_stage_tree_and_writes_summary(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    pidfile = tmp_path / "enospc_stage.pid"
    stub_body = f"""\
import os, sys, time

argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync":
    sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined:
    print("stub-rt")
    sys.exit(0)
if "tree_state_method=" in joined:
    print("tree_state_method=content_hash tree_state_sig=000000000000")
    sys.exit(0)
if "--write-reference" in argv:
    open({str(pidfile)!r}, "w").write(str(os.getpid()))
    print("trigger log write failure", flush=True)
    time.sleep(30)
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    cfg = DRIVER.SessionConfig(
        backend="gloo", sizes=(), log_dir=tmp_path / "logs", ranks=(2,), ranks_specified=True, stage_timeout=10.0
    )
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})
    runner = DRIVER.SessionRunner(config=cfg, repo_root=repo, env=env, teardown_grace_s=0.4, teardown_reap_s=1.0)

    real_open = open

    class _FailingWriteFile:
        def __init__(self, f: Any) -> None:
            self._f = f
            self._writes = 0

        def write(self, data: bytes) -> int:
            self._writes += 1
            if self._writes >= 2:
                raise OSError(28, "No space left on device")
            return self._f.write(data)

        def flush(self) -> None:
            self._f.flush()

        def close(self) -> None:
            self._f.close()

    def _patched_open(path: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        f = real_open(path, mode, *args, **kwargs)
        if str(path).endswith("03_ref_gloo_tiny_n2.log") and "w" in mode:
            return _FailingWriteFile(f)
        return f

    with mock.patch("builtins.open", side_effect=_patched_open):
        rc = runner.run()

    assert rc == 1
    assert pidfile.exists()
    stage_pid = int(pidfile.read_text(encoding="utf-8").strip())
    alive = True
    try:
        os.kill(stage_pid, 0)
    except OSError:
        alive = False
    finally:
        if alive:
            try:
                os.kill(stage_pid, signal.SIGKILL)
            except OSError:
                pass
    assert not alive, f"stage pid={stage_pid} survived unexpected OSError in log writer"
    summary = (cfg.log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "03_ref_gloo_tiny_n2                  1 (OSError)" in summary
    assert "04_gloo_tiny_n2                      SKIPPED (ref failed)" in summary
    assert "09_rank_failure_gloo_n2              0 " in summary


# --------------------------------------------------------------------------- 10. helper capture & git interruptibility


def test_interruptible_capture_drains_large_stdout_and_enforces_deadline_and_interrupt_on_grandchild(
    tmp_path: Path,
) -> None:
    cfg = DRIVER.SessionConfig(backend="gloo", sizes=(), log_dir=tmp_path / "logs")
    runner = DRIVER.SessionRunner(config=cfg, repo_root=tmp_path, env=_clean_tracked_env())

    # Large stdout/stderr exceeding OS pipe buffer must drain cleanly without deadlocking.
    res_big = runner._run_interruptible_capture(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('A' * 200000); sys.stdout.flush(); "
            "sys.stderr.write('E' * 70000); sys.stderr.flush()",
        ],
        timeout=5.0,
    )
    assert res_big.returncode == 0 and not res_big.timed_out
    assert len(res_big.stdout) == 200000
    assert len(res_big.stderr) == 70000

    # Grandchild holding stdout open past parent exit must still respect timeout and be killed.
    pidfile_to = tmp_path / "gc_timeout.pid"
    gc_to_ready = tmp_path / "gc_timeout.ready"
    gc_to_code = f"import pathlib, time; pathlib.Path({str(gc_to_ready)!r}).write_text('ready'); time.sleep(20)"
    t0 = time.monotonic()
    try:
        res_to = runner._run_interruptible_capture(
            [
                sys.executable,
                "-c",
                "import subprocess, sys; "
                f"p = subprocess.Popen([sys.executable, '-c', {gc_to_code!r}]); "
                f"open({str(pidfile_to)!r}, 'w').write(str(p.pid))",
            ],
            timeout=0.45,
        )
        dt_to = time.monotonic() - t0
        assert res_to.timed_out and res_to.returncode == 124
        assert dt_to < 2.5, f"capture ignored deadline on grandchild holding stdout: {dt_to:.2f}s"
        assert gc_to_ready.exists(), "timeout grandchild never started"
        gc_pid_to = int(pidfile_to.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(gc_pid_to, 0)
    finally:
        if pidfile_to.exists():
            try:
                os.kill(int(pidfile_to.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass

    # Grandchild holding stdout open must also abort promptly on session interrupt and be killed.
    runner_int = DRIVER.SessionRunner(config=cfg, repo_root=tmp_path, env=_clean_tracked_env())
    pidfile_int = tmp_path / "gc_int.pid"
    gc_int_ready = tmp_path / "gc_int.ready"
    gc_int_code = f"import pathlib, time; pathlib.Path({str(gc_int_ready)!r}).write_text('ready'); time.sleep(20)"

    def _trigger_int() -> None:
        deadline = time.monotonic() + 10.0
        while not (pidfile_int.exists() and gc_int_ready.exists()) and time.monotonic() < deadline:
            time.sleep(0.01)
        runner_int.interrupted_sig_num = int(signal.SIGTERM)

    th = threading.Thread(target=_trigger_int, daemon=True)
    th.start()
    t1 = time.monotonic()
    try:
        res_int = runner_int._run_interruptible_capture(
            [
                sys.executable,
                "-c",
                "import subprocess, sys; "
                f"p = subprocess.Popen([sys.executable, '-c', {gc_int_code!r}]); "
                f"open({str(pidfile_int)!r}, 'w').write(str(p.pid))",
            ],
            timeout=10.0,
        )
        dt_int = time.monotonic() - t1
        th.join(timeout=5.0)
        assert res_int.interrupted and res_int.returncode == 143
        assert dt_int < 2.5, f"capture ignored interrupt on grandchild holding stdout: {dt_int:.2f}s"
        gc_pid_int = int(pidfile_int.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(gc_pid_int, 0)
    finally:
        th.join(timeout=2.0)
        if pidfile_int.exists():
            try:
                os.kill(int(pidfile_int.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass


def test_tree_state_git_interrupt_aborts_without_content_hash_fallback(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=True)
    fake_git_dir = tmp_path / "fake_git"
    fake_git_dir.mkdir()
    git_ready = tmp_path / "git.ready"
    git_pidfile = tmp_path / "git.pid"
    fake_git = fake_git_dir / "git"
    fake_git.write_text(
        f"#!{sys.executable}\n"
        f"import os, pathlib, time\n"
        f"pathlib.Path({str(git_pidfile)!r}).write_text(str(os.getpid()))\n"
        f"pathlib.Path({str(git_ready)!r}).write_text('ready')\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    _write_stub_uv(bin_dir, _standard_stub_uv_body())

    cfg = DRIVER.SessionConfig(backend="gloo", sizes=(), log_dir=tmp_path / "logs", ranks=(2,), ranks_specified=True)
    env = _clean_tracked_env({"PATH": f"{fake_git_dir}:{bin_dir}:{os.environ.get('PATH', '')}"})
    runner = DRIVER.SessionRunner(config=cfg, repo_root=repo, env=env)

    def _interrupt_git() -> None:
        deadline = time.monotonic() + 10.0
        while not git_ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        runner.interrupted_sig_num = int(signal.SIGTERM)

    th = threading.Thread(target=_interrupt_git, daemon=True)
    th.start()
    t0 = time.monotonic()
    try:
        with mock.patch.object(
            DRIVER,
            "_content_hash_tree_state",
            side_effect=AssertionError("must not fall back to content_hash when interrupted"),
        ):
            rc = runner.run()
        dt = time.monotonic() - t0
        th.join(timeout=5.0)

        assert rc == 143
        assert dt < 3.0, f"git interrupt waited too long: {dt:.2f}s"
        git_pid = int(git_pidfile.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(git_pid, 0)
    finally:
        th.join(timeout=2.0)
        if git_pidfile.exists():
            try:
                os.kill(int(git_pidfile.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
    summary = (cfg.log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "INTERRUPTED by SIGTERM (exit 143)" in summary


def test_empty_env_is_respected_by_git_and_ps_resolution(tmp_path: Path) -> None:
    assert DRIVER._try_git_tree_state(REPO_ROOT, tmp_path / "logs", env={}) is None
    assert DRIVER._which_in_env("ps", {}) is None
    with mock.patch.object(os.path, "isfile", return_value=False):
        assert DRIVER._resolve_ps_bin(env={}) is None


def test_helper_failure_and_timeout_report_stderr_without_conflating_with_zero_gpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Default helper timeout must be generous enough for cold CUDA initialisation.
    default_runner = DRIVER.SessionRunner(config=DRIVER.SessionConfig(backend="gloo", sizes=()), repo_root=tmp_path)
    assert default_runner.helper_timeout_s >= 180.0

    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})

    # 1. GPU count probe fails with stderr -> explicit failure error, NOT "0 visible CUDA GPUs".
    stub_gpu_fail = """\
import sys
argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync": sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined: print("stub-rt"); sys.exit(0)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
if "torch.cuda.device_count()" in joined:
    print("CUDA_DRIVER_INIT_BOOM", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_gpu_fail)
    cfg1 = DRIVER.SessionConfig(backend="nccl", sizes=(), log_dir=tmp_path / "logs_gpu_fail")
    rc1 = DRIVER.SessionRunner(config=cfg1, repo_root=repo, env=env).run()
    err1 = capsys.readouterr().err
    assert rc1 == 2
    assert "CUDA GPU count probe failed (exit 1)" in err1
    assert "CUDA_DRIVER_INIT_BOOM" in err1
    assert "0 visible CUDA GPUs" not in err1
    assert (cfg1.log_dir / "summary.txt").is_file()

    # 2. GPU count probe times out with stderr -> explicit timeout error, NOT "0 visible CUDA GPUs".
    stub_gpu_timeout = """\
import sys, time
argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync": sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined: print("stub-rt"); sys.exit(0)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
if "torch.cuda.device_count()" in joined:
    print("SLOW_CUDA_INIT_MSG", file=sys.stderr, flush=True)
    time.sleep(20)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_gpu_timeout)
    cfg2 = DRIVER.SessionConfig(backend="nccl", sizes=(), log_dir=tmp_path / "logs_gpu_to")
    rc2 = DRIVER.SessionRunner(config=cfg2, repo_root=repo, env=env, helper_timeout_s=0.3).run()
    err2 = capsys.readouterr().err
    assert rc2 == 2
    assert "CUDA GPU count probe timed out" in err2
    assert "SLOW_CUDA_INIT_MSG" in err2
    assert "0 visible CUDA GPUs" not in err2
    assert (cfg2.log_dir / "summary.txt").is_file()

    # 3. Post-01_env runtime identity probe failure must error with stderr rather than reset to "uninitialized".
    rt_calls = tmp_path / "rt_calls_helper_err.txt"
    stub_rt_fail = f"""\
import pathlib, sys
argv = sys.argv[1:]
joined = " ".join(argv)
counter = pathlib.Path({str(rt_calls)!r})
if argv and argv[0] == "sync": sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined:
    n = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(n))
    if n == 1:
        # Pre-sync failure is allowed (fresh venv not synced yet).
        sys.exit(1)
    print("POST_SYNC_TORCH_IMPORT_ERROR", file=sys.stderr)
    sys.exit(1)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_rt_fail)
    cfg3 = DRIVER.SessionConfig(
        backend="gloo", sizes=(), log_dir=tmp_path / "logs_rt_fail", ranks=(2,), ranks_specified=True
    )
    rc3 = DRIVER.SessionRunner(config=cfg3, repo_root=repo, env=env).run()
    err3 = capsys.readouterr().err
    assert rc3 == 2
    assert "runtime environment probe failed (exit 1)" in err3
    assert "POST_SYNC_TORCH_IMPORT_ERROR" in err3
    assert (cfg3.log_dir / "summary.txt").is_file()


# --------------------------------------------------------------------------- 11. artifact cleanup & resume guard


def test_failed_timeout_and_corrupt_ref_stages_remove_partial_archive_and_tmp(tmp_path: Path) -> None:
    cfg = DRIVER.SessionConfig(backend="gloo", sizes=(), log_dir=tmp_path / "logs", stage_timeout=0.8)
    runner = DRIVER.SessionRunner(
        config=cfg, repo_root=tmp_path, env=_clean_tracked_env(), teardown_grace_s=0.25, teardown_reap_s=0.5
    )
    runner.log_dir.mkdir(parents=True, exist_ok=True)
    runner.refs_dir.mkdir(parents=True, exist_ok=True)
    runner.tree_method, runner.tree_sig = "content_hash", "0" * 64

    ref_zip = runner.refs_dir / "ref_gloo_tiny_n2.zip"
    ref_tmp = Path(f"{ref_zip}.tmp")
    ready_to = tmp_path / "ref_timeout.ready"

    # 1. Stage timeout (124) after writing partial .zip and .tmp must remove both.
    out_to = runner.run_stage(
        "ref_timeout",
        ref_zip,
        [
            sys.executable,
            "-c",
            f"import pathlib, time; pathlib.Path({str(ref_zip)!r}).write_bytes(b'corrupt'); "
            f"pathlib.Path({str(ref_tmp)!r}).write_bytes(b'partial'); "
            f"pathlib.Path({str(ready_to)!r}).write_text('ready'); time.sleep(10)",
        ],
    )
    assert ready_to.exists(), "timeout stage exited before writing partial files"
    assert not out_to.ok and out_to.returncode == 124
    assert not ref_zip.exists() and not ref_tmp.exists()

    # 2. Non-zero exit (1) after writing partial .zip and .tmp must remove both.
    out_fail = runner.run_stage(
        "ref_fail",
        ref_zip,
        [
            sys.executable,
            "-c",
            f"import pathlib, sys; pathlib.Path({str(ref_zip)!r}).write_bytes(b'corrupt'); "
            f"pathlib.Path({str(ref_tmp)!r}).write_bytes(b'partial'); sys.exit(1)",
        ],
    )
    assert not out_fail.ok and out_fail.returncode == 1
    assert not ref_zip.exists() and not ref_tmp.exists()

    # 3. Exit 0 with corrupt archive + leftover .tmp must remove both .zip and .tmp.
    out_corrupt = runner.run_stage(
        "ref_corrupt",
        ref_zip,
        [
            sys.executable,
            "-c",
            f"import pathlib; pathlib.Path({str(ref_zip)!r}).write_bytes(b'badzip'); "
            f"pathlib.Path({str(ref_tmp)!r}).write_bytes(b'leftover')",
        ],
    )
    assert not out_corrupt.ok and out_corrupt.returncode == 1
    assert not ref_zip.exists() and not ref_tmp.exists()


def test_interrupt_during_resume_check_preserves_disk_and_skips_stage_recording(tmp_path: Path) -> None:
    cfg = DRIVER.SessionConfig(backend="gloo", sizes=(), log_dir=tmp_path / "logs", ranks=(2,), ranks_specified=True)
    runner = DRIVER.SessionRunner(config=cfg, repo_root=tmp_path, env=_clean_tracked_env())
    runner.log_dir.mkdir(parents=True, exist_ok=True)
    runner.refs_dir.mkdir(parents=True, exist_ok=True)
    runner.tree_method, runner.tree_sig = "content_hash", "0" * 64

    ref_zip = runner.refs_dir / "ref.zip"
    ref_zip.write_bytes(b"preexisting-archive")
    done_file = runner.log_dir / "st1.done"
    done_file.write_text("sig=stale_sig\nartifact_sha256=stale\n", encoding="utf-8")
    log_file = runner.log_dir / "st1.log"
    log_file.write_text("previous good log\n", encoding="utf-8")

    # Interrupt arriving while is_stage_resumed() runs (returning False) must not touch disk or record st1.
    orig_is_resumed = runner.is_stage_resumed

    def _interrupt_on_check(name: str, sig: str, req: Path | None) -> tuple[bool, str]:
        res = orig_is_resumed(name, sig, req)
        runner.interrupted_sig_num = int(signal.SIGTERM)
        return res

    with mock.patch.object(runner, "is_stage_resumed", side_effect=_interrupt_on_check):
        out = runner.run_stage("st1", ref_zip, [sys.executable, "-c", "print('must not run')"])
    assert not out.ok and out.returncode == 143 and out.interrupted_sig == int(signal.SIGTERM)
    assert done_file.is_file()
    assert ref_zip.read_bytes() == b"preexisting-archive"
    assert log_file.read_text(encoding="utf-8") == "previous good log\n"
    assert runner.stages == []

    # Same guard inside run_torchrun_stage() when interrupt arrives during resume check.
    runner.interrupted_sig_num = None
    with mock.patch.object(runner, "is_stage_resumed", side_effect=_interrupt_on_check):
        out_tr = runner.run_torchrun_stage("04_gloo_tiny_n2", 2, "a" * 64, ["driver.py"])
    assert not out_tr.ok and out_tr.returncode == 143 and out_tr.interrupted_sig == int(signal.SIGTERM)
    assert runner.stages == []


def test_relative_log_dir_keeps_stage_args_signatures_and_errors_relative(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret_parent = tmp_path / "secret_user_checkout_dir"
    repo = _make_stub_repo(secret_parent, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    call_log = tmp_path / "calls.log"
    _write_stub_uv(bin_dir, _standard_stub_uv_body(call_log=call_log))
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})

    # Run with relative --log-dir; stage arguments and .done signatures must stay relative.
    cfg = DRIVER.SessionConfig(backend="gloo", sizes=(), log_dir=Path("rel_logs"), ranks=(2,), ranks_specified=True)
    runner = DRIVER.SessionRunner(config=cfg, repo_root=repo, env=env)
    rc = runner.run()
    assert rc == 0
    calls = call_log.read_text(encoding="utf-8")
    assert "--write-reference rel_logs/refs/ref_gloo_tiny_n2.zip" in calls
    assert "--reference rel_logs/refs/ref_gloo_tiny_n2.zip" in calls
    assert "secret_user_checkout_dir" not in calls

    # Relocating the repo directory preserves .done signatures when --log-dir is relative.
    relocated_repo = tmp_path / "other_checkout_dir" / "repo"
    shutil.copytree(repo, relocated_repo)
    call_log.unlink()
    runner2 = DRIVER.SessionRunner(config=cfg, repo_root=relocated_repo, env=env)
    assert runner2.run() == 0
    s2 = (relocated_repo / "rel_logs" / "summary.txt").read_text(encoding="utf-8")
    assert s2.count("0 (resumed)") == 4

    # Corrupt-artifact stderr diagnostic must also use the relative path (no machine path leak).
    _write_stub_uv(bin_dir, _standard_stub_uv_body(corrupt_ref_n={2}))
    cfg_corrupt = DRIVER.SessionConfig(
        backend="gloo", sizes=(), log_dir=Path("rel_logs"), ranks=(2,), ranks_specified=True, force=True
    )
    capsys.readouterr()
    rc_corrupt = DRIVER.SessionRunner(config=cfg_corrupt, repo_root=repo, env=env).run()
    err_corrupt = capsys.readouterr().err
    assert rc_corrupt == 1
    assert "rel_logs/refs/ref_gloo_tiny_n2.zip" in err_corrupt
    assert "secret_user_checkout_dir" not in err_corrupt


# --------------------------------------------------------------------------- 12. CLI pipes, wrapper, early signals


def test_cli_help_and_error_survive_closed_stdout_and_stderr_pipes() -> None:
    p_help = subprocess.Popen(
        [sys.executable, str(SCRIPT_PY), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert p_help.stdout is not None
    p_help.stdout.close()
    _, err_help = p_help.communicate(timeout=15)
    assert p_help.returncode == 0
    assert err_help == b"", err_help

    p_err = subprocess.Popen(
        [sys.executable, str(SCRIPT_PY), "--unknown-flag"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert p_err.stderr is not None
    p_err.stderr.close()
    out_err, _ = p_err.communicate(timeout=15)
    assert p_err.returncode == 2
    assert out_err == b"", out_err

    # Unicode superscript digit in --ranks must exit 2 cleanly (never raise ValueError).
    r_uni = subprocess.run(
        [sys.executable, str(SCRIPT_PY), "--ranks", "\u00b2"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r_uni.returncode == 2
    assert "Traceback" not in r_uni.stderr


def test_wrapper_interpreter_fallback_order_and_opaque_arg_passthrough(tmp_path: Path) -> None:
    bash_bin = "/bin/bash" if os.path.isfile("/bin/bash") else shutil.which("bash")
    assert bash_bin is not None

    repo = _make_stub_repo(tmp_path, with_git=False)

    def _base_bin(name: str) -> Path:
        bd = tmp_path / name
        bd.mkdir()
        for tool in ("dirname", "pwd", "ps"):
            tp = shutil.which(tool)
            if tp is not None:
                os.symlink(tp, bd / tool)
        return bd

    # 1. `python3 >= 3.10` on PATH is preferred directly without calling `uv python find`,
    #    passes `--python` and awkward forms like `--only --python` through untouched, and execs directly.
    bin_py3 = _base_bin("bin_py3")
    os.symlink(sys.executable, bin_py3 / "python3")
    uv_log_1 = tmp_path / "uv_calls_1.log"
    pid_probe = tmp_path / "orch_pid.txt"
    uv_stub_body = f"""\
import json, os, pathlib, sys, uuid, zipfile

argv = sys.argv[1:]
joined = " ".join(argv)
with open({str(uv_log_1)!r}, "a", encoding="utf-8") as uf:
    uf.write(joined + "\\n")
if argv[:2] == ["python", "find"]:
    print({sys.executable!r})
    sys.exit(0)
if argv and argv[0] == "sync":
    sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined:
    print("stub-rt-wrapper")
    sys.exit(0)
if "tree_state_method=" in joined:
    print("tree_state_method=content_hash tree_state_sig=000000000000")
    sys.exit(0)
if "--write-reference" in argv:
    pathlib.Path({str(pid_probe)!r}).write_text(str(os.getppid()), encoding="utf-8")
    ref_path = pathlib.Path(argv[argv.index("--write-reference") + 1])
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    spec_doc = {{"spec": {{"schema_version": 5}}, "environment": {{"archive_id": uuid.uuid4().hex}}}}
    entries_doc = {{
        "tiny_random:s": {{"tensor_member": "tensors/0000.bin", "tensor_dtype": "float32", "tensor_shape": [4]}}
    }}
    with zipfile.ZipFile(ref_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", b"\\x00" * 16)
        zf.writestr("spec.json", json.dumps(spec_doc).encode())
        zf.writestr("entries.json", json.dumps(entries_doc).encode())
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_py3, uv_stub_body)
    proc1 = subprocess.Popen(
        [
            bash_bin,
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            "--ranks",
            "2",
            "--python",
            "3.12",
            "--only",
            "--python",
            "--log-dir",
            str(tmp_path / "logs_py3"),
        ],
        cwd=repo,
        env=_clean_tracked_env({"PATH": str(bin_py3)}),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, err1 = proc1.communicate(timeout=30)
    finally:
        _kill_process_group_quiet(proc1)
    assert proc1.returncode == 0, err1
    assert int(pid_probe.read_text(encoding="utf-8").strip()) == proc1.pid
    calls1 = uv_log_1.read_text(encoding="utf-8").splitlines()
    assert not any(c.startswith("python find") for c in calls1)
    assert any(c.startswith("sync ") and "--python 3.12" in c for c in calls1)
    assert any("--write-reference" in c and "--only --python" in c and "--python 3.12" in c for c in calls1)

    # 2. When `python3` fails the >= 3.10 check, wrapper resolves via `uv python find --no-project ">=3.10"`
    #    (ignoring any `--python` or awkward CLI tokens) and `exec`s the resolved interpreter directly.
    bin_find = _base_bin("bin_find")
    fake_py3 = bin_find / "python3"
    fake_py3.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_py3.chmod(0o755)
    uv_log_1.unlink()
    pid_probe.unlink()
    _write_stub_uv(bin_find, uv_stub_body)
    proc2 = subprocess.Popen(
        [
            bash_bin,
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            "--ranks",
            "2",
            "--python",
            "3.9",
            "--python",
            "3.12",
            "--only",
            "--python",
            "--log-dir",
            str(tmp_path / "logs_find"),
        ],
        cwd=repo,
        env=_clean_tracked_env({"PATH": str(bin_find)}),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, err2 = proc2.communicate(timeout=30)
    finally:
        _kill_process_group_quiet(proc2)
    assert proc2.returncode == 0, err2
    assert int(pid_probe.read_text(encoding="utf-8").strip()) == proc2.pid
    calls2 = uv_log_1.read_text(encoding="utf-8").splitlines()
    find_calls = [c for c in calls2 if c.startswith("python find")]
    assert find_calls == ["python find --no-project --resolve-links >=3.10"]
    assert any(c.startswith("sync ") and "--python 3.12" in c for c in calls2)
    assert any("--write-reference" in c and "--only --python" in c for c in calls2)

    # 3. When `uv python find` fails, wrapper falls back to `exec uv run --no-project --python ">=3.10" python ...`
    #    passing all user flags through untouched.
    bin_run = _base_bin("bin_run")
    fake_py3_run = bin_run / "python3"
    fake_py3_run.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_py3_run.chmod(0o755)
    uv_run_log = tmp_path / "uv_run_fallback.log"
    uv_run_stub = f"""\
import os, sys
argv = sys.argv[1:]
with open({str(uv_run_log)!r}, "a", encoding="utf-8") as f:
    f.write(" ".join(argv) + "\\n")
if argv[:2] == ["python", "find"]:
    sys.exit(1)
if argv[:5] == ["run", "--no-project", "--python", ">=3.10", "python"]:
    os.execv({sys.executable!r}, [{sys.executable!r}, *argv[5:]])
sys.exit(99)
"""
    _write_stub_uv(bin_run, uv_run_stub)
    r_run = subprocess.run(
        [
            bash_bin,
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--python",
            "3.13",
            "--help",
        ],
        cwd=repo,
        env=_clean_tracked_env({"PATH": str(bin_run)}),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r_run.returncode == 0, r_run.stderr
    assert "Usage: bash scripts/multi_gpu_check.sh [options]" in r_run.stdout
    run_calls = uv_run_log.read_text(encoding="utf-8").splitlines()
    assert run_calls[:2] == [
        "python find --no-project --resolve-links >=3.10",
        "python find --no-project >=3.10",
    ]
    assert run_calls[2].startswith("run --no-project --python >=3.10 python ")
    assert run_calls[2].endswith("multi_gpu_check.py --python 3.13 --help")

    # 4. When neither Python >= 3.10 nor `uv` is available on PATH, wrapper exits 2 with a clear error.
    bin_none = _base_bin("bin_none")
    r_none = subprocess.run(
        [bash_bin, str(repo / "scripts" / "multi_gpu_check.sh"), "--help"],
        cwd=repo,
        env={"PATH": str(bin_none), "TMPDIR": "/tmp"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r_none.returncode == 2
    assert "Python >= 3.10 or 'uv' is required" in r_none.stderr


def test_main_installs_signal_handlers_immediately_after_parse_args(tmp_path: Path) -> None:
    log_dir = tmp_path / "early_sig_logs"
    observed_handlers: dict[str, Any] = {}
    prev_term = signal.getsignal(signal.SIGTERM)
    prev_hup = signal.getsignal(signal.SIGHUP)
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

        def _chdir_spy(_path: Any) -> None:
            observed_handlers["term"] = signal.getsignal(signal.SIGTERM)
            observed_handlers["hup"] = signal.getsignal(signal.SIGHUP)
            os.kill(os.getpid(), signal.SIGTERM)

        with (
            mock.patch.object(DRIVER, "REPO_ROOT", tmp_path),
            mock.patch.object(os, "chdir", side_effect=_chdir_spy),
        ):
            rc = DRIVER.main(["--backend", "gloo", "--ranks", "2", "--log-dir", str(log_dir)])
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGHUP, prev_hup)

    assert rc == 143
    assert observed_handlers["term"] not in (signal.SIG_DFL, signal.SIG_IGN)
    assert observed_handlers["hup"] == signal.SIG_IGN
    summary = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "INTERRUPTED by SIGTERM (exit 143)" in summary
    assert "01_env" not in summary


# --------------------------------------------------------------------------- 13. harness & full behaviour coverage


def test_leftover_detector_catches_env_only_process_and_cleanup_reaps_it(tmp_path: Path) -> None:
    tag = f"tt_env_only_{os.getpid()}"
    ready = tmp_path / "env_only.ready"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import pathlib, time; pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(30)",
        ],
        env=dict(os.environ, _TT_TEST_LEFTOVER_TAG=tag),
        start_new_session=True,
    )
    try:
        _wait_for_path(ready, proc, timeout=10.0)
        hits = _list_matching_pids(tag)
        assert any(entry.split()[0] == str(proc.pid) for entry in hits), hits
        assert _list_matching_pids(f"{tag}_other") == []

        _cleanup_tagged_processes(tag)
        proc.wait(timeout=5.0)
        assert _list_matching_pids(tag) == []
    finally:
        _kill_process_group_quiet(proc)
        _cleanup_tagged_processes(tag)


def test_timeout_teardown_signal_deterministic_via_hook(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    later_file = tmp_path / "later_hook.started"
    stub_body = f"""\
import pathlib, signal, sys, time
argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync": sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined: print("stub-rt"); sys.exit(0)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
if "--write-reference" in argv:
    ws = int(argv[argv.index("--world-size") + 1])
    if ws == 2:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(30)
    pathlib.Path({str(later_file)!r}).write_text("started")
    sys.exit(0)
if "distributed_rank_failure_check.py" in joined:
    pathlib.Path({str(later_file)!r}).write_text("started")
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})

    # 1. Default disposition: SIGTERM delivered inside on_timeout_teardown hook interrupts cleanly.
    prev_term = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        cfg1 = DRIVER.SessionConfig(
            backend="gloo", sizes=(), ranks=(2, 4), ranks_specified=True, log_dir=tmp_path / "l1", stage_timeout=0.15
        )
        runner1 = DRIVER.SessionRunner(
            config=cfg1,
            repo_root=repo,
            env=env,
            teardown_grace_s=0.3,
            teardown_reap_s=0.5,
            on_timeout_teardown=lambda _st: os.kill(os.getpid(), signal.SIGTERM),
        )
        rc1 = runner1.run()
    finally:
        signal.signal(signal.SIGTERM, prev_term)
    assert rc1 == 143
    assert not later_file.exists()
    s1 = (cfg1.log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "INTERRUPTED by SIGTERM (exit 143)" in s1
    assert "143 (interrupted:SIGTERM)" in s1
    assert "during timeout teardown" in (cfg1.log_dir / "03_ref_gloo_tiny_n2.log").read_text(encoding="utf-8")

    # 2. Ignored disposition: SIGHUP=SIG_IGN delivered inside on_timeout_teardown must NOT interrupt.
    prev_hup = signal.getsignal(signal.SIGHUP)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        cfg2 = DRIVER.SessionConfig(
            backend="gloo", sizes=(), ranks=(2, 4), ranks_specified=True, log_dir=tmp_path / "l2", stage_timeout=0.15
        )
        runner2 = DRIVER.SessionRunner(
            config=cfg2,
            repo_root=repo,
            env=env,
            teardown_grace_s=0.3,
            teardown_reap_s=0.5,
            on_timeout_teardown=lambda _st: os.kill(os.getpid(), signal.SIGHUP),
        )
        rc2 = runner2.run()
    finally:
        signal.signal(signal.SIGHUP, prev_hup)
    assert rc2 == 1
    assert later_file.exists()
    s2 = (cfg2.log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "INTERRUPTED" not in s2
    assert "03_ref_gloo_tiny_n2                  124 " in s2


def test_stage_runs_in_isolated_session_shielded_from_orchestrator_group_signals(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    tag = f"tt_sess_{os.getpid()}"
    ready_file = tmp_path / "sess.ready"
    go_file = tmp_path / "sess.go"
    ids_file = tmp_path / "sess.ids"

    stub_body = f"""\
import json, os, pathlib, signal, sys, time, uuid, zipfile
argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync": sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined: print("stub-rt"); sys.exit(0)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
if "--write-reference" in argv:
    # Explicitly reset SIGHUP to SIG_DFL inside the child: only session/pgid isolation protects it
    # when SIGHUP is sent to the orchestrator's process group under an ignored-SIGHUP session.
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    pid = os.getpid()
    pathlib.Path({str(ids_file)!r}).write_text(f"{{pid}} {{os.getpgid(pid)}} {{os.getsid(pid)}}")
    pathlib.Path({str(ready_file)!r}).write_text("ready")
    deadline = time.monotonic() + 20.0
    while not pathlib.Path({str(go_file)!r}).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    ref_path = pathlib.Path(argv[argv.index("--write-reference") + 1])
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    spec_doc = {{"spec": {{"schema_version": 5}}, "environment": {{"archive_id": uuid.uuid4().hex}}}}
    entries_doc = {{
        "tiny_random:s": {{"tensor_member": "tensors/0000.bin", "tensor_dtype": "float32", "tensor_shape": [4]}}
    }}
    with zipfile.ZipFile(ref_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", b"\\x00" * 16)
        zf.writestr("spec.json", json.dumps(spec_doc).encode())
        zf.writestr("entries.json", json.dumps(entries_doc).encode())
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "_TT_TEST_LEFTOVER_TAG": tag})
    log_dir = tmp_path / "logs"

    def _preexec() -> None:
        os.setsid()
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    proc = subprocess.Popen(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            "--ranks",
            "2",
            "--log-dir",
            str(log_dir),
        ],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=_preexec,
    )
    try:
        _wait_for_path(ready_file, proc, timeout=20.0)
        child_pid, child_pgid, child_sid = (int(x) for x in ids_file.read_text(encoding="utf-8").split())
        assert child_pgid == child_pid and child_sid == child_pid
        assert child_pgid != proc.pid and child_sid != proc.pid
        # Send SIGHUP to the orchestrator's process group; child has SIGHUP=SIG_DFL so it survives
        # ONLY because start_new_session=True isolated it into its own session/process group.
        os.killpg(proc.pid, signal.SIGHUP)
        time.sleep(0.1)
        go_file.write_text("go", encoding="utf-8")
        out, _ = proc.communicate(timeout=30.0)
        assert proc.returncode == 0, out
        assert _list_matching_pids(tag) == []
    finally:
        _kill_process_group_quiet(proc)
        _cleanup_tagged_processes(tag)


def test_cli_options_and_session_env_propagated_to_stages(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    call_log = tmp_path / "calls.jsonl"

    stub_body = f"""\
import json, os, pathlib, sys, uuid, zipfile
argv = sys.argv[1:]
joined = " ".join(argv)
rec = {{
    "argv": argv,
    "TMPDIR": os.environ.get("TMPDIR"),
    "PYTHONFAULTHANDLER": os.environ.get("PYTHONFAULTHANDLER"),
    "NCCL_MULTI_RANK_GPU_ENABLE": os.environ.get("NCCL_MULTI_RANK_GPU_ENABLE"),
}}
with open({str(call_log)!r}, "a", encoding="utf-8") as cf:
    cf.write(json.dumps(rec) + "\\n")
if argv and argv[0] == "sync": sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined: print("stub-rt"); sys.exit(0)
if "torch.cuda.device_count()" in joined and "tree_state_method=" not in joined: print("1"); sys.exit(0)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
if "--write-reference" in argv:
    ref_path = pathlib.Path(argv[argv.index("--write-reference") + 1])
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    spec_doc = {{"spec": {{"schema_version": 5}}, "environment": {{"archive_id": uuid.uuid4().hex}}}}
    entries_doc = {{
        "tiny_random:s": {{"tensor_member": "tensors/0000.bin", "tensor_dtype": "float32", "tensor_shape": [4]}}
    }}
    with zipfile.ZipFile(ref_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", b"\\x00" * 16)
        zf.writestr("spec.json", json.dumps(spec_doc).encode())
        zf.writestr("entries.json", json.dumps(entries_doc).encode())
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)

    cfg = DRIVER.parse_args(
        [
            "--backend",
            "nccl",
            "--shared-gpu",
            "--ranks",
            "2",
            "--sizes",
            "5M",
            "--python",
            "3.12",
            "--reference-mode",
            "sketch",
            "--pg-timeout",
            "7",
            "--step-timeout",
            "42",
            "--only",
            "domain_head_only_f32",
            "--skip",
            "workers_uneven_f32",
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )
    long_tmp = "/" + "t" * 70
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "TMPDIR": long_tmp})
    env.pop("PYTHONFAULTHANDLER", None)
    env.pop("NCCL_MULTI_RANK_GPU_ENABLE", None)

    runner = DRIVER.SessionRunner(config=cfg, repo_root=repo, env=env)
    rc = runner.run()
    assert rc == 0

    records = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
    assert records, "no stage invocations recorded"
    # Verify _prepare_session_env effects reached every spawned stage process.
    for rec in records:
        assert rec["TMPDIR"] == "/tmp"
        assert rec["PYTHONFAULTHANDLER"] == "1"
        assert rec["NCCL_MULTI_RANK_GPU_ENABLE"] == "1"

    argvs = [r["argv"] for r in records]
    sync_cmds = [a for a in argvs if a and a[0] == "sync"]
    ref_cmds = [a for a in argvs if "--write-reference" in a]
    tr_cmds = [a for a in argvs if "torchrun" in a]
    rf_cmds = [a for a in argvs if any("distributed_rank_failure_check.py" in tok for tok in a)]
    assert sync_cmds and ref_cmds and tr_cmds and rf_cmds

    for cmd in (*sync_cmds, *ref_cmds, *tr_cmds, *rf_cmds):
        idx = cmd.index("--python")
        assert cmd[idx + 1] == "3.12"
    for cmd in ref_cmds:
        assert cmd[cmd.index("--reference-mode") + 1] == "sketch"
        assert cmd[cmd.index("--only") + 1] == "domain_head_only_f32"
        assert cmd[cmd.index("--skip") + 1] == "workers_uneven_f32"
    for cmd in tr_cmds:
        assert cmd[cmd.index("--timeout") + 1] == "42"
        assert cmd[cmd.index("--only") + 1] == "domain_head_only_f32"
        assert cmd[cmd.index("--skip") + 1] == "workers_uneven_f32"
    for cmd in rf_cmds:
        assert cmd[cmd.index("--pg-timeout") + 1] == "7"


def test_env_stage_rebinds_done_marker_when_runtime_identity_changes_after_sync(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    synced_flag = tmp_path / "venv.synced"
    sync_count = tmp_path / "sync.count"

    stub_body = f"""\
import json, pathlib, sys, uuid, zipfile
argv = sys.argv[1:]
joined = " ".join(argv)
synced = pathlib.Path({str(synced_flag)!r})
count_f = pathlib.Path({str(sync_count)!r})
if argv and argv[0] == "sync":
    n = int(count_f.read_text()) + 1 if count_f.exists() else 1
    count_f.write_text(str(n))
    synced.write_text("yes")
    sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined:
    if not synced.exists():
        # Fresh environment before first `uv sync`: pre-sync probe fails (uninitialized).
        sys.exit(1)
    print("post-sync-runtime-sig-v2")
    sys.exit(0)
if "tree_state_method=" in joined:
    print("tree_state_method=content_hash tree_state_sig=000000000000")
    sys.exit(0)
if "--write-reference" in argv:
    ref_path = pathlib.Path(argv[argv.index("--write-reference") + 1])
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    spec_doc = {{"spec": {{"schema_version": 5}}, "environment": {{"archive_id": uuid.uuid4().hex}}}}
    entries_doc = {{
        "tiny_random:s": {{"tensor_member": "tensors/0000.bin", "tensor_dtype": "float32", "tensor_shape": [4]}}
    }}
    with zipfile.ZipFile(ref_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", b"\\x00" * 16)
        zf.writestr("spec.json", json.dumps(spec_doc).encode())
        zf.writestr("entries.json", json.dumps(entries_doc).encode())
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})
    cfg = DRIVER.SessionConfig(backend="gloo", sizes=(), ranks=(2,), ranks_specified=True, log_dir=tmp_path / "logs")

    assert DRIVER.SessionRunner(config=cfg, repo_root=repo, env=env).run() == 0
    assert int(sync_count.read_text(encoding="utf-8")) == 1

    # Immediate re-run must resume 01_env (because 01_env.done was rebound to post-sync runtime identity).
    assert DRIVER.SessionRunner(config=cfg, repo_root=repo, env=env).run() == 0
    assert int(sync_count.read_text(encoding="utf-8")) == 1
    summary = (cfg.log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "01_env                               0 (resumed)" in summary
    assert summary.count("0 (resumed)") == 4


@pytest.mark.parametrize(("kill_sig", "expected_rc"), [(signal.SIGKILL, 137), (signal.SIGABRT, 134)])
def test_signal_killed_stage_records_posix_code_and_continues_session(
    tmp_path: Path, kill_sig: signal.Signals, expected_rc: int
) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    stub_body = f"""\
import os, signal, sys
argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync": sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined: print("stub-rt"); sys.exit(0)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
if "--write-reference" in argv:
    os.kill(os.getpid(), signal.{kill_sig.name})
if "distributed_rank_failure_check.py" in joined:
    print("result: PASS (no hang)")
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})
    cfg = DRIVER.SessionConfig(backend="gloo", sizes=(), ranks=(2,), ranks_specified=True, log_dir=tmp_path / "logs")

    rc = DRIVER.SessionRunner(config=cfg, repo_root=repo, env=env).run()
    assert rc == 1
    summary = (cfg.log_dir / "summary.txt").read_text(encoding="utf-8")
    assert f"03_ref_gloo_tiny_n2                  {expected_rc} " in summary
    assert "04_gloo_tiny_n2                      SKIPPED (ref failed)" in summary
    assert "09_rank_failure_gloo_n2              0 " in summary


def test_ref_failure_skips_dependent_released_and_gloo_comparison_stages(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    stub_body = """\
import json, pathlib, sys, uuid, zipfile
argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync": sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined: print("stub-rt"); sys.exit(0)
if "torch.cuda.device_count()" in joined and "tree_state_method=" not in joined: print("2"); sys.exit(0)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
if "--write-reference" in argv:
    ref_path = pathlib.Path(argv[argv.index("--write-reference") + 1])
    if "released" in ref_path.name or "ref_gloo_" in ref_path.name:
        sys.exit(1)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    spec_doc = {"spec": {"schema_version": 5}, "environment": {"archive_id": uuid.uuid4().hex}}
    entries_doc = {
        "tiny_random:s": {"tensor_member": "tensors/0000.bin", "tensor_dtype": "float32", "tensor_shape": [4]}
    }
    with zipfile.ZipFile(ref_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", b"\\x00" * 16)
        zf.writestr("spec.json", json.dumps(spec_doc).encode())
        zf.writestr("entries.json", json.dumps(entries_doc).encode())
    sys.exit(0)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})
    cfg = DRIVER.SessionConfig(
        backend="nccl", sizes=("5M",), ranks=(2,), ranks_specified=True, log_dir=tmp_path / "logs"
    )

    rc = DRIVER.SessionRunner(config=cfg, repo_root=repo, env=env).run()
    assert rc == 1
    summary = (cfg.log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "03_ref_nccl_tiny_n2                  0 " in summary
    assert "04_nccl_tiny_n2                      0 " in summary
    assert "05_ref_nccl_released_n2              1 " in summary
    assert "06_nccl_released_n2                  SKIPPED (ref failed)" in summary
    assert "07_ref_gloo_tiny_n2                  1 " in summary
    assert "08_gloo_tiny_n2                      SKIPPED (ref failed)" in summary
    assert "09_rank_failure_gloo_n2              0 " in summary
    assert "10_rank_failure_nccl_n2              0 " in summary


def test_pick_free_port_failure_records_no_port_and_continues(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    _write_stub_uv(bin_dir, _standard_stub_uv_body())
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})
    cfg = DRIVER.SessionConfig(backend="gloo", sizes=(), ranks=(2, 4), ranks_specified=True, log_dir=tmp_path / "logs")

    real_pick = DRIVER.pick_free_port
    calls = {"n": 0}

    def _flaky_port() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("ephemeral port exhausted")
        return real_pick()

    with mock.patch.object(DRIVER, "pick_free_port", side_effect=_flaky_port):
        rc = DRIVER.SessionRunner(config=cfg, repo_root=repo, env=env).run()
    assert rc == 1
    summary = (cfg.log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "04_gloo_tiny_n2                      1 (no port)" in summary
    assert "04_gloo_tiny_n4                      0 " in summary
    assert "09_rank_failure_gloo_n2              0 " in summary
    assert "09_rank_failure_gloo_n4              0 " in summary


def test_second_signal_during_teardown_preserves_first_signal_and_completes_cleanup(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    tag = f"tt_second_sig_{os.getpid()}"
    stage_ready = tmp_path / "stage.ready"
    teardown_ready = tmp_path / "teardown.ready"
    child_pidfile = tmp_path / "child.pid"

    stub_body = f"""\
import os, pathlib, signal, sys, time
argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync": sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined: print("stub-rt"); sys.exit(0)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
if "--write-reference" in argv:
    pathlib.Path({str(child_pidfile)!r}).write_text(str(os.getpid()))
    def _on_term(_s, _f):
        pathlib.Path({str(teardown_ready)!r}).write_text("in_teardown")
    signal.signal(signal.SIGTERM, _on_term)
    pathlib.Path({str(stage_ready)!r}).write_text("ready")
    time.sleep(30)
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "_TT_TEST_LEFTOVER_TAG": tag})
    log_dir = tmp_path / "logs"

    def _preexec() -> None:
        os.setsid()
        for managed in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(managed, signal.SIG_DFL)

    proc = subprocess.Popen(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            "--ranks",
            "2",
            "--log-dir",
            str(log_dir),
        ],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=_preexec,
    )
    try:
        _wait_for_path(stage_ready, proc, timeout=20.0)
        # First signal: SIGINT.
        os.killpg(proc.pid, signal.SIGINT)
        # Wait until the stage process receives SIGTERM from the orchestrator's teardown.
        _wait_for_path(teardown_ready, proc, timeout=15.0)
        # Send both a second SIGINT and a different signal (SIGTERM) during active teardown.
        os.killpg(proc.pid, signal.SIGINT)
        os.killpg(proc.pid, signal.SIGTERM)
        out, _ = proc.communicate(timeout=30.0)
        assert proc.returncode == 130, out
        child_pid = int(child_pidfile.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert _list_matching_pids(tag) == []
    finally:
        _kill_process_group_quiet(proc)
        _cleanup_tagged_processes(tag)

    summary = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "INTERRUPTED by SIGINT (exit 130)" in summary
    assert "130 (interrupted:SIGINT)" in summary


def test_closed_stdout_mid_session_completes_cleanly(tmp_path: Path) -> None:
    repo = _make_stub_repo(tmp_path, with_git=False)
    bin_dir = tmp_path / "bin"
    _symlink_python_dir(bin_dir)
    tag = f"tt_closed_stdout_{os.getpid()}"
    stub_body = """\
import json, pathlib, sys, uuid, zipfile
argv = sys.argv[1:]
joined = " ".join(argv)
if argv and argv[0] == "sync":
    sys.stdout.write("short line\\n")
    sys.stdout.write(("Y" * 512 + "\\n") * 250)
    sys.stdout.flush()
    sys.exit(0)
if "sys.executable" in joined and "torch.version.cuda" in joined: print("stub-rt"); sys.exit(0)
if "tree_state_method=" in joined: print("tree_state_method=content_hash tree_state_sig=000000000000"); sys.exit(0)
if "--write-reference" in argv:
    sys.stdout.write("ref short line\\n")
    sys.stdout.write(("R" * 512 + "\\n") * 250)
    sys.stdout.flush()
    ref_path = pathlib.Path(argv[argv.index("--write-reference") + 1])
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    spec_doc = {"spec": {"schema_version": 5}, "environment": {"archive_id": uuid.uuid4().hex}}
    entries_doc = {
        "tiny_random:s": {"tensor_member": "tensors/0000.bin", "tensor_dtype": "float32", "tensor_shape": [4]}
    }
    with zipfile.ZipFile(ref_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("tensors/0000.bin", b"\\x00" * 16)
        zf.writestr("spec.json", json.dumps(spec_doc).encode())
        zf.writestr("entries.json", json.dumps(entries_doc).encode())
    sys.exit(0)
if "torchrun" in argv:
    sys.stdout.write(("T" * 512 + "\\n") * 250)
    sys.stdout.flush()
    sys.exit(0)
print("done")
sys.exit(0)
"""
    _write_stub_uv(bin_dir, stub_body)
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "_TT_TEST_LEFTOVER_TAG": tag})
    log_dir = tmp_path / "logs"

    proc = subprocess.Popen(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            "--ranks",
            "2",
            "--log-dir",
            str(log_dir),
        ],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        first_line = proc.stdout.readline()
        assert first_line, "driver exited before emitting first stdout line"
        proc.stdout.close()
        _, err = proc.communicate(timeout=30.0)
        assert proc.returncode == 0, err
        assert err == b"", f"unexpected stderr after closing stdout mid-session: {err!r}"
        assert _list_matching_pids(tag) == []
    finally:
        _kill_process_group_quiet(proc)
        _cleanup_tagged_processes(tag)

    summary = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "01_env                               0 " in summary
    assert "03_ref_gloo_tiny_n2                  0 " in summary
    assert "04_gloo_tiny_n2                      0 " in summary
    assert "09_rank_failure_gloo_n2              0 " in summary
    assert len((log_dir / "04_gloo_tiny_n2.log").read_bytes()) > 100_000

    # Also verify a short-output resumed session survives a closed stdout pipe with empty stderr.
    proc2 = subprocess.Popen(
        [
            "/bin/bash",
            str(repo / "scripts" / "multi_gpu_check.sh"),
            "--backend",
            "gloo",
            "--ranks",
            "2",
            "--log-dir",
            str(log_dir),
        ],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert proc2.stdout is not None
        assert proc2.stdout.readline()
        proc2.stdout.close()
        _, err2 = proc2.communicate(timeout=30.0)
        assert proc2.returncode == 0, err2
        assert err2 == b"", f"unexpected stderr on resumed closed stdout: {err2!r}"
        assert _list_matching_pids(tag) == []
    finally:
        _kill_process_group_quiet(proc2)
        _cleanup_tagged_processes(tag)
