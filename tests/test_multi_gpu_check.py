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
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

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
    """Return live non-zombie process command lines containing `tag` (excluding this process)."""
    self_pid = str(os.getpid())
    ps_bin = shutil.which("ps") or ("/bin/ps" if os.path.isfile("/bin/ps") else None)
    if ps_bin is not None:
        out = subprocess.run(
            [ps_bin, "-axo", "pid=,stat=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if out.returncode == 0:
            hits: list[str] = []
            for line in out.stdout.splitlines():
                parts = line.strip().split(None, 2)
                if len(parts) < 3 or parts[0] == self_pid or parts[1].startswith("Z"):
                    continue
                if tag in parts[2] and "ps -axo" not in parts[2]:
                    hits.append(line.strip())
            return hits
    if os.path.isdir("/proc"):
        hits = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or entry == self_pid:
                continue
            try:
                stat = Path(f"/proc/{entry}/stat").read_bytes()
                rparen = stat.rfind(b")")
                if rparen >= 0 and stat[rparen + 2 :].split()[0] == b"Z":
                    continue
                cmd = Path(f"/proc/{entry}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if tag in cmd:
                    hits.append(f"{entry} {cmd}")
            except OSError:
                continue
        return hits
    return []


def _wait_for_path(path: Path, proc: subprocess.Popen[Any], *, timeout: float = 20.0) -> None:
    """Wait until `path` exists or fail fast if `proc` exits prematurely or `timeout` elapses."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if proc.poll() is not None:
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
        ["--step-timeout", "inf"],
        ["--stage-timeout", "-inf"],
        ["--stage-timeout", "1" + "0" * 400],
        ["--ranks"],
        ["--ranks", ","],
        ["--ranks", "1"],
        ["--ranks", "abc"],
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
    # Spawn a grandchild in its own new session that ignores SIGTERM and inherits _TIMESTRAL_STAGE_TOKEN.
    gc_code = (
        "import pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(grandchild_ready)!r}).write_text('ready'); "
        "time.sleep(60)"
    )
    subprocess.Popen([sys.executable, "-c", gc_code, {tag!r}], start_new_session=True)
    # Parent ignores SIGTERM after dumping >128 KiB to stdout (larger than OS pipe buffer).
    def _on_term(_sig, _frame):
        sys.stdout.write(("X" * 1024 + "\\n") * 160 + {dump_sentinel!r} + "\\n")
        sys.stdout.flush()
    signal.signal(signal.SIGTERM, _on_term)
    while not pathlib.Path({str(grandchild_ready)!r}).exists():
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
    env = _clean_tracked_env({"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"})

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
    finally:
        _kill_process_group_quiet(proc)

    assert proc.returncode == 1, out
    assert grandchild_ready.exists(), "grandchild never started"
    summary = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "03_ref_gloo_tiny_n2                  124 " in summary
    assert "04_gloo_tiny_n2                      SKIPPED (ref failed)" in summary
    assert "09_rank_failure_gloo_n2              0 " in summary
    ref_log = (log_dir / "03_ref_gloo_tiny_n2.log").read_text(encoding="utf-8")
    assert dump_sentinel in ref_log
    assert "timed out after 1s" in ref_log
    assert not (log_dir / "03_ref_gloo_tiny_n2.done").exists()
    assert not (log_dir / "refs" / "ref_gloo_tiny_n2.zip").exists()
    assert _list_matching_pids(tag) == []


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
            # Write a partial .tmp file to verify interrupted stages clean up partial artifacts.
            ref_path.parent.mkdir(parents=True, exist_ok=True)
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

    stage_timeout = "1" if moment == "timeout_teardown" else "30"
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
    finally:
        _kill_process_group_quiet(proc)

    assert _list_matching_pids(tag) == []
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
