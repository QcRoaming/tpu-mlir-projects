"""Offline compilation plans shared by the resume project samples.

Plans only describe commands unless the caller explicitly requests execution.
This module has no ML framework or TPU runtime import-time dependencies.
"""

import dataclasses
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Iterable


def repo_root() -> Path:
    """Use an external compiler checkout; CPU-only workflows need no checkout."""
    configured = os.environ.get("TPU_MLIR_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent


def require_local_file(path) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise ValueError(f"A local file is required: {path}")
    return value


def require_local_dir(path) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise ValueError(f"A local directory is required: {path}")
    return value


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with require_local_file(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_command(name: str, *args) -> tuple[str, ...]:
    if Path(name).name != name or not name.endswith(".py"):
        raise ValueError("Tool names must identify a Python script in python/tools")
    script = repo_root() / "python" / "tools" / name
    if not script.is_file():
        raise ValueError(
            f"Compiler tool not found: {script}. Set TPU_MLIR_ROOT to an existing "
            "TPU-MLIR source checkout containing python/tools and a usable install/.")
    return (sys.executable, str(script), *(str(arg) for arg in args))


def compiler_environment() -> dict[str, str]:
    """Use existing build artifacts without invoking envsetup download hooks."""
    env = os.environ.copy()
    root = repo_root()
    install = Path(env.get("INSTALL_PATH", str(root / "install"))).expanduser().resolve()

    def prepend(key, candidates):
        existing = [str(path) for path in candidates if path.is_dir()]
        if env.get(key):
            existing.append(env[key])
        env[key] = os.pathsep.join(existing)

    env.update({
        "PROJECT_ROOT": str(root),
        "INSTALL_PATH": str(install),
        "TPUC_ROOT": str(install),
        "REGRESSION_PATH": str(root / "regression"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })
    prepend("PATH", [install / "bin", root / "python" / "tools", root / "python" / "utils"])
    prepend("PYTHONPATH", [root / "python", install / "python"])
    # A caller who selected the real SDK keeps its library order and device mode.
    if env.get("USING_CMODEL", "True").lower() in ("true", "1"):
        prepend("LD_LIBRARY_PATH", [install / "lib", root / "capi" / "lib"])
        env["USING_CMODEL"] = "True"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("FORBID_GEN_RISCV_CODE", "1")
    return env


@contextmanager
def compiler_runtime_environment(force_cmodel: bool = False):
    """Isolate the shared simulator aliases that model_runner switches on load."""
    env = compiler_environment()
    if force_cmodel:
        env["USING_CMODEL"] = "True"
    install = Path(env["INSTALL_PATH"])
    if (env.get("USING_CMODEL", "True").lower() not in ("true", "1") or
            not (install / "lib").is_dir()):
        yield env
        return
    # The existing model_runner contains unquoted alias-switch shell commands.
    # A generated /tmp path has no spaces, including when the user's output path does.
    with tempfile.TemporaryDirectory(prefix="resume-cmodel-") as directory:
        overlay = Path(directory)
        library_dir = overlay / "lib"
        library_dir.mkdir()
        for entry in install.iterdir():
            if entry.name != "lib":
                (overlay / entry.name).symlink_to(entry.resolve(),
                                                  target_is_directory=entry.is_dir())
        for entry in (install / "lib").iterdir():
            (library_dir / entry.name).symlink_to(entry.resolve(),
                                                 target_is_directory=entry.is_dir())
        original_bmlib = library_dir / "libbmlib.so.0.orig"
        if not original_bmlib.exists() and (install / "lib" / "libbmlib.so.0").exists():
            if original_bmlib.is_symlink():
                original_bmlib.unlink()
            original_bmlib.symlink_to((install / "lib" / "libbmlib.so.0").resolve())
        env["TPUC_ROOT"] = str(overlay)
        env["LD_LIBRARY_PATH"] = str(library_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        yield env


@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]
    cwd: Path


def _write_json(path: Path, value):
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_workflow(stages: Iterable[Stage], output_dir: Path, execute: bool = False) -> dict:
    """Persist an inspectable plan and, optionally, sequential logged commands.

    A failed command stops the workflow and leaves its status and log on disk.
    Child processes use argv directly; commands never go through a shell.
    """
    stages = list(stages)
    names = set()
    for stage in stages:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stage.name) or stage.name in names:
            raise ValueError(f"Invalid or duplicate stage name: {stage.name}")
        if not stage.argv or any(not isinstance(arg, str) or "\0" in arg for arg in stage.argv):
            raise ValueError(f"Invalid command arguments for {stage.name}")
        names.add(stage.name)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler = repo_root() if (repo_root() / "python" / "tools").is_dir() else None
    try:
        if compiler is None:
            raise OSError("CPU-only workflow: no compiler configured")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(),
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    plan = {
        "compiler_root": str(compiler) if compiler is not None else None,
        "compiler_commit": commit,
        "python": sys.executable,
        "offline": True,
        "execution_requested": bool(execute),
        "stages": [{"name": stage.name, "argv": list(stage.argv),
                    "cwd": str(Path(stage.cwd).resolve())} for stage in stages],
    }
    _write_json(output_dir / "plan.json", plan)
    for stage in stages:
        print(f"[{stage.name}] (cwd={Path(stage.cwd).resolve()}) {shlex.join(stage.argv)}")
    if not execute:
        return plan

    manifest = {"plan": plan, "status": "running", "results": []}
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    try:
        with compiler_runtime_environment() as environment:
            return _execute_stages(stages, manifest, manifest_path, output_dir, environment)
    except BaseException as error:
        if manifest["status"] != "failed":
            manifest["status"] = "failed"
            manifest["results"].append({"stage": "runtime_environment", "returncode": None,
                                        "error": str(error)})
            _write_json(manifest_path, manifest)
        raise


def _execute_stages(stages, manifest, manifest_path, output_dir, environment):
    for stage in stages:
        log_path = output_dir / f"{stage.name}.log"
        start = time.monotonic()
        result = {"stage": stage.name, "log": str(log_path), "returncode": None}
        try:
            Path(stage.cwd).mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(list(stage.argv), cwd=stage.cwd, env=environment,
                                         stdout=log, stderr=subprocess.STDOUT, check=False)
            result["returncode"] = process.returncode
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, stage.argv)
        except BaseException as error:
            result["elapsed_seconds"] = time.monotonic() - start
            result["error"] = str(error)
            manifest["results"].append(result)
            manifest["status"] = "failed"
            _write_json(manifest_path, manifest)
            print(f"Compilation stopped at {stage.name}; see {log_path}", file=sys.stderr)
            raise
        result["elapsed_seconds"] = time.monotonic() - start
        manifest["results"].append(result)
        _write_json(manifest_path, manifest)
    manifest["status"] = "passed"
    _write_json(manifest_path, manifest)
    return manifest
