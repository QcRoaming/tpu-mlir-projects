"""Launch an isolated BM1684X simulator process without rewriting installed libraries."""

import os
from pathlib import Path
import subprocess
import sys

if __package__ in {None, "", "yolov8"}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import compiler_runtime_environment
else:
    from ..common import compiler_runtime_environment


def launch_cmodel_if_needed(script, arguments):
    if os.environ.get("YOLO_CMODEL_LIBRARY_DIR"):
        return
    # The loader reads LD_LIBRARY_PATH at process startup, before Python imports.
    with compiler_runtime_environment(force_cmodel=True) as environment:
        overlay = Path(environment["TPUC_ROOT"])
        directory = overlay / "lib"
        original = directory / "libcmodel_bm1684x.so"
        if not original.is_file() or not overlay.name.startswith("resume-cmodel-"):
            raise RuntimeError("An installed CModel runtime and isolated alias overlay "
                               "are required")
        alias = directory / "libcmodel.so"
        if alias.is_symlink():
            alias.unlink()
        elif alias.exists():
            raise RuntimeError("Refusing to overwrite a non-symlink CModel alias")
        alias.symlink_to(original.resolve())
        bmlib = directory / "libbmlib.so.0"
        default = directory / "libbmlib.so.0.orig"
        if default.is_file() and bmlib.is_symlink():
            bmlib.unlink()
            bmlib.symlink_to(default.resolve())
        environment["YOLO_CMODEL_LIBRARY_DIR"] = str(directory)
        result = subprocess.run([sys.executable, str(Path(script).resolve()), *arguments],
                                env=environment)
        raise SystemExit(result.returncode)
