"""Check plan isolation, offline execution and failed-stage reporting."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Stage, compiler_environment, repo_root, run_workflow, tool_command


class WorkflowTests(unittest.TestCase):

    def test_plan_does_not_execute_or_create_stage_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = Stage("never", (sys.executable, "-c", "raise RuntimeError('executed')"),
                          root / "stage")
            result = run_workflow([stage], root / "plans")
            self.assertFalse(result["execution_requested"])
            self.assertFalse(stage.cwd.exists())
            self.assertFalse((root / "plans" / "manifest.json").exists())

    def test_execution_preserves_spaces_and_enforces_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = ("import json,os,sys; "
                    "print(json.dumps([sys.argv[1], os.environ['HF_HUB_OFFLINE'], "
                    "os.environ['TRANSFORMERS_OFFLINE']]))")
            stage = Stage("argument", (sys.executable, "-c", code, "path with spaces"), root)
            result = run_workflow([stage], root, execute=True)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(json.loads((root / "argument.log").read_text()),
                             ["path with spaces", "1", "1"])

    def test_failure_is_recorded_and_stops_later_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = [Stage("fail", (sys.executable, "-c", "raise SystemExit(7)"), root),
                      Stage("later", (sys.executable, "-c", "print('unexpected')"), root)]
            with self.assertRaises(subprocess.CalledProcessError):
                run_workflow(stages, root, execute=True)
            result = json.loads((root / "manifest.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["results"][0]["returncode"], 7)
            self.assertFalse((root / "later.log").exists())

    def test_invalid_stage_cannot_escape_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                run_workflow([Stage("../escape", ("unused",), root)], root)

    def test_environment_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("common.compiler_runtime_environment", side_effect=OSError("setup failed")):
                with self.assertRaises(OSError):
                    run_workflow([Stage("never", ("unused",), root)], root, execute=True)
            result = json.loads((root / "manifest.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["results"][0]["stage"], "runtime_environment")

    def test_device_library_selection_is_preserved(self):
        with patch.dict("os.environ", {"USING_CMODEL": "False", "LD_LIBRARY_PATH": "/sdk/lib"}):
            env = compiler_environment()
            self.assertEqual(env["USING_CMODEL"], "False")
            self.assertEqual(env["LD_LIBRARY_PATH"], "/sdk/lib")

    def test_tools_are_resolved_inside_repo(self):
        with tempfile.TemporaryDirectory(prefix="compiler with spaces ") as directory:
            root = Path(directory)
            tools = root / "python/tools"
            tools.mkdir(parents=True)
            script = tools / "llm_convert.py"
            script.touch()
            with patch.dict("os.environ", {"TPU_MLIR_ROOT": str(root)}):
                command = tool_command("llm_convert.py", "path with spaces")
                self.assertEqual(command, (sys.executable, str(script), "path with spaces"))
                with self.assertRaises(ValueError):
                    tool_command("../../other.py")

    def test_external_compiler_environment_and_custom_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install = root / "custom install"
            (root / "python/tools").mkdir(parents=True)
            (install / "bin").mkdir(parents=True)
            with patch.dict("os.environ", {"TPU_MLIR_ROOT": str(root),
                                           "INSTALL_PATH": str(install)}):
                env = compiler_environment()
                self.assertEqual(repo_root(), root)
                self.assertEqual(env["PROJECT_ROOT"], str(root))
                self.assertEqual(env["INSTALL_PATH"], str(install))
                self.assertEqual(env["PATH"].split(":")[0], str(install / "bin"))

    def test_missing_compiler_tool_explains_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"TPU_MLIR_ROOT": directory}):
                with self.assertRaisesRegex(ValueError, "Set TPU_MLIR_ROOT"):
                    tool_command("llm_convert.py")


if __name__ == "__main__":
    unittest.main()
