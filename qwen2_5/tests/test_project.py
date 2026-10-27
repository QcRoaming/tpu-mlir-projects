import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import chat
import compare

# The CLI is named compile.py; import under an unambiguous name for tests.
spec = importlib.util.spec_from_file_location("qwen_compile", PROJECT / "compile.py")
compile_project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compile_project)


class Tokenizer:
    eos_token_id = 99
    def __init__(self, prompt=None):
        self.prompt = prompt or [1, 2]
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return self.prompt
    def get_vocab(self):
        return {"<|im_end|>": 98}
    def decode(self, ids, skip_special_tokens):
        return "".join(str(x) for x in ids if x not in (98, 99))
    def __len__(self):
        return 128


class FakeBackend:
    def __init__(self, tokens=(10, 11, 99), seq_length=8, supports_append=True):
        self.tokens = iter(tokens)
        self.info = {"seq_length": seq_length, "kv_bytes": 256,
                     "supports_append": supports_append}
        self.calls = []
        self.cached_ids, self.resident_ids = [], []
        self.pending_token = None
        self.fail_phase = self.bad_length_phase = None
    def _step(self, phase):
        if phase == self.fail_phase:
            raise RuntimeError("injected worker failure")
        self.pending_token = next(self.tokens)
        return {"event": "token", "phase": phase, "token_id": self.pending_token,
                "cached_tokens": len(self.resident_ids) + (phase == self.bad_length_phase),
                "total_ms": 50 if phase == "append" else 5, "prepare_ms": 1,
                "embedding_ms": 1, "blocks_ms": 2, "head_ms": 1, "cache_update_ms": 0}
    def prefill(self, ids):
        self.cached_ids = []
        self.resident_ids = list(ids)
        self.calls.append(("prefill", list(ids)))
        return self._step("prefill")
    def append(self, ids, expected_length):
        self.cached_ids = []
        assert expected_length == len(self.resident_ids)
        self.calls.append(("append", list(ids), expected_length))
        self.resident_ids.extend(ids)
        return self._step("append")
    def decode(self):
        self.cached_ids = []
        assert self.pending_token is not None
        self.calls.append(("decode",))
        self.resident_ids.append(self.pending_token)
        return self._step("decode")
    def reset(self):
        self.cached_ids, self.resident_ids = [], []
        self.pending_token = None


class GenerationTests(unittest.TestCase):
    def test_eos_and_decode_metrics(self):
        backend = FakeBackend()
        emitted = []
        report = chat.generate(backend, Tokenizer(), [], {99}, 5, emitted.append)
        self.assertEqual(report["generated_ids"], [10, 11, 99])
        self.assertEqual(report["stop_reason"], "eos")
        self.assertEqual(report["decode_steps"], 2)
        self.assertEqual(report["decode_tokens_per_second"], 200)
        self.assertEqual(backend.calls, [("prefill", [1, 2]), ("decode",), ("decode",)])
        self.assertEqual(backend.cached_ids, backend.resident_ids)
        self.assertEqual(backend.cached_ids, [1, 2, 10, 11])
        self.assertEqual("".join(emitted), report["text"])

    def test_context_budget_prevents_extra_decode(self):
        backend = FakeBackend(tokens=(10,), seq_length=3)
        report = chat.generate(backend, Tokenizer(), [], {99}, 100)
        self.assertEqual(report["stop_reason"], "context_limit")
        self.assertEqual(report["decode_steps"], 0)
        self.assertIsNone(report["decode_tokens_per_second"])
        self.assertEqual(len(backend.calls), 1)

    def test_overflow_rejected_before_worker_request(self):
        backend = FakeBackend(seq_length=2)
        with self.assertRaises(ValueError):
            chat.generate(backend, Tokenizer(), [], {99}, 1)
        self.assertEqual(backend.calls, [])

    def test_invalid_output_token_fails(self):
        with self.assertRaises(RuntimeError):
            chat.generate(FakeBackend(tokens=(999,)), Tokenizer(), [], {99}, 1)

    def test_invalid_prompt_token_rejected_before_prefill(self):
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            chat.generate(backend, Tokenizer(prompt=[999]), [], {99}, 1)
        self.assertEqual(backend.calls, [])

    def test_three_turns_append_exact_suffix_including_previous_eos(self):
        backend = FakeBackend(tokens=(10, 99, 11, 12, 99, 13, 99), seq_length=32)
        tokenizer = Tokenizer()
        prompts = ([1, 2], [1, 2, 10, 99, 3, 4], [1, 2, 10, 99, 3, 4, 11, 12, 99, 5])
        for prompt, reused, added in zip(prompts, (0, 3, 8), (2, 3, 2)):
            tokenizer.prompt = prompt
            report = chat.generate(backend, tokenizer, [], {99}, 8)
            self.assertEqual(report["prefill_mode"], "append" if reused else "full")
            self.assertEqual((report["reused_tokens"], report["new_prefill_tokens"]),
                             (reused, added))
            self.assertEqual(report["cached_tokens"], len(backend.resident_ids))
            self.assertEqual(backend.cached_ids, backend.resident_ids)
            self.assertEqual(backend.pending_token, 99)
            self.assertEqual(report["decode_tokens_per_second"], 200)
        self.assertEqual([call for call in backend.calls if call[0] == "append"],
                         [("append", [99, 3, 4], 3), ("append", [99, 5], 8)])

    def test_first_eos_and_generation_limit_leave_last_output_for_next_append(self):
        backend = FakeBackend(tokens=(99, 10, 99), seq_length=16)
        tokenizer = Tokenizer()
        for prompt, reason in (([1, 2], "eos"), ([1, 2, 99, 3], "max_new_tokens"),
                               ([1, 2, 99, 3, 10, 4], "eos")):
            tokenizer.prompt = prompt
            report = chat.generate(backend, tokenizer, [], {99}, 1)
            self.assertEqual(report["stop_reason"], reason)
            self.assertEqual(backend.cached_ids, prompt)
            self.assertEqual(backend.resident_ids, prompt)
        self.assertEqual(backend.calls, [("prefill", [1, 2]), ("append", [99, 3], 2),
                                         ("append", [10, 4], 4)])

    def test_incompatible_prefix_and_disabled_or_legacy_reuse_prefill(self):
        cases = [([1, 2, 10], True, True), ([1, 2], True, True),
                 ([1, 3, 4, 5], True, True), ([1, 2, 10, 99, 3], False, True),
                 ([1, 2, 10, 99, 3], True, False)]
        for prompt, reuse, capability in cases:
            with self.subTest(prompt=prompt, reuse=reuse, capability=capability):
                backend = FakeBackend(tokens=(10, 99, 99), seq_length=16,
                                      supports_append=capability)
                if not capability:
                    del backend.info["supports_append"]
                chat.generate(backend, Tokenizer(), [], {99}, 4)
                report = chat.generate(backend, Tokenizer(prompt), [], {99}, 4, reuse_kv=reuse)
                self.assertEqual(report["prefill_mode"], "full")
                self.assertEqual(backend.calls[-1], ("prefill", prompt))
                self.assertEqual(report["reused_tokens"], 0)

    def test_reset_forces_prefill_and_overflow_sends_nothing(self):
        backend = FakeBackend(tokens=(10, 99, 99), seq_length=16)
        chat.generate(backend, Tokenizer(), [], {99}, 4)
        calls, cached = list(backend.calls), list(backend.cached_ids)
        with self.assertRaises(ValueError):
            chat.generate(backend, Tokenizer(list(range(16))), [], {99}, 4)
        self.assertEqual(backend.calls, calls)
        self.assertEqual(backend.cached_ids, cached)
        backend.reset()
        prompt = [1, 2, 10, 99, 3]
        report = chat.generate(backend, Tokenizer(prompt), [], {99}, 4)
        self.assertEqual(report["prefill_mode"], "full")

    def test_worker_failure_or_wrong_cache_length_invalidates_reuse(self):
        for phase, failure in (("append", "fail_phase"), ("decode", "fail_phase"),
                               ("prefill", "bad_length_phase"), ("append", "bad_length_phase"),
                               ("decode", "bad_length_phase")):
            with self.subTest(phase=phase, failure=failure):
                backend = FakeBackend(tokens=(10, 99, 11, 99, 99), seq_length=16)
                prompt = [1, 2]
                if phase != "prefill":
                    chat.generate(backend, Tokenizer(), [], {99}, 4)
                    prompt = [1, 2, 10, 99, 3]
                setattr(backend, failure, phase)
                with self.assertRaises(RuntimeError):
                    chat.generate(backend, Tokenizer(prompt), [], {99}, 4)
                self.assertEqual(backend.cached_ids, [])
                setattr(backend, failure, None)
                report = chat.generate(backend, Tokenizer(prompt), [], {99}, 1)
                self.assertEqual(report["prefill_mode"], "full")

    def test_output_callback_failure_discards_successfully_appended_cache(self):
        backend = FakeBackend(tokens=(10, 99, 11, 99), seq_length=16)
        chat.generate(backend, Tokenizer(), [], {99}, 4)
        prompt = [1, 2, 10, 99, 3]
        def fail_text(chunk):
            raise RuntimeError("output callback failed")
        with self.assertRaisesRegex(RuntimeError, "output callback failed"):
            chat.generate(backend, Tokenizer(prompt), [], {99}, 4, on_text=fail_text)
        self.assertEqual(backend.calls[-1], ("append", [99, 3], 3))
        self.assertEqual(backend.resident_ids, prompt)
        self.assertEqual(backend.cached_ids, [])
        report = chat.generate(backend, Tokenizer(prompt), [], {99}, 4)
        self.assertEqual(report["prefill_mode"], "full")
        self.assertEqual(backend.calls[-1], ("prefill", prompt))

    def test_cli_exposes_both_cache_choices(self):
        result = subprocess.run([sys.executable, str(PROJECT / "chat.py"), "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("--reuse-kv", result.stdout)
        self.assertIn("--no-reuse-kv", result.stdout)

    def test_stop_ids_include_local_config_and_qwen_turn_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "generation_config.json").write_text('{"eos_token_id": [97, 99]}')
            self.assertEqual(chat.stop_ids(Tokenizer(), path), {97, 98, 99})


class CompilationTests(unittest.TestCase):
    def test_variants_follow_runtime_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{"model_type":"qwen2"}')
            (path / "model.safetensors").touch()
            tools = path / "python/tools"
            tools.mkdir(parents=True)
            (tools / "llm_convert.py").touch()
            with patch.dict("os.environ", {"TPU_MLIR_ROOT": str(path)}):
                stages = compile_project.build_stages(path, path / "out")
            self.assertEqual([stage.name for stage in stages], ["compile_bf16", "compile_w4bf16"])
            for stage, mode in zip(stages, ("bf16", "w4bf16")):
                argv = list(stage.argv)
                self.assertEqual(argv[argv.index("--chip") + 1], "bm1684x")
                self.assertEqual(argv[argv.index("--quantize") + 1], mode)
                self.assertEqual(argv[argv.index("--seq_length") + 1], "4096")
                self.assertEqual(argv[argv.index("--batch") + 1], "1")
                for flag in ("--dynamic", "--do_sample", "--embedding_disk", "--use_history_kv"):
                    self.assertNotIn(flag, argv)

    def test_missing_local_weights_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{"model_type":"qwen2"}')
            with self.assertRaises(ValueError):
                compile_project.build_stages(path, path / "out")


class ComparisonTests(unittest.TestCase):
    def test_matching_and_divergent_tokens(self):
        a = {"prompt_ids": [1], "generated_ids": [10, 11, 99]}
        b = {"prompt_ids": [1], "generated_ids": [10, 12, 99, 99]}
        result = compare.agreement(a, b)
        self.assertEqual(result["common_prefix_tokens"], 1)
        self.assertEqual(result["position_token_agreement"], .5)
        self.assertFalse(result["same_sequence"])

    def test_prompt_mismatch_is_not_compared(self):
        with self.assertRaises(ValueError):
            compare.agreement({"prompt_ids": [1]}, {"prompt_ids": [2]})


class ProtocolTests(unittest.TestCase):
    def make_worker(self, path, delay=0):
        # This is a protocol test, not a TPU simulator or a performance result.
        path.write_text("#!" + sys.executable + "\n" +
                        "import json, sys, time\n"
                        "def reply(obj): print('QWEN_JSON ' + json.dumps(obj), flush=True)\n"
                        "print('SDK diagnostic', flush=True)\n"
                        "reply({'event':'ready','seq_length':8,'supports_append':True})\n"
                        "cached = 0\n"
                        "for line in sys.stdin:\n"
                        "    if line.startswith('QUIT'): break\n"
                        f"    time.sleep({delay!r})\n"
                        "    parts = line.split()\n"
                        "    if parts[0] == 'RESET':\n"
                        "        cached = 0\n"
                        "        reply({'event':'reset','cached_tokens':cached})\n"
                        "        continue\n"
                        "    if parts[0] == 'PREFILL': cached = int(parts[1])\n"
                        "    elif parts[0] == 'APPEND':\n"
                        "        expected, count = map(int, parts[1:3])\n"
                        "        if expected != cached or count != len(parts[3:]):\n"
                        "            cached = 0\n"
                        "            reply({'event':'error','error':'invalid APPEND state'})\n"
                        "            continue\n"
                        "        cached += count\n"
                        "    elif parts[0] == 'DECODE':\n"
                        "        if not cached:\n"
                        "            reply({'event':'error','error':'empty cache'})\n"
                        "            continue\n"
                        "        cached += 1\n"
                        "    reply({'event':'token','token_id':10,'cached_tokens':cached,\n"
                        "           'total_ms':1,'request':line.strip()})\n")
        path.chmod(0o755)

    def test_subprocess_ready_prefill_append_decode_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "fake_worker"
            self.make_worker(worker)
            backend = chat.Backend(worker, Path(directory) / "unused.bmodel", timeout=2)
            try:
                self.assertEqual(backend.info["seq_length"], 8)
                self.assertEqual(backend.prefill([1, 2])["token_id"], 10)
                backend.cached_ids = [1, 2]
                appended = backend.append([10, 98, 3], expected_length=2)
                self.assertEqual(appended["request"], "APPEND 2 3 10 98 3")
                self.assertEqual(appended["cached_tokens"], 5)
                self.assertEqual(backend.cached_ids, [])
                self.assertEqual(backend.decode()["cached_tokens"], 6)
                backend.cached_ids = [1, 2, 10, 98, 3, 10]
                self.assertEqual(backend.reset()["cached_tokens"], 0)
                self.assertEqual(backend.cached_ids, [])
                backend.cached_ids = [1]
                with self.assertRaises(RuntimeError):
                    backend.append([3], expected_length=999)
                self.assertEqual(backend.cached_ids, [])
                with self.assertRaises(RuntimeError):
                    backend.decode()
                backend.cached_ids = [1]
            finally:
                backend.close()
            self.assertEqual(backend.cached_ids, [])
            self.assertIsNotNone(backend.process.poll())

    def test_timeout_invalidates_process_and_late_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "fake_worker"
            self.make_worker(worker, delay=.15)
            backend = chat.Backend(worker, Path(directory) / "unused.bmodel", timeout=2)
            backend.timeout = .02
            backend.cached_ids = [1]
            with self.assertRaises(TimeoutError):
                backend.prefill([1])
            self.assertEqual(backend.cached_ids, [])
            self.assertIsNotNone(backend.process.poll())
            with self.assertRaises(RuntimeError):
                backend.decode()


if __name__ == "__main__":
    unittest.main()
