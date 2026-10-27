#!/usr/bin/env python3
"""Local tokenizer frontend for the persistent BMRuntime worker."""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PREFIX = "QWEN_JSON "


class Backend:
    """A single local worker process owns a single device/session's KV cache."""
    def __init__(self, executable: Path, bmodel: Path, device: int = 0,
                 timeout: float = 180.0):
        self.timeout = timeout
        self.cached_ids = []
        self.responses = queue.Queue()
        self.process = subprocess.Popen([str(executable), str(bmodel), str(device)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.info = self._response()
            if self.info.get("event") != "ready":
                raise RuntimeError(f"运行时未就绪: {self.info}")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith(PREFIX):
                    try:
                        self.responses.put(json.loads(line[len(PREFIX):]))
                    except ValueError as error:
                        self.responses.put(error)
                else:
                    print(line.rstrip(), file=sys.stderr)
        finally:
            self.responses.put(EOFError("BMRuntime worker 已退出"))

    def _response(self):
        try:
            result = self.responses.get(timeout=self.timeout)
        except queue.Empty as error:
            # Without request IDs, a late reply would otherwise be mistaken for
            # the next request's token. Never reuse an uncertain session.
            self.close()
            raise TimeoutError("等待 BMRuntime 响应超时") from error
        if isinstance(result, BaseException):
            self.close()
            raise result
        if result.get("event") == "error":
            raise RuntimeError(result.get("error", "BMRuntime error"))
        return result

    def command(self, command: str):
        # Only a fully completed generate() can commit a reusable token prefix.
        self.cached_ids = []
        if self.process.poll() is not None:
            raise RuntimeError("BMRuntime worker 已停止")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self._response()

    def prefill(self, ids):
        return self.command("PREFILL " + str(len(ids)) + " " + " ".join(map(str, ids)))

    def append(self, ids, expected_length):
        return self.command("APPEND " + str(expected_length) + " " + str(len(ids)) +
                            " " + " ".join(map(str, ids)))

    def decode(self):
        return self.command("DECODE")

    def reset(self):
        return self.command("RESET")

    def close(self):
        self.cached_ids = []
        if self.process.poll() is None:
            try:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=3)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()


def stop_ids(tokenizer, model_dir: Path):
    """Honor tokenizer EOS and all local generation_config EOS IDs."""
    ids = set()
    value = tokenizer.eos_token_id
    values = value if isinstance(value, (list, tuple)) else [value]
    ids.update(int(x) for x in values if x is not None)
    config_path = model_dir / "generation_config.json"
    if config_path.is_file():
        value = json.loads(config_path.read_text()).get("eos_token_id", [])
        values = value if isinstance(value, list) else [value]
        ids.update(int(x) for x in values if x is not None)
    # Qwen2.5-Instruct ends turns with im_end; never accept the unknown-token fallback.
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        ids.add(int(vocab["<|im_end|>"]))
    if not ids:
        raise ValueError("本地 tokenizer/generation_config 未提供 EOS token")
    return ids


def render_prompt(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, list) or not ids or isinstance(ids[0], list):
        raise ValueError("chat_template 必须生成单条非空 token 序列")
    return [int(x) for x in ids]


def generate(backend, tokenizer, messages, eos_ids, max_new_tokens=128, on_text=None,
             reuse_kv=True):
    """Greedy generation with explicit capacity checks and observable timings."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须为正数")
    start = time.perf_counter()
    ids = render_prompt(tokenizer, messages)
    if any(token < 0 or token >= len(tokenizer) for token in ids):
        raise ValueError("本地 chat_template 返回的 prompt token ID 超出 tokenizer 词表")
    tokenize_ms = (time.perf_counter() - start) * 1000
    seq = int(backend.info["seq_length"])
    if len(ids) >= seq:
        raise ValueError(f"输入含 {len(ids)} tokens，容量为 {seq}；请 /clear 或缩短输入")
    limit = min(max_new_tokens, seq - len(ids))
    generated, timings = [], []
    text = ""
    emitted = ""
    cached = getattr(backend, "cached_ids", [])
    supports_append = backend.info.get("supports_append", False)
    reused = len(cached) if (reuse_kv and supports_append and
                            0 < len(cached) < len(ids) and
                            ids[:len(cached)] == cached) else 0
    # Clear before any inference or callback can fail; commit only after success.
    backend.cached_ids = []
    result = backend.append(ids[reused:], reused) if reused else backend.prefill(ids)
    ttft_ms = (time.perf_counter() - start) * 1000
    reason = "max_new_tokens" if limit == max_new_tokens else "context_limit"
    for step in range(limit):
        if result.get("event") != "token":
            raise RuntimeError(f"运行时返回了非 token 响应: {result}")
        if supports_append and result.get("cached_tokens") != len(ids) + step:
            raise RuntimeError("前端与 worker 的 KV 缓存长度不一致；下次请求将重新 prefill")
        token = int(result["token_id"])
        if token < 0 or token >= len(tokenizer):
            raise RuntimeError(f"模型与 tokenizer 不匹配，token_id={token}")
        generated.append(token)
        timings.append(result)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        # Avoid displaying incomplete UTF-8 byte-token sequences as replacement glyphs.
        stable = text.rstrip("\ufffd")
        if on_text and stable.startswith(emitted):
            on_text(stable[len(emitted):])
            emitted = stable
        if token in eos_ids:
            reason = "eos"
            break
        if step + 1 < limit:
            result = backend.decode()
    if on_text and text.startswith(emitted):
        on_text(text[len(emitted):])
    elapsed_ms = (time.perf_counter() - start) * 1000
    decode_ms = sum(float(t["total_ms"]) for t in timings[1:])
    count = len(timings) - 1
    # The final predicted token has not been fed back into the transformer yet.
    backend.cached_ids = ids + generated[:-1]
    return {"prompt_ids": ids, "generated_ids": generated, "text": text,
            "stop_reason": reason, "prompt_tokens": len(ids), "generated_tokens": len(generated),
            "prefill_mode": "append" if reused else "full", "reused_tokens": reused,
            "new_prefill_tokens": len(ids) - reused, "cached_tokens": len(backend.cached_ids),
            "tokenization_ms": tokenize_ms, "ttft_ms": ttft_ms,
            "end_to_end_ms": elapsed_ms, "decode_steps": count,
            "decode_tokens_per_second": count * 1000 / decode_ms if decode_ms > 0 else None,
            "decode_scope": "worker total_ms summed after first token; "
            "includes device copies/synchronization",
            "stages": timings, "runtime": backend.info}


def append_report(path: Path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="本地 tokenizer 目录；无需加载 Python 模型权重")
    parser.add_argument("--bmodel", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path(__file__).parent / "build/qwen_bmrt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--system", default="你是一个乐于助人的助手。")
    parser.add_argument("--prompt", help="单次问答；缺省为交互模式")
    parser.add_argument("--report", type=Path, help="保存 JSONL：token IDs、逐阶段实测时间和缓存内存")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--reuse-kv", action=argparse.BooleanOptionalAction, default=True,
                        help="前缀一致时复用当前会话 KV；--no-reuse-kv 每轮完整 prefill")
    args = parser.parse_args()
    if args.device < 0 or args.max_new_tokens <= 0 or args.timeout <= 0:
        parser.error("device 必须非负，max-new-tokens 和 timeout 必须为正数")
    for path, directory in ((args.model_dir, True), (args.bmodel, False), (args.worker, False)):
        if not (path.is_dir() if directory else path.is_file()):
            parser.error(f"本地路径不存在: {path}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir.resolve()),
                                              local_files_only=True, trust_remote_code=False)
    eos = stop_ids(tokenizer, args.model_dir)
    backend = Backend(args.worker.resolve(), args.bmodel.resolve(), args.device, args.timeout)
    messages = [{"role": "system", "content": args.system}]
    try:
        print("本地模型已加载。/clear 清空会话，/exit 退出。", file=sys.stderr)
        while True:
            try:
                user_text = args.prompt if args.prompt is not None else input("用户：")
            except (EOFError, KeyboardInterrupt):
                break
            if user_text.strip() == "/exit":
                break
            if user_text.strip() == "/clear":
                backend.reset()
                messages = messages[:1]
                if args.prompt is not None:
                    break
                continue
            if not user_text.strip():
                if args.prompt is not None:
                    break
                continue
            pending = messages + [{"role": "user", "content": user_text}]
            try:
                print("助手：", end="", flush=True)
                report = generate(backend, tokenizer, pending, eos, args.max_new_tokens,
                                  lambda chunk: print(chunk, end="", flush=True),
                                  reuse_kv=args.reuse_kv)
                print()
                messages = pending + [{"role": "assistant", "content": report["text"]}]
                report["bmodel"] = str(args.bmodel.resolve())
                report["bmodel_file_bytes"] = args.bmodel.stat().st_size
                speed = report["decode_tokens_per_second"]
                speed_text = f"{speed:.3f} tokens/s" if speed is not None else "N/A（没有 decode 步骤）"
                print(f"TTFT={report['ttft_ms']:.2f} ms，decode={speed_text}，"
                      f"prefill={report['prefill_mode']}，复用={report['reused_tokens']}，"
                      f"新增输入={report['new_prefill_tokens']}，"
                      f"停止原因={report['stop_reason']}", file=sys.stderr)
                if args.report:
                    append_report(args.report, report)
            except (ValueError, RuntimeError, TimeoutError) as error:
                print(f"\n{error}", file=sys.stderr)
                if args.prompt is not None:
                    raise
                if backend.process.poll() is not None:
                    break
            if args.prompt is not None:
                break
    finally:
        backend.close()


if __name__ == "__main__":
    main()
