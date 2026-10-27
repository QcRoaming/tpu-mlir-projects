#!/usr/bin/env python3
"""Compare real BF16/INT4 JSONL runs; optionally generate a local CPU reference."""
import argparse
import json
import os
import statistics
from pathlib import Path


def read_reports(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"没有报告记录: {path}")
    return rows


def agreement(left, right):
    if left["prompt_ids"] != right["prompt_ids"]:
        raise ValueError("必须使用完全相同的 prompt token IDs 比较")
    a, b = left["generated_ids"], right["generated_ids"]
    width = max(len(a), len(b))
    equal = sum(x == y for x, y in zip(a, b))
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return {"same_sequence": a == b, "common_prefix_tokens": prefix,
            "position_token_agreement": equal / width if width else 1.0,
            "left_tokens": len(a), "right_tokens": len(b)}


def summarize(rows):
    decoding = [stage for row in rows for stage in row["stages"][1:]]
    decode_ms = sum(stage["total_ms"] for stage in decoding)
    return {"samples": len(rows),
            "median_ttft_ms": statistics.median(row["ttft_ms"] for row in rows),
            "median_prefill_ms": statistics.median(row["stages"][0]["total_ms"] for row in rows),
            "decode_tokens_per_second": len(decoding) * 1000 / decode_ms if decode_ms else None,
            "decode_steps": len(decoding), "memory": rows[0]["runtime"],
            "bmodel_file_bytes": rows[0].get("bmodel_file_bytes"),
            "stage_mean_decode_ms": {
                key: statistics.mean(stage[key] for stage in decoding) if decoding else None
                for key in ("prepare_ms", "embedding_ms", "blocks_ms", "head_ms",
                            "cache_update_ms")}}


def local_cpu_reference(rows, model_dir: Path):
    """Explicit optional operation: loads existing weights, never calls a Hub API."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not model_dir.is_dir():
        raise ValueError("CPU reference 必须使用已经存在的本地模型目录")
    model = AutoModelForCausalLM.from_pretrained(str(model_dir.resolve()), local_files_only=True,
                                               trust_remote_code=False, torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()), local_files_only=True,
                                              trust_remote_code=False)
    from chat import stop_ids
    eos = sorted(stop_ids(tokenizer, model_dir))
    references = []
    with torch.inference_mode():
        for row in rows:
            prompt = torch.tensor([row["prompt_ids"]], dtype=torch.long)
            generated = model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                       max_new_tokens=max(1, len(row["generated_ids"])),
                                       do_sample=False,
                                       eos_token_id=eos, pad_token_id=(tokenizer.pad_token_id
                                       if tokenizer.pad_token_id is not None else eos[0]))
            ids = generated[0, prompt.shape[1]:].tolist()
            references.append({"prompt_ids": row["prompt_ids"], "generated_ids": ids,
                               "text": tokenizer.decode(ids, skip_special_tokens=True),
                               "backend": "local CPU FP32 reference"})
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--int4", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("comparison.json"))
    parser.add_argument("--cpu-reference-model", type=Path,
                        help="可选：显式加载本地大模型到 CPU，可能需要约 30 GB+ RAM")
    args = parser.parse_args()
    bf16, int4 = read_reports(args.bf16), read_reports(args.int4)
    if len(bf16) != len(int4):
        raise ValueError("BF16 与 INT4 报告样本数不同")
    result = {"bf16": summarize(bf16), "w4bf16": summarize(int4),
              "token_comparison": [agreement(a, b) for a, b in zip(bf16, int4)],
              "interpretation": "Greedy token agreement is a diagnostic, "
              "not perplexity, logit error or task accuracy."}
    if args.cpu_reference_model:
        reference = local_cpu_reference(bf16, args.cpu_reference_model)
        result["cpu_reference"] = reference
        result["bf16_vs_cpu"] = [agreement(a, b) for a, b in zip(bf16, reference)]
        result["int4_vs_cpu"] = [agreement(a, b) for a, b in zip(int4, reference)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
