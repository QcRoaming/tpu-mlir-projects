#!/usr/bin/env python3
"""Prepare or execute two local-weight Qwen2.5 compilation variants."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Stage, require_local_dir, run_workflow, tool_command


def build_stages(model_dir: Path, output_dir: Path, seq_length: int = 4096,
                 variants=("bf16", "w4bf16"), group_size: int = 64,
                 validate: bool = False):
    if seq_length < 2:
        raise ValueError("seq_length 必须至少为 2")
    if group_size <= 0:
        raise ValueError("group_size 必须为正数")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen2":
        raise ValueError("本示例接受 Qwen2.5 文本模型（config.model_type=qwen2）")
    # Qwen2.5 uses qwen2 model_type in the local HF configuration.
    if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("pytorch_model*.bin")):
        raise ValueError("本地模型目录缺少 safetensors 或 pytorch_model*.bin 权重")
    stages = []
    for variant in variants:
        if variant not in ("bf16", "w4bf16"):
            raise ValueError(f"不支持的量化方式: {variant}")
        destination = output_dir / variant
        argv = tool_command("llm_convert.py", "--model_path", str(model_dir),
                            "--seq_length", str(seq_length), "--chip", "bm1684x",
                            "--quantize", variant, "--q_group_size", str(group_size),
                            "--batch", "1", "--num_device", "1", "--num_core", "1",
                            "--out_dir", str(destination))
        if validate:
            argv += ("--gen_test_input",)
        stages.append(Stage(f"compile_{variant}", argv, output_dir))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="已经存在的本地 Qwen2.5-7B-Instruct 目录")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen2_5"))
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--variants", nargs="+", choices=("bf16", "w4bf16"),
                        default=["bf16", "w4bf16"])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--validate-blocks", action="store_true",
                        help="显式执行转换时让 model_deploy 校验各子网数值")
    parser.add_argument("--execute", action="store_true", help="执行转换，默认仅生成计划")
    args = parser.parse_args()
    model_dir = require_local_dir(args.model_dir)
    stages = build_stages(model_dir, args.output_dir.resolve(), args.seq_length,
                          args.variants, args.group_size, args.validate_blocks)
    run_workflow(stages, args.output_dir.resolve(), execute=args.execute)


if __name__ == "__main__":
    main()
