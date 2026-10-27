# Qwen2.5-7B-Instruct 配置与运行

先完成[根 README 的 Python 与 TPU-MLIR 环境配置](../README.md)。下列命令从本仓库根目录执行。默认配置为 BM1684X、单设备、batch=1、静态序列容量 4096、greedy 生成，支持 BF16 与 W4BF16。

## 1. 准备本地模型

```bash
cd "$APP_ROOT"
export QWEN_MODEL=/data/models/Qwen2.5-7B-Instruct
python -m pip install transformers==5.6.2
```

`QWEN_MODEL` 必须包含 `config.json`（`model_type=qwen2`）、完整 safetensors 或 PyTorch 权重及 tokenizer 文件。编译所需 PyTorch 与其他转换依赖沿用 TPU-MLIR 的环境；入口不下载模型。Python 聊天入口只加载本地 tokenizer。

## 2. 生成与执行编译计划

```bash
# 只生成计划，检查 artifacts/qwen2_5/plan.json。
python qwen2_5/compile.py --model-dir "$QWEN_MODEL" \
  --output-dir artifacts/qwen2_5

# 执行 BF16 与 W4BF16 转换，并请求逐子网数值校验。
python qwen2_5/compile.py --model-dir "$QWEN_MODEL" \
  --output-dir artifacts/qwen2_5 --validate-blocks --execute
```

默认 INT4 权重量化分组大小为 64，可通过 `--group-size` 修改；只需要一种精度时传 `--variants w4bf16` 或 `--variants bf16`。输出位于 `bf16/` 与 `w4bf16/`，运行时选择各自目录下最终组合的 BModel。

worker 对应静态 embedding、block、block_cache 和融合 greedy LM head 的子网签名。动态序列、采样、多设备、磁盘 embedding 等会改变接口，当前入口未启用这些配置。

## 3. 构建 C++ worker

`NNTOOLCHAIN_ROOT` 应包含 `include/bmruntime_interface.h`、BMRuntime 与 bmlib 的 `lib/`。实卡使用设备配套 SDK；构建检查也可使用外部 TPU-MLIR 的 `third_party/nntoolchain`。

```bash
export NNTOOLCHAIN_ROOT=/data/sophon_sdk
cmake -S qwen2_5 -B qwen2_5/build \
  -DNNTOOLCHAIN_ROOT="$NNTOOLCHAIN_ROOT"
cmake --build qwen2_5/build -j2
```

这里只构建应用的 `qwen_bmrt`，不构建 TPU-MLIR 编译器。

## 4. 启动对话

在真实 SDK 的 BM1684X 运行环境中执行，将 BModel 路径换成最终组合模型：

```bash
export USING_CMODEL=False
export TPUC_ROOT="$NNTOOLCHAIN_ROOT"
export LD_LIBRARY_PATH="$NNTOOLCHAIN_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python qwen2_5/chat.py --model-dir "$QWEN_MODEL" \
  --bmodel /data/models/qwen_final_w4bf16.bmodel --device 0 \
  --worker qwen2_5/build/qwen_bmrt --max-new-tokens 128 \
  --report artifacts/qwen2_5/int4.jsonl
```

缺省进入交互对话；`--prompt '解释 KV Cache'` 执行单次问答。`/clear` 清空会话与设备 KV，`/exit` 退出。上下文容量不足会明确报错或限制输出长度，不静默截断历史。

默认跨轮复用当前会话 KV：完整模板的 token 前缀逐项一致时，worker 通过 APPEND 串行处理新增后缀；前缀不匹配时完整 Prefill。`--no-reuse-kv` 可关闭复用用于对照。生成 N 个输出后，最后一个输出还未进入 KV。

## 5. BF16 与 INT4 对照

用相同 prompt、tokenizer 和生成上限，分别运行两种 BModel，保存 `bf16.jsonl` 与 `int4.jsonl` 后比较：

```bash
python qwen2_5/compare.py --bf16 artifacts/qwen2_5/bf16.jsonl \
  --int4 artifacts/qwen2_5/int4.jsonl \
  --output artifacts/qwen2_5/comparison.json
```

报告包含真实 TTFT、decode 耗时、token IDs、复用长度与样例 KV/设备缓冲字节。token 一致率不能替代模型任务精度，样例缓冲字节不等于设备峰值内存。真实 7B 精度与设备性能仍需实测。

## 流程图与测试

![Qwen 整体框架](../docs/images/qwen-architecture.png)

[调度流程图](../docs/images/qwen-scheduling.svg) · [可编辑 Mermaid](../docs/images/qwen-scheduling.mmd)

```bash
python -m pytest qwen2_5/tests -q
```

这些 host 测试使用假 tokenizer/worker，检查容量、EOS、前缀复用、全量回退和 IPC 协议，不加载 7B 权重。
