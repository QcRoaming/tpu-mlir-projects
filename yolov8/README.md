# YOLOv8s 配置与运行

先完成[根 README 的 Python 与 TPU-MLIR 环境配置](../README.md)。以下命令从本仓库根目录执行。默认输入为 RGB、归一化到 `[0,1]` 的 `float32 [1,3,640,640]`，输出为已解码检测头 `[1,84,8400]`，采用按类别 NMS 与原图坐标恢复。

## 1. 准备本地模型与数据

```bash
cd "$APP_ROOT"
export YOLO_ONNX=/data/models/yolov8s.onnx
export YOLO_CALIBRATION=/data/calibration_images
```

已有 ONNX 可直接编译。若使用本地 `yolov8s.pt`，在导出环境准备 `ultralytics` 后，将 `--onnx "$YOLO_ONNX"` 换成 `--weights /data/models/yolov8s.pt`。导出使用静态 batch=1、无内嵌 NMS；入口只接受本地文件。

校准集是图片目录，校准与运行共用 `vision.py` 的 Letterbox/RGB/NCHW 预处理。COCO 类别模型使用 80 类；自定义模型还需设置对应类别数与 COCO category 映射。

## 2. F16 与 INT8 编译

```bash
# 默认仅生成计划；确认后在末尾添加 --execute。
python yolov8/compile.py --onnx "$YOLO_ONNX" \
  --dataset "$YOLO_CALIBRATION" --input-num 100 \
  --precision F16 --output artifacts/yolov8/f16

# F16 执行与数值校验。
python yolov8/compile.py --onnx "$YOLO_ONNX" \
  --dataset "$YOLO_CALIBRATION" --input-num 100 \
  --precision F16 --output artifacts/yolov8/f16 --execute

# INT8 Max 校准。
python yolov8/compile.py --onnx "$YOLO_ONNX" \
  --dataset "$YOLO_CALIBRATION" --input-num 100 \
  --precision INT8 --calibration max \
  --output artifacts/yolov8/int8_max --execute

# INT8 KL 校准与敏感层混合精度回退。
python yolov8/compile.py --onnx "$YOLO_ONNX" \
  --dataset "$YOLO_CALIBRATION" --input-num 100 \
  --precision INT8 --calibration kl --search-mixed --expected-cos 0.99 \
  --output artifacts/yolov8/int8_mixed --execute
```

目标均为 BM1684X。提供 `--dataset` 时，transform/deploy 使用同一预处理输入做输出比对，失败后停止并保留日志。`--qtable /path/to/reviewed.qtable` 可指定浮点层，层名须来自当前 Top MLIR。数值相似度与 COCO mAP 分别验证。

## 3. 配置四路视频

编辑 [`four_streams.json`](four_streams.json)，将四个 `source` 改成实际视频路径或 RTSP URL。相对路径以配置文件所在目录为准。

先使用已有 ONNX 运行 CPU 路径：

```bash
python yolov8/run.py --model "$YOLO_ONNX" --backend onnx \
  --config yolov8/four_streams.json --queue-size 4 --buffers 6 --policy block \
  --metrics artifacts/yolov8/cpu_metrics.json \
  --detections artifacts/yolov8/cpu_detections.jsonl
```

CModel 使用已编译 BModel：

```bash
python yolov8/run.py \
  --model artifacts/yolov8/f16/yolov8_bm1684x_f16.bmodel --backend cmodel \
  --config yolov8/four_streams.json --queue-size 4 --buffers 6 --policy block \
  --metrics artifacts/yolov8/cmodel_metrics.json
```

四路各有一个解码线程，后面共享预处理、同步推理、后处理各一个线程。三条有界队列使用同一策略：`block` 满时等待；`drop-oldest` 淘汰最旧帧并统计丢帧。正常结束满足每路 `submitted = completed + dropped`；推理完成且复制输出后归还输入 BufferPool。

## 4. 切换真实设备

使用配套驱动、BMRuntime/bmlib 和提供 `pyruntime_bm` 的真实 SDK，进入该 SDK 的 Python 运行环境。将 SDK 目录与 Python binding 路径改成自己的安装位置：

```bash
export SOPHON_SDK=/data/sophon_sdk
export USING_CMODEL=False
export TPUC_ROOT="$SOPHON_SDK"
export LD_LIBRARY_PATH="$SOPHON_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import pyruntime_bm'

python yolov8/run.py \
  --model artifacts/yolov8/f16/yolov8_bm1684x_f16.bmodel \
  --backend device --device 0 --config yolov8/four_streams.json \
  --metrics artifacts/yolov8/device_metrics.json
```

SDK 的 Python binding 需先按该 SDK 的安装说明加入 Python 环境。四路输入共享一个所选设备。`--timeout` 可取消队列等待，阻塞中的 native 解码/推理仍要等调用返回；严格终止可将 CLI 放在独立进程中运行。

## 5. COCO 精度评估

```bash
python yolov8/coco_predict.py --model "$YOLO_ONNX" --backend onnx \
  --images /data/coco/val2017 --annotations /data/coco/instances_val2017.json \
  --output artifacts/yolov8/coco_onnx.json

python evaluate_coco.py --annotations /data/coco/instances_val2017.json \
  --results artifacts/yolov8/coco_onnx.json \
  --image-ids artifacts/yolov8/coco_onnx.image_ids.json \
  --output artifacts/yolov8/coco_score.json
```

F16/INT8/混合精度对照时使用同一图片 ID 集合、预处理和后处理阈值。`diagnostics.py` 可比较浮点/量化 NPZ 的 cosine、RMSE 和相对误差，并生成需审阅的候选 qtable。

## 流程图与测试

![YOLO 整体框架](../docs/images/yolov8-architecture.png)

[调度流程图](../docs/images/yolov8-scheduling.svg) · [可编辑 Mermaid](../docs/images/yolov8-scheduling.mmd)

```bash
python -m pytest yolov8/tests -q
python smoke.py --output-dir artifacts/smoke_cpu
```

小合成模型用于验证接口、队列和编译链路。真实 YOLOv8s/COCO 精度与实卡吞吐需另行实测。
