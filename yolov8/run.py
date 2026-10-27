"""Run bounded multi-stream detection using existing local models or a fixture."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, "", "yolov8"}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from yolov8.backends import BmodelBackend, OnnxBackend
    from yolov8.pipeline import Pipeline, SyntheticSource, VideoSource
    from yolov8.runtime import launch_cmodel_if_needed
else:
    from .backends import BmodelBackend, OnnxBackend
    from .pipeline import Pipeline, SyntheticSource, VideoSource
    from .runtime import launch_cmodel_if_needed


def sources_from_args(args):
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        entries = config["streams"]
        names = [str(entry["id"]) for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Stream ids must be unique")
        sources = {}
        for entry in entries:
            name = str(entry["id"])
            if entry.get("synthetic", False):
                count = int(entry.get("frames", 16))
                if count < 1:
                    raise ValueError("Synthetic stream frame counts must be positive")
                sources[name] = SyntheticSource(count, stream=len(sources))
            else:
                source = str(entry["source"])
                if not source.startswith(("rtsp://", "rtsps://")):
                    source = str((args.config.parent / source).resolve())
                    if not Path(source).is_file():
                        raise ValueError(f"Local video does not exist: {source}")
                sources[name] = VideoSource(source, args.max_frames)
        return sources
    if args.sources:
        result = {}
        for index, source in enumerate(args.sources):
            if not source.startswith(("rtsp://", "rtsps://")):
                source = str(Path(source).resolve())
                if not Path(source).is_file():
                    raise ValueError(f"Local video does not exist: {source}")
            result[f"stream{index}"] = VideoSource(source, args.max_frames)
        return result
    return {f"stream{index}": SyntheticSource(args.synthetic_frames, stream=index)
            for index in range(args.streams)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("onnx", "cmodel", "device"), default="onnx")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-name")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--sources", nargs="+")
    inputs.add_argument("--config", type=Path)
    inputs.add_argument("--synthetic-frames", type=int, default=16)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--queue-size", type=int, default=4)
    parser.add_argument("--policy", choices=("block", "drop-oldest"), default="block")
    parser.add_argument("--buffers", type=int)
    parser.add_argument("--classes", type=int, default=80)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--detections", type=Path)
    args = parser.parse_args()
    if min(args.streams, args.synthetic_frames, args.queue_size, args.timeout, args.classes) <= 0:
        parser.error("Frame/stream/queue/class counts and timeout must be positive")
    if args.max_frames < 0 or (args.buffers is not None and args.buffers < 1):
        parser.error("--max-frames must be nonnegative and --buffers must be positive")
    if not args.model.is_file():
        parser.error("--model must identify an existing local file")
    if args.backend == "cmodel":
        launch_cmodel_if_needed(__file__, sys.argv[1:])
    sources = sources_from_args(args)
    backend = (OnnxBackend(args.model, args.output_name) if args.backend == "onnx" else
               BmodelBackend(args.model, args.backend, args.device, args.output_name))
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    sink_file = None
    if args.detections:
        args.detections.parent.mkdir(parents=True, exist_ok=True)
        sink_file = args.detections.open("w", encoding="utf-8")

    def sink(packet, detections):
        if sink_file is not None:
            sink_file.write(json.dumps(dict(stream=packet.stream, frame=packet.index,
                                             detections=detections.tolist())) + "\n")

    pipeline = Pipeline(backend, sources, args.queue_size, args.policy, args.buffers,
                        args.classes, args.confidence, args.iou, sink)
    try:
        report = pipeline.run(args.timeout)
    except BaseException:
        report = pipeline.report()
        report.update(status="failed", backend=args.backend, model=str(args.model.resolve()))
        args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    finally:
        if sink_file is not None:
            sink_file.close()
    source_kinds = {name: "synthetic" if isinstance(source, SyntheticSource) else "video"
                    for name, source in sources.items()}
    report.update(status="completed", backend=args.backend, model=str(args.model.resolve()),
                  synthetic=all(kind == "synthetic" for kind in source_kinds.values()),
                  source_kinds=source_kinds, queue_policy=args.policy)
    args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
