#!/usr/bin/env python3
"""Summarize real Top/Tpu IR, layer groups and assigned memory addresses."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def layer_group_info(attribute: str) -> dict:
    """Extract the documented scalar/slice fields from a parsed LG attribute."""
    fields = {}
    for key in ("out_addr", "out_size", "buffer_addr", "buffer_size", "id", "stage",
                "slice_idx", "group_type"):
        match = re.search(r"\b" + key + r"\s*=\s*(-?\d+)", attribute)
        if match:
            fields[key] = int(match.group(1))
    for dimension in "ncdhw":
        for suffix in ("idx", "slice"):
            key = f"{dimension}_{suffix}"
            match = re.search(r"\b" + key + r"\s*=\s*\[([^\]]*)\]", attribute)
            if match:
                fields[key] = [int(value.strip()) for value in match.group(1).split(",")
                               if value.strip()]
    return fields


def inspect_ir(path: Path) -> dict:
    try:
        from mlir.ir import Context, Module, RankedTensorType, IntegerAttr
        from mlir.dialects import top, tpu, func  # Register the repository dialects.
    except ImportError as error:
        raise RuntimeError("MLIR bindings are required; use the existing TPU-MLIR "
                           "build environment") from error

    groups = []
    modules = []
    operations = []
    counts = Counter()
    with Context() as context:
        context.allow_unregistered_dialects = True
        module = Module.parse(path.read_text(encoding="utf-8"))

        def visit(operation, group_id=None):
            operation = getattr(operation, "operation", operation)
            name = operation.name
            attributes = {entry.name: str(entry.attr) for entry in operation.attributes}
            if name == "builtin.module":
                modules.append(attributes)
            if name == "tpu.Group":
                group_id = len(groups)
                groups.append({"id": group_id, "location": str(operation.location),
                               "attributes": attributes, "lmem_address_high_water_bytes": 0,
                               "lmem_live_peak_bytes": None})
            if name.startswith(("top.", "tpu.")):
                counts[name] += 1
                record = {"index": len(operations), "op": name,
                          "location": str(operation.location), "group_id": group_id,
                          "result_types": [str(value.type) for value in operation.results],
                          "gmem_addresses": [], "attributes": attributes}
                for value in operation.results:
                    if RankedTensorType.isinstance(value.type):
                        encoding = RankedTensorType(value.type).encoding
                        if encoding is not None and IntegerAttr.isinstance(encoding):
                            record["gmem_addresses"].append(IntegerAttr(encoding).value)
                if "ginfo" in attributes:
                    info = layer_group_info(attributes["ginfo"])
                    record["layer_group"] = info
                    if group_id is not None:
                        high_water = max(info.get("out_addr", 0) + info.get("out_size", 0),
                                         info.get("buffer_addr", 0) + info.get("buffer_size", 0))
                        groups[group_id]["lmem_address_high_water_bytes"] = max(
                            groups[group_id]["lmem_address_high_water_bytes"], high_water)
                operations.append(record)
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        visit(child, group_id)

        visit(module.operation)
    return {"source": str(path.resolve()), "module_attributes": modules,
            "op_counts": dict(sorted(counts.items())), "groups": groups, "operations": operations,
            "memory_contract": "LMEM addresses/sizes use the compiler allocator's per-lane units. "
            "Address high-water includes holes. Live peak requires allocator lifetime data and "
            "is not inferred from a sum of tensor sizes. GMEM values are raw encoded addresses."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", type=Path, nargs="+", required=True,
                        help="Top, Tpu and/or final MLIR files; weights are not loaded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [inspect_ir(path.resolve()) for path in args.mlir]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    for report in reports:
        print(f"{report['source']}: {len(report['operations'])} ops, "
              f"{len(report['groups'])} groups")


if __name__ == "__main__":
    main()
