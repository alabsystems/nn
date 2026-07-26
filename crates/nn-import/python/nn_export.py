#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# SPDX-License-Identifier: Apache-2.0

"""Export a PyTorch model for nn::convert().

Produces:
  - graph.json:  torch.export graph in nn-import JSON schema (v8)
  - weights.safetensors:  model parameters
  - reference.safetensors (optional):  intermediate activations for L3 parity

Usage:
  python nn_export.py --model module:Class --output-dir ./exported
  python nn_export.py --model module:Class --checkpoint weights.pt --output-dir ./exported
  python nn_export.py --model module:Class --output-dir ./exported --input-shape 1 3 224 224
  python nn_export.py --model module:Class --output-dir ./exported --reference

Requirements:
  pip install torch safetensors

Part of #3525, #3771, #2306 (nn::convert).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks with clear error messages
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.export
    import torch.fx
except ImportError:
    print(
        "ERROR: PyTorch is required but not installed.\n"
        "Install it with:  pip install torch\n"
        "See https://pytorch.org/get-started/locally/ for platform-specific instructions.",
        file=sys.stderr,
    )
    sys.exit(1)

_SAFETENSORS_AVAILABLE = True
try:
    from safetensors.torch import save_file as _safetensors_save_file  # noqa: F401
except ImportError:
    _SAFETENSORS_AVAILABLE = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ScalarType mapping (torch dtype -> integer matching nn-import parse.rs)
# ---------------------------------------------------------------------------

_DTYPE_TO_SCALAR_TYPE: dict[torch.dtype, int] = {
    torch.uint8: 1,
    torch.int8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bool: 11,
    torch.bfloat16: 13,
}


def _scalar_type(dtype: torch.dtype) -> int:
    """Map a torch dtype to the ScalarType integer used by nn-import."""
    return _DTYPE_TO_SCALAR_TYPE.get(dtype, 7)


# ---------------------------------------------------------------------------
# Argument serialization (produces JSON matching parse.rs Argument enum)
# ---------------------------------------------------------------------------


def _tensor_arg(name: str) -> dict:
    """Serialize a tensor reference: ``{"as_tensor": {"name": "..."}}``."""
    return {"as_tensor": {"name": name}}


def _serialize_arg(arg) -> dict:
    """Serialize a torch.export argument to nn JSON format.

    Each variant is a single-key JSON object matching the ``Argument`` enum in
    ``crates/nn-import/src/parse.rs`` (e.g., ``{"as_int": 42}``).
    """
    if isinstance(arg, torch.fx.Node):
        return _tensor_arg(arg.name)
    # bool MUST be checked before int (bool is a subclass of int in Python)
    if isinstance(arg, bool):
        return {"as_bool": arg}
    if isinstance(arg, int):
        return {"as_int": arg}
    if isinstance(arg, float):
        return {"as_float": arg}
    if arg is None:
        return {"as_none": True}
    if isinstance(arg, str):
        return {"as_string": arg}
    if isinstance(arg, torch.dtype):
        return {"as_scalar_type": _scalar_type(arg)}
    if isinstance(arg, torch.memory_format):
        return {"as_memory_format": int(arg)}
    if isinstance(arg, torch.device):
        return {"as_device": {"type": str(arg.type), "index": arg.index}}
    if isinstance(arg, (list, tuple)):
        if all(isinstance(a, torch.fx.Node) for a in arg):
            return {"as_tensors": [{"name": a.name} for a in arg]}
        # bool before int — bool subclasses int in Python
        if all(isinstance(a, bool) for a in arg):
            return {"as_bools": list(arg)}
        if all(isinstance(a, int) and not isinstance(a, bool) for a in arg):
            return {"as_ints": list(arg)}
        if all(isinstance(a, float) for a in arg):
            return {"as_floats": list(arg)}
        # Fallback: coerce to int list
        return {"as_ints": [int(a) for a in arg]}
    # Last resort: coerce to int
    return {"as_int": int(arg)}


# ---------------------------------------------------------------------------
# TensorMeta extraction (produces JSON matching parse.rs TensorMeta)
# ---------------------------------------------------------------------------


def _tensor_meta(node: torch.fx.Node) -> dict | None:
    """Extract TensorMeta from a node's metadata.

    Matches the ``TensorMeta`` struct in ``crates/nn-import/src/parse.rs``:
    dtype (ScalarType int), sizes (``Vec<SymInt>``), requires_grad, strides.
    """
    meta = node.meta.get("val")
    if meta is None:
        meta = node.meta.get("tensor_meta")
    if meta is None:
        return None
    if isinstance(meta, torch.Tensor):
        return {
            "dtype": _scalar_type(meta.dtype),
            "sizes": [{"as_int": s} for s in meta.shape],
            "requires_grad": meta.requires_grad,
            "strides": [{"as_int": s} for s in meta.stride()],
        }
    if hasattr(meta, "shape") and hasattr(meta, "dtype"):
        stride_fn = getattr(meta, "stride", None)
        strides = list(stride_fn()) if callable(stride_fn) else []
        return {
            "dtype": _scalar_type(meta.dtype),
            "sizes": [{"as_int": s} for s in meta.shape],
            "requires_grad": getattr(meta, "requires_grad", False),
            "strides": [{"as_int": s} for s in strides],
        }
    return None


# ---------------------------------------------------------------------------
# Input classification (parameter / buffer / user_input)
# ---------------------------------------------------------------------------


def _classify_inputs(
    ep: torch.export.ExportedProgram,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Classify graph inputs as parameters, buffers, or user inputs.

    Returns ``(params, buffers, user_inputs)`` where params/buffers map
    graph-level placeholder name to ``nn.Module`` attribute name, and
    user_inputs is a list of placeholder names for runtime inputs.
    """
    params: dict[str, str] = {}
    buffers: dict[str, str] = {}
    user_inputs: list[str] = []
    for spec in ep.graph_signature.input_specs:
        kind = spec.kind.name if hasattr(spec.kind, "name") else str(spec.kind)
        arg_name = spec.arg.name if hasattr(spec.arg, "name") else str(spec.arg)
        if kind in ("PARAMETER", "parameter"):
            params[arg_name] = spec.target or arg_name
        elif kind in ("BUFFER", "buffer"):
            buffers[arg_name] = spec.target or arg_name
        elif kind in ("USER_INPUT", "user_input"):
            user_inputs.append(arg_name)
    return params, buffers, user_inputs


def _input_spec_for(name: str, params: dict, buffers: dict) -> dict:
    """Build an InputSpec for a placeholder node.

    Matches ``InputSpec`` enum in ``crates/nn-import/src/parse_specs.rs``.
    """
    if name in params:
        return {"parameter": {"arg": {"name": name}, "parameter_name": params[name]}}
    if name in buffers:
        return {
            "buffer": {
                "arg": {"name": name},
                "buffer_name": buffers[name],
                "persistent": True,
            },
        }
    return {"user_input": {"arg": _tensor_arg(name)}}


# ---------------------------------------------------------------------------
# Node serialization
# ---------------------------------------------------------------------------


def _serialize_call_function(node: torch.fx.Node) -> dict:
    """Serialize a call_function node to nn JSON.

    Matches the ``Node`` struct in ``crates/nn-import/src/parse.rs``:
    target, inputs (``Vec<NamedArgument>``), outputs, metadata.
    """
    target = str(node.target)
    if target.startswith("aten."):
        target = "torch.ops." + target

    node_inputs: list[dict] = []
    schema = getattr(node.target, "_schema", None)
    for i, arg in enumerate(node.args):
        if schema and i < len(schema.arguments):
            arg_name = schema.arguments[i].name
        else:
            arg_name = f"arg{i}"
        node_inputs.append(
            {"name": arg_name, "arg": _serialize_arg(arg), "kind": 1},
        )
    for kw_name, kw_val in node.kwargs.items():
        node_inputs.append(
            {"name": kw_name, "arg": _serialize_arg(kw_val), "kind": 2},
        )

    return {
        "target": target,
        "inputs": node_inputs,
        "outputs": [_tensor_arg(node.name)],
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Graph JSON export (schema v8, matching crates/nn-import/src/parse.rs)
# ---------------------------------------------------------------------------


def export_graph_json(ep: torch.export.ExportedProgram) -> dict:
    """Convert an ExportedProgram to nn-import JSON schema v8.

    The output matches the ``ExportedProgram`` struct in
    ``crates/nn-import/src/parse.rs`` and is parsed by
    ``parse_exported_program()``.
    """
    params, buffers, _ = _classify_inputs(ep)
    nodes_json: list[dict] = []
    inputs_json: list[dict] = []
    outputs_json: list[dict] = []
    tensor_values: dict[str, dict] = {}
    input_specs: list[dict] = []
    output_specs: list[dict] = []

    for node in ep.graph_module.graph.nodes:
        meta = _tensor_meta(node)
        if meta:
            tensor_values[node.name] = meta

        if node.op == "placeholder":
            inputs_json.append(_tensor_arg(node.name))
            input_specs.append(_input_spec_for(node.name, params, buffers))
        elif node.op == "call_function":
            nodes_json.append(_serialize_call_function(node))
        elif node.op == "output":
            args = (
                node.args[0]
                if isinstance(node.args[0], (list, tuple))
                else [node.args[0]]
            )
            for arg in args:
                if isinstance(arg, torch.fx.Node):
                    outputs_json.append(_tensor_arg(arg.name))
                    output_specs.append(
                        {"user_output": {"arg": _tensor_arg(arg.name)}},
                    )

    return {
        "graph_module": {
            "graph": {
                "inputs": inputs_json,
                "outputs": outputs_json,
                "nodes": nodes_json,
                "tensor_values": tensor_values,
                "is_single_tensor_return": len(outputs_json) == 1,
            },
            "signature": {
                "input_specs": input_specs,
                "output_specs": output_specs,
            },
            "module_call_graph": [],
        },
        "schema_version": {"major": 8, "minor": 15},
        "opset_version": {"aten": 10},
        "range_constraints": {},
    }


# ---------------------------------------------------------------------------
# Weights export
# ---------------------------------------------------------------------------


def _require_safetensors() -> None:
    """Ensure safetensors is importable, with a clear error if not."""
    if not _SAFETENSORS_AVAILABLE:
        print(
            "ERROR: safetensors is required but not installed.\n"
            "Install it with:  pip install safetensors",
            file=sys.stderr,
        )
        sys.exit(1)


def export_weights(ep: torch.export.ExportedProgram, path: Path) -> None:
    """Save model weights as safetensors."""
    _require_safetensors()
    from safetensors.torch import save_file

    state: dict[str, torch.Tensor] = {}
    for name, param in ep.named_parameters():
        state[name] = param.detach().cpu().contiguous()
    for name, buf in ep.named_buffers():
        state[name] = buf.detach().cpu().contiguous()
    save_file(state, str(path))


# ---------------------------------------------------------------------------
# Reference activations export
# ---------------------------------------------------------------------------


def export_reference(
    model: torch.nn.Module,
    example_inputs: tuple,
    path: Path,
    ep: torch.export.ExportedProgram | None = None,
) -> None:
    """Capture reference activations for L3 parity checking.

    Registers forward hooks on every named submodule to capture intermediate
    activations.  When *ep* is provided, input and output tensors are also
    saved under their graph-level names so that ``check_reference_parity``
    in Rust can look them up by the names in the imported graph signature.
    """
    _require_safetensors()
    from safetensors.torch import save_file

    activations: dict[str, torch.Tensor] = {}
    hooks: list = []

    def make_hook(name: str):
        def hook_fn(_module, _input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu().contiguous().clone()
            elif isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        activations[f"{name}.{i}"] = (
                            o.detach().cpu().contiguous().clone()
                        )

        return hook_fn

    for name, mod in model.named_modules():
        if name:
            hooks.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        output = model(*example_inputs)

    if isinstance(output, torch.Tensor):
        activations["output"] = output.detach().cpu().contiguous().clone()

    for h in hooks:
        h.remove()

    # Save inputs/outputs under graph-level names for L3 parity.
    if ep is not None:
        _add_graph_level_refs(activations, ep, example_inputs, output)

    save_file(activations, str(path))


def _add_graph_level_refs(
    activations: dict[str, torch.Tensor],
    ep: torch.export.ExportedProgram,
    example_inputs: tuple,
    output,
) -> None:
    """Save input/output tensors under graph-level names for L3 parity."""
    _, _, user_input_names = _classify_inputs(ep)
    for name, tensor in zip(user_input_names, example_inputs):
        if isinstance(tensor, torch.Tensor):
            activations[name] = tensor.detach().cpu().contiguous().clone()
    # Normalize output to a list so multi-output models are handled correctly.
    outputs = [output] if isinstance(output, torch.Tensor) else list(output)
    for i, spec in enumerate(ep.graph_signature.output_specs):
        arg_name = spec.arg.name if hasattr(spec.arg, "name") else str(spec.arg)
        if i < len(outputs) and isinstance(outputs[i], torch.Tensor):
            activations[arg_name] = outputs[i].detach().cpu().contiguous().clone()


# ---------------------------------------------------------------------------
# Top-level export orchestrator
# ---------------------------------------------------------------------------

DEFAULT_INPUT_SHAPE: list[int] = [1, 3, 224, 224]


def export_model(
    model: torch.nn.Module,
    example_inputs: tuple,
    output_dir: Path,
    *,
    capture_reference: bool = False,
) -> None:
    """Export a PyTorch model for ``nn::convert()``.

    Writes to *output_dir*:
      - ``graph.json``:  torch.export graph in nn-import JSON schema v8
      - ``weights.safetensors``:  model parameters and buffers
      - ``reference.safetensors`` (if *capture_reference*):  intermediate
        activations for L3 parity checking
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    log.info("Running torch.export.export()...")
    ep = torch.export.export(model, example_inputs)

    # -- graph.json --
    graph_json = export_graph_json(ep)
    graph_path = output_dir / "graph.json"
    with open(graph_path, "w") as f:
        json.dump(graph_json, f, indent=2)
    node_count = len(graph_json["graph_module"]["graph"]["nodes"])
    log.info("Wrote %s (%d nodes)", graph_path, node_count)

    # -- weights.safetensors --
    weights_path = output_dir / "weights.safetensors"
    export_weights(ep, weights_path)
    log.info("Wrote %s", weights_path)

    # -- reference.safetensors (optional) --
    if capture_reference:
        ref_path = output_dir / "reference.safetensors"
        export_reference(model, example_inputs, ref_path, ep=ep)
        log.info("Wrote %s", ref_path)


# ---------------------------------------------------------------------------
# CLI entry point (called as subprocess by `nn convert --from-pytorch`)
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Export a PyTorch model for nn::convert()",
        epilog=(
            "Example:\n"
            "  python nn_export.py --model nn_models:NnNet --output-dir ./exported\n"
            "  python nn_export.py --model nn_models:NnNet --checkpoint weights.pt "
            "--output-dir ./exported --reference"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model spec as 'module:Class' (e.g. 'nn_models:NnNet')",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a .pt checkpoint; loaded via model.load_state_dict(torch.load(...))",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write graph.json, weights.safetensors, etc.",
    )
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=DEFAULT_INPUT_SHAPE,
        help="Input tensor shape (default: 1 3 224 224)",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Also capture intermediate activations to reference.safetensors",
    )
    args = parser.parse_args()

    # -- Validate model spec --
    if ":" not in args.model:
        parser.error(
            "Model spec must be 'module:Class' (e.g. 'nn_models:NnNet'). "
            "Got: " + args.model
        )

    module_name, class_name = args.model.rsplit(":", 1)

    # -- Dynamic import of the model class --
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        print(
            f"ERROR: Could not import module '{module_name}': {exc}\n"
            f"Make sure the module is installed or its parent package is on sys.path.",
            file=sys.stderr,
        )
        sys.exit(1)

    cls = getattr(module, class_name, None)
    if cls is None:
        available = [a for a in dir(module) if not a.startswith("_")]
        print(
            f"ERROR: Module '{module_name}' has no attribute '{class_name}'.\n"
            f"Available public names: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    log.info("Instantiating %s.%s ...", module_name, class_name)
    model = cls()

    # -- Load checkpoint if provided --
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            print(
                f"ERROR: Checkpoint file not found: {args.checkpoint}",
                file=sys.stderr,
            )
            sys.exit(1)
        log.info("Loading checkpoint: %s", args.checkpoint)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    # -- Create example input from --input-shape --
    log.info("Input shape: %s", list(args.input_shape))
    example_input = (torch.randn(*args.input_shape),)

    # -- Export --
    export_model(
        model,
        example_input,
        args.output_dir,
        capture_reference=args.reference,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
