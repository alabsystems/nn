#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# SPDX-License-Identifier: Apache-2.0

"""Export Kokoro TTS model segments for nn::convert().

Exports each of the 5 Kokoro pipeline segments as separate
graph.json + weights.safetensors pairs, ready for import via
`nn_import::import_model()`.

Segments:
  0: PlBert + bert_encoder  -- ALBERT encoder -> Linear projection
  1: TextEncoder             -- Embedding + Conv + BiLSTM + projection
  2: ProsodyPredictor        -- DurationEncoder + duration BiLSTM
  3: F0EnergyPredictor       -- shared BiLSTM + F0/N heads
  4: FullDecoder+Generator   -- (already exported, skip by default)

Non-exportable bridges (length_regulate, harmonic_source, iSTFT) are
kept as NativeOps in the compiled pipeline.

Usage:
  # Export all segments:
  python export_kokoro_segments.py --output-dir ./models/kokoro-82m

  # Export specific segment:
  python export_kokoro_segments.py --output-dir ./models/kokoro-82m --segment plbert

  # With reference activations for L3 parity:
  python export_kokoro_segments.py --output-dir ./models/kokoro-82m --reference

Requirements:
  pip install kokoro torch safetensors

Part of nn::convert Kokoro integration. Design: designs/2026-03-29-kokoro-segment-export.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.export
    import torch.nn as nn
except ImportError:
    print(
        "ERROR: PyTorch is required. Install: pip install torch",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from safetensors.torch import save_file
except ImportError:
    print(
        "ERROR: safetensors is required. Install: pip install safetensors",
        file=sys.stderr,
    )
    sys.exit(1)

# Import the generic nn_export utilities for graph JSON serialization.
from nn_export import export_graph_json, export_weights

# ---------------------------------------------------------------------------
# Segment wrapper modules
# ---------------------------------------------------------------------------


class PlBertSegment(nn.Module):
    """Segment 0: PlBert (ALBERT) + bert_encoder Linear.

    Input: input_ids [B, T] (long).
    Output: bert_features [B, d_en, T].

    Position and token-type embeddings are computed internally from
    input_ids shape (matching the Rust compiled_kokoro_trace_fns.rs
    pattern where these are pre-computed outside the trace scope).
    """

    def __init__(self, model):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(input_ids, dtype=torch.long)
        bert_output = self.bert(input_ids, attention_mask=mask)
        if isinstance(bert_output, (tuple, list)):
            bert_output = bert_output[0]
        encoded = self.bert_encoder(bert_output)
        return encoded.transpose(1, 2)


class TextEncoderSegment(nn.Module):
    """Segment 1: TextEncoder (Embedding + 3xConv + BiLSTM + Linear).

    Input: input_ids [B, T] (long).
    Output: text_features [B, d_en, T].
    """

    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_lengths = torch.tensor(
            [input_ids.shape[-1]], device=input_ids.device, dtype=torch.long,
        )
        text_mask = torch.zeros(
            input_ids.shape, dtype=torch.bool, device=input_ids.device,
        )
        return self.text_encoder(input_ids, input_lengths, text_mask)


class ProsodySegment(nn.Module):
    """Segment 2: ProsodyPredictor (DurationEncoder + BiLSTM + projection).

    Inputs: bert_features [B, d_en, T], style [B, style_dim].
    Outputs: (dur_logits [B, T, max_dur], features [B, d_en+style_dim, T]).
    """

    def __init__(self, model):
        super().__init__()
        self.predictor = model.predictor

    def forward(
        self,
        bert_features: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = bert_features.shape[-1]
        device = bert_features.device
        input_lengths = torch.tensor([seq_len], device=device, dtype=torch.long)
        text_mask = torch.zeros(
            (bert_features.shape[0], seq_len), dtype=torch.bool, device=device,
        )
        d = self.predictor.text_encoder(
            bert_features, style, input_lengths, text_mask,
        )
        x = self.predictor.lstm(d)
        if isinstance(x, tuple):
            x = x[0]
        dur_logits = self.predictor.duration_proj(x)
        features = d.transpose(-1, -2)  # [B, d_en+style_dim, T]
        return dur_logits, features


class F0EnergySegment(nn.Module):
    """Segment 3: F0EnergyPredictor (shared BiLSTM + F0/N heads).

    Inputs: aligned_dur [B, d_en+style_dim, T_mel], style [B, style_dim].
    Outputs: (f0 [B, 1, 2*T_mel], energy [B, 1, 2*T_mel]).
    """

    def __init__(self, model):
        super().__init__()
        self.predictor = model.predictor

    def forward(
        self,
        aligned_dur: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predictor.F0Ntrain(aligned_dur, style)


class FullDecoderSegment(nn.Module):
    """Segment 4: FullDecoder + Generator.

    Inputs: x [B, d_en, T], style [B, style_dim], har_source [B, 2*n_bins, T_stft].
    Outputs: (magnitude [B, n_bins, T_out], phase [B, n_bins, T_out]).

    Note: This segment is already exported. Included for completeness.
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        har_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(x, style, har_source)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

SEGMENTS = {
    "plbert": {
        "class": PlBertSegment,
        "dir_name": "plbert",
        "description": "PlBert + bert_encoder (ALBERT -> Linear projection)",
    },
    "text_encoder": {
        "class": TextEncoderSegment,
        "dir_name": "text_encoder",
        "description": "TextEncoder (Embedding + Conv + BiLSTM + Linear)",
    },
    "prosody": {
        "class": ProsodySegment,
        "dir_name": "prosody",
        "description": "ProsodyPredictor (DurationEncoder + BiLSTM + projection)",
    },
    "f0_energy": {
        "class": F0EnergySegment,
        "dir_name": "f0_energy",
        "description": "F0EnergyPredictor (shared BiLSTM + F0/N heads)",
    },
    "decoder": {
        "class": FullDecoderSegment,
        "dir_name": "decoder",
        "description": "FullDecoder + Generator (already exported)",
    },
}


def _make_example_inputs(
    segment_name: str,
    seq_len: int = 32,
    t_mel: int = 64,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """Create example input tensors for a given segment.

    These must match the forward() signature of the corresponding wrapper
    module. Shapes use small representative values for export; the compiled
    model is shape-polymorphic along time dimensions.
    """
    B = 1
    d_en = 512
    style_dim = 128
    n_bins = 11  # n_fft/2 + 1 = 20/2 + 1

    if segment_name == "plbert":
        return (torch.randint(0, 178, (B, seq_len), device=device),)
    if segment_name == "text_encoder":
        return (torch.randint(0, 178, (B, seq_len), device=device),)
    if segment_name == "prosody":
        return (
            torch.randn(B, d_en, seq_len, device=device),
            torch.randn(B, style_dim, device=device),
        )
    if segment_name == "f0_energy":
        return (
            torch.randn(B, d_en + style_dim, t_mel, device=device),
            torch.randn(B, style_dim, device=device),
        )
    if segment_name == "decoder":
        t_stft = t_mel * 2  # 2x from F0 upsample
        return (
            torch.randn(B, d_en, t_mel, device=device),
            torch.randn(B, style_dim, device=device),
            torch.randn(B, 2 * n_bins, t_stft, device=device),
        )
    raise ValueError(f"Unknown segment: {segment_name}")


def export_segment(
    model,
    segment_name: str,
    output_dir: Path,
    *,
    capture_reference: bool = False,
) -> bool:
    """Export a single Kokoro segment.

    Returns True on success, False on failure (with error logged).
    """
    info = SEGMENTS[segment_name]
    seg_dir = output_dir / info["dir_name"]
    seg_dir.mkdir(parents=True, exist_ok=True)

    log.info("Exporting segment: %s (%s)", segment_name, info["description"])

    # Wrap the model in the segment-specific module.
    try:
        wrapper = info["class"](model)
        wrapper.eval()
    except (AttributeError, TypeError) as exc:
        log.error("  Failed to create wrapper for %s: %s", segment_name, exc)
        return False

    # Create example inputs.
    example_inputs = _make_example_inputs(segment_name)

    # Verify forward pass works before export.
    try:
        with torch.no_grad():
            ref_output = wrapper(*example_inputs)
        if isinstance(ref_output, tuple):
            shapes = [tuple(t.shape) for t in ref_output]
            log.info("  Reference output shapes: %s", shapes)
        else:
            log.info("  Reference output shape: %s", tuple(ref_output.shape))
    except Exception as exc:
        log.error("  Forward pass failed for %s: %s", segment_name, exc)
        log.error("  This segment may need a different wrapper implementation.")
        return False

    # Export via torch.export.
    try:
        log.info("  Running torch.export.export()...")
        ep = torch.export.export(wrapper, example_inputs)
    except Exception as exc:
        log.error("  torch.export.export() failed for %s: %s", segment_name, exc)
        log.error(
            "  The segment may contain ops unsupported by torch.export. "
            "Consider using torch.export with decomposition tables or "
            "wrapping custom ops."
        )
        return False

    # Write graph.json (nn-import schema v8).
    graph_json = export_graph_json(ep)
    graph_path = seg_dir / "graph.json"
    with open(graph_path, "w") as f:
        json.dump(graph_json, f, indent=2)
    node_count = len(graph_json["graph_module"]["graph"]["nodes"])
    log.info("  Wrote %s (%d nodes)", graph_path, node_count)

    # Write weights.safetensors.
    weights_path = seg_dir / "weights.safetensors"
    export_weights(ep, weights_path)
    log.info("  Wrote %s", weights_path)

    # Write reference activations if requested.
    if capture_reference:
        ref_path = seg_dir / "reference.safetensors"
        ref_tensors = {}
        for i, inp in enumerate(example_inputs):
            if isinstance(inp, torch.Tensor):
                ref_tensors[f"input_{i}"] = inp.detach().cpu().contiguous()
        if isinstance(ref_output, tuple):
            for i, out in enumerate(ref_output):
                if isinstance(out, torch.Tensor):
                    ref_tensors[f"output_{i}"] = (
                        out.detach().cpu().contiguous()
                    )
        elif isinstance(ref_output, torch.Tensor):
            ref_tensors["output_0"] = ref_output.detach().cpu().contiguous()
        save_file(ref_tensors, str(ref_path))
        log.info("  Wrote %s", ref_path)

    # Write segment metadata.
    meta_path = seg_dir / "segment_meta.json"
    meta = {
        "segment": segment_name,
        "description": info["description"],
        "node_count": node_count,
        "input_count": len(example_inputs),
        "output_count": (
            len(ref_output) if isinstance(ref_output, tuple) else 1
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("  Export complete for %s.", segment_name)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Export Kokoro TTS segments for nn::convert()",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Base output directory (segments written to subdirectories)",
    )
    parser.add_argument(
        "--segment",
        choices=list(SEGMENTS.keys()) + ["all"],
        default="all",
        help="Which segment to export (default: all)",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Also save reference activations for L3 parity testing",
    )
    parser.add_argument(
        "--skip-decoder",
        action="store_true",
        help="Skip the decoder segment (already exported)",
    )
    args = parser.parse_args()

    # Load the Kokoro model.
    try:
        from kokoro import KPipeline
    except ImportError:
        print(
            "ERROR: kokoro package not installed.\n"
            "Install: pip install kokoro\n"
            "The kokoro package provides the PyTorch model weights.",
            file=sys.stderr,
        )
        sys.exit(1)

    log.info("Loading Kokoro pipeline...")
    pipeline = KPipeline(lang_code="a")
    model = pipeline.model
    model.eval()
    log.info("Model loaded.")

    # Determine which segments to export.
    if args.segment == "all":
        segments_to_export = list(SEGMENTS.keys())
        if args.skip_decoder:
            segments_to_export.remove("decoder")
    else:
        segments_to_export = [args.segment]

    # Export each segment.
    results = {}
    for seg_name in segments_to_export:
        ok = export_segment(
            model,
            seg_name,
            args.output_dir,
            capture_reference=args.reference,
        )
        results[seg_name] = "OK" if ok else "FAILED"

    # Summary.
    log.info("")
    log.info("=== Export Summary ===")
    for seg_name, status in results.items():
        log.info("  %-20s %s", seg_name, status)

    failed = sum(1 for s in results.values() if s == "FAILED")
    if failed > 0:
        log.info("")
        log.info(
            "%d segment(s) failed. See errors above. Segments with BiLSTM "
            "may need torch.export decomposition tables or custom wrappers.",
            failed,
        )
        sys.exit(1)
    else:
        log.info("")
        log.info("All segments exported successfully.")


if __name__ == "__main__":
    main()
