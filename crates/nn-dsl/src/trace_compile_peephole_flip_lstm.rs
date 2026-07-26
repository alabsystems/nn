// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: Flip + LstmSequence + Flip → LstmSequence(reverse=true).
//!
//! Absorbs the two `flip(dim=0)` dispatches surrounding a backward-direction
//! LSTM into the kernel's built-in reverse mode. Saves 2 Metal dispatches per
//! backward LSTM layer — ~192 dispatches total in Kokoro BiLSTM segments.
//! Part of #1815.

use crate::tensor_ir::TensorOpKind;

use super::super::{CompiledStep, NativeOpKind};

/// Scan for `Flip → LstmSequence → Flip` triples and absorb both flips.
///
/// Matches:
/// - `steps[i]` is `Dispatch` with `kernel.name() == "flip"` and IndexSelect dim 0
/// - `steps[i+1]` is `NativeOp{LstmSequence { reverse: false, .. }}`
/// - `steps[i+2]` is `Dispatch` with `kernel.name() == "flip"` and IndexSelect dim 0
/// - `use_counts[i] == 1` (flip_in output consumed only by LSTM)
/// - `use_counts[i+1] == 1` (LSTM output consumed only by flip_out)
///
/// The reversed LSTM kernel reads input timesteps from `seq_len-1` to 0 and
/// writes output in the same reversed order, eliminating both external flips.
pub(super) fn absorb_flip_lstm(steps: &mut [CompiledStep], use_counts: &[usize]) {
    let len = steps.len();
    if len < 3 {
        return;
    }
    let mut i = 0;
    while i + 2 < len {
        if try_absorb(steps, i, use_counts) {
            i += 3; // skip past the absorbed triple
        } else {
            i += 1;
        }
    }
}

/// Try to absorb flip(i) + LSTM(i+1) + flip(i+2) into LSTM(reverse=true).
fn try_absorb(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    // Step[i] must be a dim-0 flip dispatch.
    if !is_flip_dim0(&steps[i]) {
        return false;
    }

    // Fan-out: flip_in output must have exactly 1 consumer (the LSTM).
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Step[i+1] must be NativeOp{LstmSequence} with reverse=false.
    let is_lstm_forward = matches!(
        &steps[i + 1],
        CompiledStep::NativeOp {
            op: NativeOpKind::LstmSequence { reverse: false, .. },
            ..
        }
    );
    if !is_lstm_forward {
        return false;
    }

    // Fan-out: LSTM output must have exactly 1 consumer (the flip_out).
    if use_counts.get(i + 1).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Step[i+2] must be a dim-0 flip dispatch.
    if !is_flip_dim0(&steps[i + 2]) {
        return false;
    }

    // Absorb: set LSTM to reverse mode, replace both flips with passthrough.
    // IdentityPassthrough resolves its buffer from the edge_map, so:
    // - flip_in passthrough → passes through flip_in's original input
    // - LSTM(reverse=true) reads from flip_in passthrough = original unflipped input
    // - flip_out passthrough → passes through LSTM reverse output
    if let CompiledStep::NativeOp {
        op: NativeOpKind::LstmSequence { reverse, .. },
        ..
    } = &mut steps[i + 1]
    {
        *reverse = true;
    }
    steps[i] = CompiledStep::IdentityPassthrough;
    steps[i + 2] = CompiledStep::IdentityPassthrough;
    true
}

/// Check if a step is a flip dispatch on dimension 0.
///
/// The compile_flip function creates a kernel named "flip" with an IndexSelect
/// IR node. We check both the kernel name and the IndexSelect dim to ensure
/// we only absorb dim-0 flips (sequence dimension for BiLSTM).
fn is_flip_dim0(step: &CompiledStep) -> bool {
    match step {
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "flip" => {
            // Verify the IndexSelect operates on dim 0.
            kernel
                .def()
                .nodes
                .iter()
                .any(|n| matches!(n.kind, TensorOpKind::IndexSelect { dim: 0, .. }))
        }
        _ => false,
    }
}

#[cfg(test)]
#[path = "trace_compile_peephole_flip_lstm_tests.rs"]
mod tests;
