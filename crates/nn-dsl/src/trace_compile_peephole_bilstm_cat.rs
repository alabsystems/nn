// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pass 11: fuse
//! forward LstmSequence + reverse LstmSequence + Cat → BiLstmCat.
//!
//! Detects the graph pattern:
//!   Step fwd: NativeOp { LstmSequence { reverse: false, hidden_size: H } }
//!   Step rev: NativeOp { LstmSequence { reverse: true,  hidden_size: H } }
//!   Step cat: Dispatch { kernel: "cat" } consuming fwd and rev outputs
//!
//! Both LSTMs must share the same graph input and have matching hidden_size.
//! The Cat must concatenate along the last dimension (hidden dim).
//!
//! Replaces all three steps with:
//! - fwd → IdentityPassthrough
//! - rev → IdentityPassthrough
//! - cat → NativeOp { BiLstmCat { ... } } with merged weights (fwd_/rev_ prefixed)
//!
//! Saves 1-2 Metal dispatches per BiLSTM layer (Cat dispatch eliminated,
//! potential buffer reuse). In Kokoro there are 8 BiLSTM instances
//! (text=2, prosody=4, f0=2).
//!
//! Must run AFTER pass 10 (flip_lstm absorption) so that reverse LSTMs
//! already have `reverse: true`.
//!
//! Part of #4252.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, NativeOpKind};

/// Information extracted from a forward or reverse LstmSequence NativeOp.
struct LstmInfo {
    hidden_size: usize,
    input_shape: Vec<usize>,
    h_shape: Vec<usize>,
    reverse: bool,
    weight_data: HashMap<String, WeightRef>,
}

/// Fuse forward LSTM + reverse LSTM + Cat into BiLstmCat NativeOps.
///
/// Scans all `Dispatch{cat}` steps and checks whether both inputs are
/// LstmSequence NativeOps (one forward, one reverse) with matching
/// hidden_size. If so, merges their weights with `fwd_`/`rev_` prefixes
/// and replaces the cat with a BiLstmCat NativeOp.
pub(super) fn fuse_bilstm_cat(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    graph: &ComputationGraph,
) {
    let graph_nodes = graph.nodes();
    let id_to_idx: HashMap<u64, usize> = graph_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id(), i))
        .collect();

    // Scan for cat dispatches that consume exactly 2 LSTM outputs.
    for cat_idx in 0..steps.len() {
        // Must be a Dispatch with kernel name "cat".
        let is_cat = matches!(
            &steps[cat_idx],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "cat"
        );
        if !is_cat {
            continue;
        }

        // Get the cat node's graph inputs.
        let cat_node = match graph_nodes.get(cat_idx) {
            Some(n) => n,
            None => continue,
        };
        let cat_inputs = cat_node.inputs();
        if cat_inputs.len() != 2 {
            continue;
        }

        // Resolve step indices for both inputs.
        let idx_a = match id_to_idx.get(&cat_inputs[0]) {
            Some(&i) => i,
            None => continue,
        };
        let idx_b = match id_to_idx.get(&cat_inputs[1]) {
            Some(&i) => i,
            None => continue,
        };

        // Both inputs must be single-use (consumed only by this cat).
        if use_counts.get(idx_a).copied().unwrap_or(0) != 1 {
            continue;
        }
        if use_counts.get(idx_b).copied().unwrap_or(0) != 1 {
            continue;
        }

        // Extract LSTM info from both steps.
        let info_a = extract_lstm_info(&steps[idx_a]);
        let info_b = extract_lstm_info(&steps[idx_b]);
        let (info_a, info_b) = match (info_a, info_b) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };

        // One must be forward, one must be reverse.
        let (fwd_info, rev_info, fwd_idx, rev_idx) = if !info_a.reverse && info_b.reverse {
            (info_a, info_b, idx_a, idx_b)
        } else if info_a.reverse && !info_b.reverse {
            (info_b, info_a, idx_b, idx_a)
        } else {
            continue;
        };

        // Hidden sizes must match.
        if fwd_info.hidden_size != rev_info.hidden_size {
            continue;
        }

        // Input shapes must match (same input tensor for both directions).
        if fwd_info.input_shape != rev_info.input_shape {
            continue;
        }

        // Merge weight data with fwd_/rev_ prefixes.
        let mut merged_weights = HashMap::new();
        for (key, val) in &fwd_info.weight_data {
            merged_weights.insert(format!("fwd_{key}"), val.clone());
        }
        for (key, val) in &rev_info.weight_data {
            merged_weights.insert(format!("rev_{key}"), val.clone());
        }

        let hidden_size = fwd_info.hidden_size;
        let input_shape = fwd_info.input_shape.clone();
        let h_shape = fwd_info.h_shape.clone();

        // Replace cat step with BiLstmCat NativeOp.
        steps[cat_idx] = CompiledStep::NativeOp {
            op: NativeOpKind::BiLstmCat {
                hidden_size,
                input_shape,
                h_shape,
                fwd_lstm_step: fwd_idx,
                rev_lstm_step: rev_idx,
            },
            weight_data: merged_weights,
        };

        // Replace both LSTM steps with IdentityPassthrough.
        steps[fwd_idx] = CompiledStep::IdentityPassthrough;
        steps[rev_idx] = CompiledStep::IdentityPassthrough;
    }
}

/// Extract LSTM info from a NativeOp step, or None if not an LSTM.
fn extract_lstm_info(step: &CompiledStep) -> Option<LstmInfo> {
    match step {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::LstmSequence {
                    hidden_size,
                    input_shape,
                    h_shape,
                    reverse,
                },
            weight_data,
        } => Some(LstmInfo {
            hidden_size: *hidden_size,
            input_shape: input_shape.clone(),
            h_shape: h_shape.clone(),
            reverse: *reverse,
            weight_data: weight_data.clone(),
        }),
        _ => None,
    }
}

#[cfg(test)]
#[path = "trace_compile_peephole_bilstm_cat_tests.rs"]
mod tests;
