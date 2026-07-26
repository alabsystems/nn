// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Construction helpers for `CompiledModel`.
//!
//! Extracted from `compiled_model.rs` to keep files under 450 lines.
//! Contains graph topology analysis (`build_edge_map`), GPU weight upload
//! (`upload_weights`), and kernel input name extraction (`def_input_names`).
//! GEMM eligibility detection is in the `gemm` submodule.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::{Result, TensorError};
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::{CompiledStep, NativeOpKind};
use nn_dsl::TensorOpKind;

use crate::buffer::MetalBuffer;
use crate::metal_backend::checked_dim_product;

use super::CompiledModelError;

#[path = "compiled_model_build_gemm.rs"]
mod gemm;
pub(super) use gemm::extract_mixed_gemm_infos;

/// Build edge_map: for each step, the step indices that produce its inputs.
///
/// Delegates to [`nn_dsl::compute_edge_map`] for the shared logic
/// (base edges, external_node_ids, NormActivConv1d, AdaIN,
/// FusedResBlock/BatchedStyleProjection input_steps patches).
/// Part of #3261, #3299.
pub(super) fn build_edge_map(
    graph: &ComputationGraph,
    steps: &[CompiledStep],
) -> Result<Vec<Vec<usize>>> {
    Ok(nn_dsl::compute_edge_map(graph, steps))
}

/// Validate buffer planner `last_use` covers direct-access NativeOp deps.
/// FusedResBlock/BatchedStyleProjection bypass `resolve_input_slice`. #3117.
pub(super) fn validate_buffer_plan_edges(steps: &[CompiledStep], last_use: &[usize]) -> Result<()> {
    for (step_idx, step) in steps.iter().enumerate() {
        let deps: Vec<usize> = match step {
            CompiledStep::NativeOp {
                op:
                    NativeOpKind::FusedResBlock {
                        input_steps,
                        shortcut_step,
                        pool_step,
                        ..
                    },
                ..
            } => {
                let mut d = input_steps.clone();
                if let Some(sc) = shortcut_step {
                    d.push(*sc);
                }
                if let Some(ps) = pool_step {
                    d.push(*ps);
                }
                d
            }
            CompiledStep::NativeOp {
                op: NativeOpKind::BatchedStyleProjection { style_step, .. },
                ..
            } => vec![*style_step],
            CompiledStep::NativeOp {
                op: NativeOpKind::ProjectionSlice { source_step, .. },
                ..
            } => vec![*source_step],
            // FusedResBlockChain: same direct buffer access as FusedResBlock.
            // Part of #4264.
            CompiledStep::NativeOp {
                op:
                    NativeOpKind::FusedResBlockChain {
                        input_steps,
                        first_shortcut_step,
                        ..
                    },
                ..
            } => {
                let mut d = input_steps.clone();
                if let Some(sc) = first_shortcut_step {
                    d.push(*sc);
                }
                d
            }
            _ => continue,
        };
        for dep in deps {
            if dep >= last_use.len() {
                return Err(CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: format!(
                        "NativeOp references step {dep} but last_use has only {} entries",
                        last_use.len(),
                    ),
                }
                .into());
            }
            if last_use[dep] < step_idx {
                return Err(CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: format!(
                        "buffer planner releases step {dep} at step {} but NativeOp at step {step_idx} \
                         reads it directly — edge_map builders are out of sync (see #3117)",
                        last_use[dep],
                    ),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// Returns `true` when a step's uploaded weight buffers are safe to share
/// across different compiled shape variants.
///
/// Traced `ConstantWeight` nodes are excluded because they can capture
/// shape-dependent helper tensors (for example interpolation indices in
/// Kokoro's SineGen trace). Sharing them by `(step_idx, weight_name)` alone
/// aliases incompatible buffer sizes across shape variants (#3507).
pub(crate) fn shares_weight_buffers(step: &CompiledStep) -> bool {
    !matches!(
        step,
        CompiledStep::NativeOp {
            op: NativeOpKind::ConstantWeight { .. },
            ..
        }
    )
}

fn encoded_weight_len_bytes(numel: usize, dtype: ScalarType) -> Option<usize> {
    let elem_bytes = match dtype {
        ScalarType::F32 => size_of::<f32>(),
        ScalarType::F16 | ScalarType::BF16 => size_of::<u16>(),
        _ => return None,
    };
    numel.checked_mul(elem_bytes)
}

/// Upload all weight data from compiled steps to GPU buffers.
///
/// Uses `step_scalar_types` to create buffers in the correct dtype.
/// WeightRef stores f32 data regardless of original model dtype; for
/// F16/BF16 steps the f32 values are converted to f16 before upload
/// (Metal has no native bf16 — both map to `half`). See #2273.
///
/// When `shared` is `Some`, aliased GPU buffers are reused for invariant model
/// weights that match by `(step_idx, weight_name)` key — zero-copy via ARC.
/// Traced `ConstantWeight` buffers are always uploaded fresh because they may
/// be shape-dependent (#3507). Shared buffers are only reused when the
/// expected encoded byte length also matches, so compiler-generated
/// shape-sensitive weights that accidentally reuse a `(step_idx, name)` key
/// still fall back to a fresh upload instead of aliasing a stale smaller
/// buffer. Weights not found in the shared store are uploaded fresh (handles
/// fusion-order changes between shape variants). Part of #2630.
pub(super) fn upload_weights(
    steps: &[CompiledStep],
    step_scalar_types: &[ScalarType],
    ctx: &crate::context::MetalContext,
    shared: Option<&HashMap<(usize, String), MetalBuffer>>,
) -> Result<HashMap<(usize, String), MetalBuffer>> {
    let mut buffers = HashMap::new();

    for (step_idx, step) in steps.iter().enumerate() {
        let step_weights = match step {
            CompiledStep::Dispatch { weight_data, .. }
            | CompiledStep::NativeOp { weight_data, .. } => Some(weight_data),
            _ => None,
        };
        if let Some(weight_data) = step_weights {
            let shareable = shares_weight_buffers(step);
            let dtype = step_scalar_types.get(step_idx).copied().ok_or_else(|| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: "step_scalar_types index out of bounds".into(),
                })
            })?;
            for (name, weight_ref) in weight_data {
                let key = (step_idx, name.clone());
                let numel: usize = checked_dim_product(weight_ref.shape())?;
                let expected_len = encoded_weight_len_bytes(numel, dtype).ok_or_else(|| {
                    TensorError::from(CompiledModelError::WeightUploadFailed {
                        step_idx,
                        name: name.clone(),
                        reason: format!(
                            "encoded weight byte size overflow for shape {:?}",
                            weight_ref.shape()
                        ),
                    })
                })?;

                // Fast path: alias from shared store (zero GPU upload).
                if shareable {
                    if let Some(existing) = shared.and_then(|s| s.get(&key)) {
                        if existing.len() == expected_len {
                            buffers.insert(key, existing.alias());
                            continue;
                        }
                    }
                }

                // Slow path: upload fresh.
                if weight_ref.data().is_empty() {
                    if numel == 0 {
                        continue; // Zero-element shape (e.g., [0]) -- nothing to upload.
                    }
                    // Non-zero shape with empty data: weight extraction failed
                    // during tracing (e.g., unsupported dtype). Error instead of
                    // silently zero-filling, which produces wrong model output.
                    // See #2190.
                    return Err(TensorError::from(CompiledModelError::WeightUploadFailed {
                        step_idx,
                        name: name.clone(),
                        reason: format!(
                            "weight has shape {:?} ({numel} elements) but empty data — \
                             weight extraction failed during tracing (unsupported dtype?)",
                            weight_ref.shape()
                        ),
                    }));
                }
                // Validate data length matches shape product. Mismatched
                // lengths would silently upload garbage to GPU. See #2342.
                if weight_ref.data().len() != numel {
                    return Err(TensorError::from(CompiledModelError::WeightUploadFailed {
                        step_idx,
                        name: name.clone(),
                        reason: format!(
                            "weight has shape {:?} ({numel} elements) but data has {} values",
                            weight_ref.shape(),
                            weight_ref.data().len(),
                        ),
                    }));
                }
                let buf = upload_buffer_typed(ctx, weight_ref.data(), dtype).map_err(|e| {
                    TensorError::from(CompiledModelError::WeightUploadFailed {
                        step_idx,
                        name: name.clone(),
                        reason: e.to_string(),
                    })
                })?;
                buffers.insert(key, buf);
            }
        }
    }

    Ok(buffers)
}

/// Pre-compute combined LSTM biases at build time.
///
/// LSTM steps with separate `bias_ih` and `bias_hh` weight entries would
/// otherwise dispatch a GPU elementwise add on every forward pass to combine
/// them. Since biases are immutable weights, this computes the sum once on
/// CPU during model construction and uploads the combined buffer as `"bias"`.
/// The executor's `has_single` check picks up the pre-computed entry.
///
/// Saves 1 GPU dispatch per LSTM step per forward pass (2 total in Kokoro).
pub(super) fn precompute_lstm_combined_biases(
    steps: &[CompiledStep],
    step_scalar_types: &[ScalarType],
    flat: &mut HashMap<(usize, String), MetalBuffer>,
    ctx: &crate::context::MetalContext,
) -> Result<()> {
    for (step_idx, step) in steps.iter().enumerate() {
        let weight_data = match step {
            CompiledStep::NativeOp {
                op: NativeOpKind::LstmSequence { .. },
                weight_data,
                ..
            } => weight_data,
            _ => continue,
        };
        let has_bih = weight_data.contains_key("bias_ih");
        let has_bhh = weight_data.contains_key("bias_hh");
        let has_combined =
            weight_data.contains_key("bias") || flat.contains_key(&(step_idx, "bias".to_string()));
        if !has_bih || !has_bhh || has_combined {
            continue;
        }
        let bih = &weight_data["bias_ih"];
        let bhh = &weight_data["bias_hh"];
        if bih.data().len() != bhh.data().len() || bih.data().is_empty() {
            continue;
        }
        let combined: Vec<f32> = bih
            .data()
            .iter()
            .zip(bhh.data().iter())
            .map(|(a, b)| a + b)
            .collect();
        let dtype = step_scalar_types
            .get(step_idx)
            .copied()
            .unwrap_or(ScalarType::F32);
        let buf = upload_buffer_typed(ctx, &combined, dtype).map_err(|e| {
            TensorError::from(CompiledModelError::WeightUploadFailed {
                step_idx,
                name: "bias".to_string(),
                reason: format!("LSTM combined bias upload: {e}"),
            })
        })?;
        flat.insert((step_idx, "bias".to_string()), buf);
        flat.remove(&(step_idx, "bias_ih".to_string()));
        flat.remove(&(step_idx, "bias_hh".to_string()));
    }
    Ok(())
}

/// Pre-upload all ConstantValue step data to GPU buffers.
///
/// Called once at construction time. Each forward pass reuses these buffers
/// via alias instead of creating fresh CPU→GPU allocations. See #2338.
///
/// Uses `step_scalar_types` to create buffers in the correct dtype. For
/// ConstantValue nodes, the dtype comes from the next consumer's step type
/// (constants are typically consumed by a Dispatch step). When the consumer
/// expects F16/BF16, the constant is uploaded as f16. See #2273.
pub(super) fn upload_constants(
    steps: &[CompiledStep],
    step_scalar_types: &[ScalarType],
    ctx: &crate::context::MetalContext,
) -> Result<HashMap<usize, MetalBuffer>> {
    let mut buffers = HashMap::new();
    for (step_idx, step) in steps.iter().enumerate() {
        if let CompiledStep::ConstantValue { value, shape } = step {
            let numel = checked_dim_product(shape)?;
            let fill_val = *value as f32;
            if !fill_val.is_finite() {
                return Err(TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: format!(
                        "constant value {value} is not finite as f32 (becomes {fill_val})"
                    ),
                }));
            }
            let dtype = step_scalar_types.get(step_idx).copied().ok_or_else(|| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: "step_scalar_types index out of bounds".into(),
                })
            })?;
            let data = vec![fill_val; numel];
            let buf = upload_buffer_typed(ctx, &data, dtype).map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: format!("constant buffer creation failed: {e}"),
                })
            })?;
            buffers.insert(step_idx, buf);
        }
    }
    Ok(buffers)
}

/// Create a GPU buffer from f32 data, converting to f16 when the step
/// dtype is F16 or BF16 (Metal has no native bf16 compute).
fn upload_buffer_typed(
    ctx: &crate::context::MetalContext,
    data: &[f32],
    dtype: ScalarType,
) -> std::result::Result<MetalBuffer, crate::error::MetalError> {
    match dtype {
        ScalarType::F32 => ctx.create_buffer(data),
        ScalarType::F16 | ScalarType::BF16 => {
            // Convert f32 → f16 bits → u16 for Metal `half` storage.
            let encoded: Vec<u16> = data
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            ctx.create_buffer(&encoded)
        }
        _ => ctx.create_buffer(data),
    }
}

/// Convert flat `HashMap<(step_idx, name), buf>` to step-indexed
/// `Vec<HashMap<name, buf>>` for zero-alloc hot-path lookup.
///
/// Construction-time only. Each step gets its own sub-map.
pub(super) fn flat_weights_to_indexed(
    flat: HashMap<(usize, String), MetalBuffer>,
    num_steps: usize,
) -> Vec<HashMap<String, MetalBuffer>> {
    let mut indexed: Vec<HashMap<String, MetalBuffer>> =
        (0..num_steps).map(|_| HashMap::new()).collect();
    for ((step_idx, name), buf) in flat {
        if step_idx < num_steps {
            indexed[step_idx].insert(name, buf);
        }
    }
    indexed
}

/// Extract input names from a `TensorKernelDef` by iterating its nodes
/// and collecting `TensorOpKind::Input { name, .. }` entries.
pub(super) fn def_input_names(def: &nn_dsl::TensorKernelDef) -> Vec<String> {
    def.nodes
        .iter()
        .filter_map(|node| match &node.kind {
            TensorOpKind::Input { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
#[path = "compiled_model_build_tests.rs"]
mod tests;
