// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Autocast op classification for [`CompiledModelBuilder`].
//!
//! Three categories matching PyTorch's autocast:
//! - **Compute:** Conv, Embedding, Attention → always F16
//! - **Accumulate:** Softmax, Norms, Reductions → stays F32
//! - **Passthrough:** Activations, binary elementwise, data-movement → inherit input dtype
//!
//! Extracted from `compiled_model_builder.rs` for 450-line compliance. Part of #2981.

use nn_dsl::trace_compile::CompiledStep;

/// Non-GEMM compute ops that benefit from F16 regardless of threadgroup count.
/// Conv has F32 accumulators, Embedding halves weight bandwidth, Attention has
/// dedicated F16 paths. Linear/MatMul excluded — gated by mixed_gemm_infos. #2981.
pub(super) fn is_non_gemm_compute_dispatch(def: &nn_dsl::TensorKernelDef) -> bool {
    use nn_dsl::tensor_ir::TensorOpKind;
    let output_idx = def.output.index();
    if output_idx >= def.nodes.len() {
        return false;
    }
    matches!(
        &def.nodes[output_idx].kind,
        TensorOpKind::Conv1d { .. }
            | TensorOpKind::Conv2d { .. }
            | TensorOpKind::ConvTranspose1d { .. }
            | TensorOpKind::ConvTranspose2d { .. }
            | TensorOpKind::Embedding { .. }
            | TensorOpKind::Attention { .. }
    )
}

/// Whether a NativeOp is compute-dominant and safe for F16 in autocast mode.
/// FlashAttention uses F32 accumulators with F16 Q/K/V I/O. LinearActivation
/// is NOT here — it gets F16 only when simdgroup-eligible (mixed_gemm_infos
/// override above), because the naive kernel lacks F32 accumulators. #2981.
/// BatchedLinearProjection is classified here (not via extract_mixed_gemm_infos)
/// because its executor uses DynTensor matmul which routes to simd_gemm_f16
/// (F32 accumulators) when dims qualify. #3272, #3281.
/// NormLinear (fused LayerNorm/RmsNorm + Linear) uses F32 accumulators and
/// threadgroup memory for normalization, then F32 dot-product for the GEMM.
/// MSL loads half→float on input, stores float→half on output. #3287.
/// ProjectionSlice inherits dtype via passthrough.
pub(super) fn is_compute_native_op(op: &nn_dsl::NativeOpKind) -> bool {
    matches!(
        op,
        nn_dsl::NativeOpKind::FlashAttention { .. }
            | nn_dsl::NativeOpKind::NormActivConv1d { .. }
            | nn_dsl::NativeOpKind::FusedResBlock { .. }
            | nn_dsl::NativeOpKind::BatchedLinearProjection { .. }
            | nn_dsl::NativeOpKind::NormLinear { .. }
            // Fused AdaIN kernels: InstanceNorm uses F32 accumulators internally,
            // I/O buffers parameterized for half. Same pattern as NormActivConv1d.
            // Eliminates ~20 F16<->F32 cast dispatches in Kokoro Generator. #3766.
            | nn_dsl::NativeOpKind::AdainSnake { .. }
            | nn_dsl::NativeOpKind::AdainLeakyRelu { .. }
            | nn_dsl::NativeOpKind::FusedAdainSnake { .. }
            // Fused InstanceNorm + Mul + Add: InstanceNorm uses F32 accumulators
            // internally, I/O buffers parameterized for half. Part of #4252.
            | nn_dsl::NativeOpKind::FusedInstanceNormMulAdd { .. }
            // Fused Snake + InstanceNorm: InstanceNorm uses F32 accumulators
            // internally, Snake computed in F32. Part of #4264.
            | nn_dsl::NativeOpKind::FusedSnakeInstanceNorm { .. }
            // Fused upsample + conv1d: conv uses F32 accumulators internally.
            // Part of #4310.
            | nn_dsl::NativeOpKind::FusedUpsampleConv1d { .. }
            // Fused LayerNorm + Linear: same as NormLinear. F32 accumulators
            // in both norm reduction and GEMM phases. Part of #4252.
            | nn_dsl::NativeOpKind::FusedLayerNormLinear { .. }
            // Fused Conv1d + Activation: conv uses F32 accumulators internally.
            // Part of #4264.
            | nn_dsl::NativeOpKind::FusedConv1dActivation { .. }
            // Fused Conv1d + Snake + InstanceNorm: conv and norm use F32
            // accumulators internally. Part of #4264.
            | nn_dsl::NativeOpKind::FusedConv1dSnakeNorm { .. }
            // Fused 2x (Conv1d + Snake + InstanceNorm) + residual add:
            // conv and norm use F32 accumulators internally. Part of #4264.
            | nn_dsl::NativeOpKind::FusedConv1dSnakeNormResBlock { .. }
            // Fused Add + InstanceNorm + Conv1d(K=1): norm and conv use F32
            // accumulators internally. Part of #4264.
            | nn_dsl::NativeOpKind::FusedAddInstanceNormConv1x1 { .. }
            // Fused ConvTranspose1d + Activation: conv uses F32 accumulators
            // internally. Part of #4264.
            | nn_dsl::NativeOpKind::FusedConvTranspose1dActivation { .. }
            // Fused AdainLeakyRelu/AdainSnake + ConvTranspose1d: InstanceNorm uses
            // F32 accumulators, conv uses F32 accumulators. Part of #4264.
            | nn_dsl::NativeOpKind::NormActivConvTranspose1d { .. }
            // Fused BatchNorm2d: uses F32 accumulators internally for the
            // normalize-scale-shift computation. Part of #4324.
            | nn_dsl::NativeOpKind::BatchNorm2d { .. }
            // Fused InstanceNorm + Conv1d: norm and conv use F32 accumulators.
            // Part of #4264.
            | nn_dsl::NativeOpKind::FusedInstanceNormConv1d { .. }
            // Fused Conv1d + InstanceNorm: conv and norm use F32 accumulators.
            // Part of #4264.
            | nn_dsl::NativeOpKind::FusedConv1dInstanceNorm { .. }
            // Fused Linear + LayerNorm: GEMM and norm use F32 accumulators.
            // Part of #4264.
            | nn_dsl::NativeOpKind::FusedLinearLayerNorm { .. }
            // Chained FusedResBlocks: norm and conv use F32 accumulators.
            // Part of #4264.
            | nn_dsl::NativeOpKind::FusedResBlockChain { .. }
    )
}

/// Whether a Dispatch-path Linear/MatMul is bandwidth-bound and benefits from
/// F16 weight storage even without simdgroup eligibility.
///
/// Same heuristic as [`is_bandwidth_bound_linear`] but for unfused
/// `TensorOpKind::Linear` / `TensorOpKind::MatMul` Dispatch steps.
///
/// Part of #4264.
pub(super) fn is_bandwidth_bound_dispatch(def: &nn_dsl::TensorKernelDef) -> bool {
    use nn_dsl::tensor_ir::TensorOpKind;
    let output_idx = def.output.index();
    if output_idx >= def.nodes.len() {
        return false;
    }
    match &def.nodes[output_idx].kind {
        TensorOpKind::Linear { input, weight, .. } => {
            let weight_node = def.nodes.get(weight.index());
            let input_node = def.nodes.get(input.index());
            if let (Some(wn), Some(inp)) = (weight_node, input_node) {
                if wn.shape.len() != 2 {
                    return false;
                }
                let out_features = wn.shape[0];
                let in_features = wn.shape[1];
                let m: usize = inp.shape.iter().rev().skip(1).product::<usize>().max(1);
                let weight_elements = in_features.checked_mul(out_features).unwrap_or(0);
                m <= 64 && weight_elements >= 65_536
            } else {
                false
            }
        }
        TensorOpKind::MatMul { left, right, .. } => {
            let left_node = def.nodes.get(left.index());
            let right_node = def.nodes.get(right.index());
            if let (Some(ln), Some(rn)) = (left_node, right_node) {
                if ln.shape.len() < 2 || rn.shape.len() < 2 {
                    return false;
                }
                let m = ln.shape[ln.shape.len() - 2];
                let k = ln.shape[ln.shape.len() - 1];
                let n = *rn.shape.last().unwrap_or(&0);
                let weight_elements = k.checked_mul(n).unwrap_or(0);
                m <= 64 && weight_elements >= 65_536
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Whether a `LinearActivation` NativeOp is bandwidth-bound and benefits from
/// F16 weight storage even without simdgroup eligibility.
///
/// At short sequences (small M), GEMM is bottlenecked on loading weight bytes
/// from device memory, not on ALU throughput. F16 halves weight bandwidth,
/// giving ~2x speedup on bandwidth-bound matmuls. The naive per-element kernel
/// already uses F32 accumulators (`msl_accumulator_type(F16) -> "float"`),
/// so precision is preserved.
///
/// Heuristic: weights dominate when K*N is large relative to M*K (i.e.,
/// the weight matrix is much larger than the activation vector).
/// We use `K*N >= 65536 && M <= 64` as the threshold — this captures
/// PlBERT at T<=64 (768*768=589,824 >> 65536) while excluding large-batch
/// cases where the simdgroup path should be used instead.
///
/// Part of #4264.
pub(super) fn is_bandwidth_bound_linear(op: &nn_dsl::NativeOpKind) -> bool {
    let nn_dsl::NativeOpKind::LinearActivation {
        in_features,
        out_features,
        input_shape,
        ..
    } = op
    else {
        return false;
    };
    if input_shape.is_empty() {
        return false;
    }
    let m = input_shape.iter().rev().skip(1).product::<usize>().max(1);
    // Weight matrix size: K * N (stored as [N, K] for column-major).
    let weight_elements = in_features.checked_mul(*out_features).unwrap_or(0);
    // Bandwidth-bound: large weight matrix relative to small activation batch.
    // K*N >= 65536 ensures meaningful bandwidth savings from F16.
    // M <= 64 ensures we're in the bandwidth-bound regime (few output rows).
    m <= 64 && weight_elements >= 65_536
}

/// Activation and data-movement ops safe to run in the predecessor's dtype.
/// Matches PyTorch's "implicit" autocast category: these ops preserve the
/// dtype of their input rather than promoting to F32. Part of #2981.
pub(super) fn is_passthrough_safe(step: &CompiledStep) -> bool {
    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            use nn_dsl::tensor_ir::TensorOpKind;
            let output_idx = kernel.def().output.index();
            if output_idx >= kernel.def().nodes.len() {
                return false;
            }
            matches!(
                &kernel.def().nodes[output_idx].kind,
                TensorOpKind::Relu { .. }
                    | TensorOpKind::LeakyRelu { .. }
                    | TensorOpKind::Elu { .. }
                    | TensorOpKind::Softplus { .. }
                    | TensorOpKind::Exp { .. }
                    | TensorOpKind::Sigmoid { .. }
                    | TensorOpKind::Tanh { .. }
                    | TensorOpKind::Gelu { .. }
                    | TensorOpKind::GeluErf { .. }
                    | TensorOpKind::BinaryAdd { .. }
                    | TensorOpKind::BinaryMul { .. }
                    | TensorOpKind::Reshape { .. }
                    | TensorOpKind::Narrow { .. }
                    | TensorOpKind::Transpose { .. }
                    | TensorOpKind::AxisSelect { .. }
                    | TensorOpKind::Stack { .. }
                    | TensorOpKind::Concat { .. }
                    | TensorOpKind::ZeroPad1d { .. }
            )
        }
        // ProjectionSlice is a GPU narrow (data-movement only) — safe to
        // inherit the parent's dtype. This propagates F16 from
        // BatchedLinearProjection through to downstream consumers. #3272.
        CompiledStep::NativeOp { op, .. } => {
            matches!(
                op,
                nn_dsl::NativeOpKind::ProjectionSlice { .. }
                    | nn_dsl::NativeOpKind::SiluMul { .. }
                    | nn_dsl::NativeOpKind::RotaryEmbedding { .. }
                    | nn_dsl::NativeOpKind::FusedMulAdd { .. }
                    | nn_dsl::NativeOpKind::FusedSiGLU { .. }
                    | nn_dsl::NativeOpKind::FusedGeGLU { .. }
            )
        }
        _ => false,
    }
}
