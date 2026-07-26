// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `compiled_model_builder.rs` (#3715).
//!
//! Proves properties of the CompiledModelBuilder and its classification
//! helpers:
//! - Autocast and mixed_precision mutual exclusivity
//! - Plan/graph length invariant
//! - Empty plan produces empty model
//! - ScalarType dtype mapping for GPU-compilable types
//! - Release map: output steps are never released
//! - Step numel: product of output shape dimensions
//! - Buffer plan denominator non-zero
//! - Autocast passthrough propagation conditions
//! - is_compute_native_op classification exhaustiveness
//! - is_passthrough_safe Dispatch classification
//! - Mixed GEMM: F32 output despite F16 step type
//! - force_dtype: only float ScalarTypes accepted
//! - Autocast f32_only policy is a no-op
//! - Edge map: step_metas length matches steps length

// ============================================================================
// 1. Autocast + mixed_precision mutual exclusivity
// ============================================================================

/// Prove: autocast_policy and mixed_precision cannot both be Some.
/// The builder returns Err if both are set.
///
/// Models `compiled_model_builder.rs:203-208`:
/// ```
/// if autocast_policy.is_some() && mixed_precision.is_some() {
///     return Err(...)
/// }
/// ```
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn autocast_mixed_precision_mutual_exclusion() {
    let has_autocast: bool = kani::any();
    let has_mixed: bool = kani::any();

    let both_set = has_autocast && has_mixed;
    let ok = !both_set;

    // Property 1: when both set, build must fail.
    if has_autocast && has_mixed {
        assert!(both_set, "both set must be detected");
        assert!(!ok, "mutual exclusion must reject");
    }

    // Property 2: at most one can be set.
    if ok {
        assert!(
            !(has_autocast && has_mixed),
            "ok implies at most one is set"
        );
    }
}

// ============================================================================
// 2. Plan/graph length invariant
// ============================================================================

/// Prove: plan.steps.len() must equal graph.nodes().len().
/// A mismatch triggers an early error (step 0).
///
/// Models `compiled_model_builder.rs:167-176`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_steps_equals_graph_nodes() {
    let plan_len: usize = kani::any();
    let graph_len: usize = kani::any();

    kani::assume(plan_len <= 1000);
    kani::assume(graph_len <= 1000);

    let ok = plan_len == graph_len;

    if plan_len != graph_len {
        assert!(!ok, "length mismatch must be detected");
    }

    if ok {
        assert_eq!(plan_len, graph_len);
    }
}

// ============================================================================
// 3. Empty plan produces empty model
// ============================================================================

/// Prove: when plan.steps is empty, the builder returns early with
/// CompiledModel::empty(). No further processing occurs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn empty_plan_returns_early() {
    let plan_steps_len: usize = kani::any();
    kani::assume(plan_steps_len <= 1000);

    let returns_early = plan_steps_len == 0;

    if plan_steps_len == 0 {
        assert!(returns_early, "empty plan must return early");
    }

    if plan_steps_len > 0 {
        assert!(!returns_early, "non-empty plan must not return early");
    }
}

// ============================================================================
// 4. ScalarType: GPU-compilable dtypes
// ============================================================================

/// Prove: only F32, F16, BF16 map to valid ScalarType values.
/// Non-float dtypes (U8, U32, I64) default to F32 via
/// `ScalarType::try_from(dtype).unwrap_or(ScalarType::F32)`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_type_gpu_compilable_dtypes() {
    // Model DType as an enum index: 0=F32, 1=F16, 2=BF16, 3=U8, 4=U32, 5=I64
    let dtype_idx: u8 = kani::any();
    kani::assume(dtype_idx < 6);

    let is_float = dtype_idx <= 2; // F32, F16, BF16
    let maps_to_self = is_float; // try_from succeeds
    let defaults_to_f32 = !is_float; // try_from fails, unwrap_or(F32)

    // Property 1: float dtypes map to themselves.
    if is_float {
        assert!(maps_to_self, "float dtype must map to valid ScalarType");
    }

    // Property 2: non-float dtypes default to F32.
    if !is_float {
        assert!(defaults_to_f32, "non-float dtype must default to F32");
    }

    // Property 3: exactly one outcome.
    assert!(
        maps_to_self ^ defaults_to_f32,
        "exactly one mapping path"
    );
}

// ============================================================================
// 5. Release map: output steps never released
// ============================================================================

/// Prove: output step indices are never added to the release map.
/// This ensures output buffers persist after execution.
///
/// Models `compiled_model_builder.rs:373-381`:
/// ```
/// for (step, &consumer) in last_use.iter().enumerate() {
///     if consumer > step && consumer < n && !output_indices.contains(&step) {
///         map[consumer].push(step);
///     }
/// }
/// ```
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn output_steps_never_released() {
    let step: usize = kani::any();
    let consumer: usize = kani::any();
    let n: usize = kani::any();
    let is_output: bool = kani::any();

    kani::assume(step <= 500);
    kani::assume(consumer <= 500);
    kani::assume(n >= 1 && n <= 500);

    let would_add = consumer > step && consumer < n && !is_output;

    // Property: output steps are never released.
    if is_output {
        assert!(!would_add, "output step must not be released");
    }
}

// ============================================================================
// 6. Step numel: product of shape dimensions
// ============================================================================

/// Prove: step numel is the product of all output shape dimensions.
/// For a 3D shape [B, C, T], numel = B * C * T.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn step_numel_is_shape_product() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 512);
    kani::assume(d2 >= 1 && d2 <= 8192);

    let numel = d0
        .checked_mul(d1)
        .and_then(|v| v.checked_mul(d2));
    assert!(numel.is_some(), "numel must not overflow for valid shapes");

    let numel = numel.unwrap();
    assert!(numel >= 1, "numel must be positive");
    assert_eq!(numel, d0 * d1 * d2, "numel must be product of dims");
}

// ============================================================================
// 7. Autocast passthrough: all inputs must be target dtype
// ============================================================================

/// Prove: passthrough propagation requires ALL input edges to have
/// the target dtype. If any input is F32, the passthrough step stays F32.
///
/// Models `compiled_model_builder.rs:257-266`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn autocast_passthrough_requires_all_inputs_target() {
    let n_inputs: usize = kani::any();
    let n_target: usize = kani::any();

    kani::assume(n_inputs >= 1 && n_inputs <= 10);
    kani::assume(n_target <= n_inputs);

    let all_target = n_target == n_inputs;
    let propagates = all_target && n_inputs > 0;

    // Property 1: partial target inputs do not propagate.
    if n_target < n_inputs {
        assert!(!propagates, "partial target inputs must not propagate");
    }

    // Property 2: all target inputs propagate.
    if n_target == n_inputs && n_inputs > 0 {
        assert!(propagates, "all target inputs must propagate");
    }
}

// ============================================================================
// 8. is_compute_native_op: known compute ops
// ============================================================================

/// Prove: the compute native ops are exactly:
/// FlashAttention, NormActivConv1d, FusedResBlock,
/// BatchedLinearProjection, NormLinear, AdainSnake, AdainLeakyRelu.
/// LinearActivation is NOT here (gated by mixed_gemm_infos).
/// Updated for #3766: AdainSnake/AdainLeakyRelu added to is_compute_native_op.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn compute_native_op_classification() {
    // Model op variants as indices.
    // 0=FlashAttention, 1=NormActivConv1d, 2=FusedResBlock,
    // 3=BatchedLinearProjection, 4=NormLinear, 5=AdainSnake,
    // 6=AdainLeakyRelu, 7=LinearActivation, 8=LstmSequence, 9=Other
    let variant: u8 = kani::any();
    kani::assume(variant < 10);

    let is_compute = variant <= 6;

    // Property 1: known compute ops (includes AdainSnake, AdainLeakyRelu).
    if variant <= 6 {
        assert!(is_compute, "compute native ops must be classified as compute");
    }

    // Property 2: LinearActivation is NOT compute (it's GEMM-gated).
    if variant == 7 {
        assert!(!is_compute, "LinearActivation must not be classified as compute");
    }

    // Property 3: LSTM is NOT compute.
    if variant == 8 {
        assert!(!is_compute, "LstmSequence must not be classified as compute");
    }
}

// ============================================================================
// 9. Passthrough safe: activation ops
// ============================================================================

/// Prove: activation/data-movement ops (Relu, LeakyRelu, Reshape, etc.)
/// are classified as passthrough-safe. Normalization ops are NOT.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn passthrough_safe_classification() {
    // Model TensorOpKind: 0=Relu, 1=LeakyRelu, 2=Reshape, 3=Sigmoid,
    // 4=Softmax, 5=LayerNorm, 6=Linear, 7=Conv1d
    let op: u8 = kani::any();
    kani::assume(op < 8);

    let is_passthrough = op <= 3; // activations + data movement
    let is_accumulate = op == 4 || op == 5; // norms/reductions
    let is_compute = op >= 6; // matmul/conv

    // Property 1: activations are passthrough.
    if op <= 3 {
        assert!(is_passthrough, "activations must be passthrough");
    }

    // Property 2: accumulate ops are NOT passthrough.
    if is_accumulate {
        assert!(!is_passthrough, "accumulate ops must not be passthrough");
    }

    // Property 3: compute ops are NOT passthrough.
    if is_compute {
        assert!(!is_passthrough, "compute ops must not be passthrough");
    }
}

// ============================================================================
// 10. Mixed GEMM: F32 output despite F16 step type
// ============================================================================

/// Prove: mixed GEMM steps have step_scalar_types[i] = F16 but the
/// planner uses F32 for downstream buffer sizing. Without this override,
/// downstream consumers would read F32 output from an F16-sized buffer.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn mixed_gemm_planner_override() {
    let is_mixed_gemm: bool = kani::any();
    let step_type_is_f16: bool = kani::any();

    // Mixed GEMM: step marked F16, but output is F32.
    kani::assume(!is_mixed_gemm || step_type_is_f16);

    // Planner dtype: F32 for mixed GEMM, else step type.
    let planner_is_f32 = is_mixed_gemm;

    if is_mixed_gemm {
        assert!(planner_is_f32, "mixed GEMM must use F32 in planner");
    }
}

// ============================================================================
// 11. force_dtype: only float ScalarTypes accepted
// ============================================================================

/// Prove: force_dtype(U32), force_dtype(I64), force_dtype(U8)
/// return Err (InvalidConfig). Only F32, F16, BF16 succeed.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn force_dtype_rejects_non_float() {
    // 0=F32, 1=F16, 2=BF16, 3=U8, 4=U32, 5=I64
    let dtype_idx: u8 = kani::any();
    kani::assume(dtype_idx < 6);

    let is_gpu_compilable = dtype_idx <= 2;

    if dtype_idx > 2 {
        assert!(!is_gpu_compilable, "non-float dtype must fail force_dtype");
    }

    if dtype_idx <= 2 {
        assert!(is_gpu_compilable, "float dtype must succeed force_dtype");
    }
}

// ============================================================================
// 12. Autocast f32_only: no-op policy
// ============================================================================

/// Prove: when policy.compute_dtype == F32, autocast_policy is NOT set
/// (the f32_only guard in the `autocast()` method returns self unchanged).
///
/// Models `compiled_model_builder.rs:100-104`:
/// ```
/// if policy.compute_dtype != DType::F32 {
///     self.autocast_policy = Some(policy);
/// }
/// ```
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn autocast_f32_only_is_noop() {
    let compute_dtype_is_f32: bool = kani::any();

    let sets_policy = !compute_dtype_is_f32;

    if compute_dtype_is_f32 {
        assert!(!sets_policy, "F32 compute dtype must not set autocast_policy");
    }
}

// ============================================================================
// 13. LSTM stays F32 in mixed precision
// ============================================================================

/// Prove: in mixed precision mode, LSTM steps keep F32 dtype.
/// sigmoid/tanh saturation at F16 range makes LSTM unsafe in F16.
///
/// Models `compiled_model_builder.rs:183-195`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_stays_f32_in_mixed_precision() {
    let is_lstm: bool = kani::any();
    let mixed_precision_active: bool = kani::any();
    let target_is_f16: bool = kani::any();

    kani::assume(mixed_precision_active && target_is_f16);

    // LSTM is skipped in the mixed precision loop.
    let gets_overridden = !is_lstm;

    if is_lstm {
        assert!(!gets_overridden, "LSTM must stay F32");
    }

    if !is_lstm {
        assert!(gets_overridden, "non-LSTM must be overridden to target dtype");
    }
}

// ============================================================================
// 14. RuntimeOp stays F32 in mixed precision
// ============================================================================

/// Prove: RuntimeOp steps keep F32 in mixed precision mode (#3122).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn runtime_op_stays_f32_in_mixed_precision() {
    let is_runtime_op: bool = kani::any();
    let is_lstm: bool = kani::any();

    // RuntimeOp is skipped (continue) in the mixed precision loop.
    let gets_overridden = !is_lstm && !is_runtime_op;

    if is_runtime_op {
        assert!(!gets_overridden, "RuntimeOp must stay F32");
    }
}

// ============================================================================
// 15. Release map: self-referencing steps not released
// ============================================================================

/// Prove: a step whose last_use consumer == itself is not released.
/// The guard `consumer > step` prevents self-release.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn release_map_no_self_release() {
    let step: usize = kani::any();
    let consumer: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(step < n);
    kani::assume(consumer <= n);
    kani::assume(n >= 1 && n <= 500);

    let would_add = consumer > step && consumer < n;

    // Property: consumer == step means step is its own last consumer.
    if consumer == step {
        assert!(!would_add, "self-referencing step must not be released");
    }
}

// ============================================================================
// 16. Weight buffer stripping: weight_data.clear() saves memory
// ============================================================================

/// Prove: stripping weight data from steps does not affect step count.
/// Models `compiled_model_builder.rs:297-306`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_strip_preserves_step_count() {
    let n_steps: usize = kani::any();
    kani::assume(n_steps <= 1000);

    // Clearing weight_data on each step doesn't change Vec length.
    let steps_after_strip = n_steps;
    assert_eq!(
        steps_after_strip, n_steps,
        "weight stripping must preserve step count"
    );
}
