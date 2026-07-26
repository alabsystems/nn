// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for nn-verify integration tests.

// Monotonicity parametric test harness (Part of #1916).
#[allow(dead_code)]
pub(crate) mod monotonicity;

// Shared bounds contract parity helpers (Part of #1942).
#[allow(dead_code)]
pub(crate) mod bounds_helpers;

// Shared weight construction helpers (Part of #1938).
#[allow(dead_code)]
pub(crate) mod weights;

// Shared Demucs/Kokoro decoder topology configuration (Part of #1938).
#[allow(dead_code)]
pub(crate) mod demucs_topology;

// Shared attention decoder builder helpers (Part of #1970).
#[allow(dead_code)]
pub(crate) mod decoder_common;

// Shared Kokoro weight construction + bounds helpers (Part of #2404).
#[allow(dead_code)]
pub(crate) mod kokoro_weights;

// Shared Kokoro verification recording helpers (Part of #2623).
#[allow(dead_code)]
pub(crate) mod kokoro_recording;

// f64 tightness validation helpers (Part of #4316).
#[allow(dead_code)]
pub(crate) mod f64_tightness;

//
// Usage from any integration test:
// ```ignore
// mod common;
// use common::{snake_kernel, exp_kernel, ibp_scalar};
// ```

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{
    verify_tensor_and_record, verify_tensor_and_record_with_config, BoundedTensor,
    TensorParamBinding, TensorPipelineResult, VerificationSoundnessMode, VerifyConfig,
    VerifyStatus,
};
use ndarray::{ArrayD, IxDyn};
use std::path::PathBuf;

/// Resolve the workspace root directory.
fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Resolve the per-model status file path for a given kernel name (#2577).
fn status_path_for_kernel(kernel_name: &str) -> PathBuf {
    let model = nn_verify::model_for_kernel(kernel_name);
    nn_verify::model_status_path(&workspace_root(), model)
}

// Re-export kernel builders from the shared nn-dsl test utilities.
// These replace the local definitions that were duplicated across test files.
// Each integration test binary uses a different subset, so allow unused here.
#[allow(unused_imports)]
pub(crate) use nn_dsl::test_kernels::{
    binop_const_const_kernel, binop_var_var_kernel, compare_const_fold_kernel, compare_var_kernel,
    exp_kernel, parse_kernel, snake_kernel, square_kernel, unary_fn_kernel,
};

/// Run IBP on a scalar graph and return `(lower, upper)` as `f32`.
///
/// Creates a 1-element `BoundedTensor` input, propagates through the graph,
/// and extracts the scalar bounds from the output.
#[allow(dead_code)]
pub(crate) fn ibp_scalar(graph: &nn_verify::GraphNetwork, lo: f32, hi: f32) -> (f32, f32) {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), lo),
        ArrayD::from_elem(IxDyn(&[1]), hi),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (out_lo, out_hi) = output.lower_upper();
    (out_lo[[0]], out_hi[[0]])
}

/// Evaluate a constant-folded compare by running IBP with x=0.
///
/// Builds a graph from `compare_const_fold_kernel(op)` with constant params
/// `a` and `b`, then propagates a zero input through IBP and returns the
/// lower bound (which equals the folded result).
#[allow(dead_code)]
pub(crate) fn eval_compare_fold(op: nn_dsl::ir::CompareOpKind, a: f32, b: f32) -> f32 {
    let kernel = compare_const_fold_kernel(op);
    let graph = nn_verify::kernel_to_graph(&kernel, &[a, b])
        .unwrap_or_else(|e| panic!("build compare {op:?}({a},{b}) graph: {e}"));
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    output.lower_upper().0[[0]]
}

// ---------------------------------------------------------------------------
// Causal mask construction + graph propagation (Part of #1970)
// ---------------------------------------------------------------------------

/// Large negative value for masked attention positions.
///
/// Using -1e9 instead of -inf to keep NY numerics stable.
/// Softmax(-1e9) ≈ 0, which is functionally equivalent to true masking.
#[allow(dead_code)]
pub(crate) const MASK_VALUE: f32 = -1e9;

/// Strict causal: `f(t) = min(t, T_enc - 1)`.
#[allow(dead_code)]
pub(crate) fn strict_causal_alignment(t: usize, t_enc: usize) -> usize {
    t.min(t_enc.saturating_sub(1))
}

/// Build a causal mask tensor `[t_dec, t_enc]` using the given alignment.
#[allow(dead_code)]
pub(crate) fn build_causal_mask(
    t_dec: usize,
    t_enc: usize,
    alignment_fn: impl Fn(usize) -> usize,
) -> ArrayD<f32> {
    let mut data = vec![0.0f32; t_dec * t_enc];
    for t in 0..t_dec {
        let max_pos = alignment_fn(t);
        for j in 0..t_enc {
            if j > max_pos {
                data[t * t_enc + j] = MASK_VALUE;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[t_dec, t_enc]), data).expect("valid mask shape")
}

/// Build a strict causal mask.
#[allow(dead_code)]
pub(crate) fn build_strict_causal_mask(t_dec: usize, t_enc: usize) -> ArrayD<f32> {
    build_causal_mask(t_dec, t_enc, |t| strict_causal_alignment(t, t_enc))
}

/// Propagate through tensor graph with IBP and return output bounds.
#[allow(dead_code)]
pub(crate) fn graph_propagate(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input: &BoundedTensor,
) -> BoundedTensor {
    let graph = nn_verify::tensor_kernel_to_graph(def, bindings).expect("graph");
    graph.propagate_ibp(input).expect("IBP")
}

// ---------------------------------------------------------------------------
// Sinusoidal positional encoding (Part of #1970)
// ---------------------------------------------------------------------------

/// Standard sinusoidal positional encoding: PE[t, 2i] = sin(t / 10000^(2i/D)).
///
/// Key property: PE vectors at different positions are approximately orthogonal,
/// so PE @ PE^T is diagonally dominant.
#[allow(dead_code)]
pub(crate) fn sinusoidal_pe(seq_len: usize, d_model: usize) -> ArrayD<f32> {
    let mut data = vec![0.0f32; seq_len * d_model];
    for t in 0..seq_len {
        for i in 0..d_model / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
            data[t * d_model + 2 * i] = freq.sin() as f32;
            data[t * d_model + 2 * i + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), data).expect("valid PE")
}

/// Sinusoidal positional encoding with head-interleaved frequencies.
///
/// Unlike standard PE, this variant reorders frequency indices so that
/// each attention head gets a distinct subset of frequencies, interleaved
/// across the model dimension. Used by multi-head attention stacks.
#[allow(dead_code)]
pub(crate) fn sinusoidal_pe_interleaved(
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
) -> ArrayD<f32> {
    let d_k = d_model / num_heads;
    let num_pairs = d_model / 2;
    let pairs_per_head = d_k / 2;

    let mut dim_to_freq = vec![0usize; num_pairs];
    for h in 0..num_heads {
        for p in 0..pairs_per_head {
            let freq_idx = h + p * num_heads;
            let out_pair = h * pairs_per_head + p;
            if freq_idx < num_pairs && out_pair < num_pairs {
                dim_to_freq[out_pair] = freq_idx;
            }
        }
    }

    let mut data = vec![0.0f32; seq_len * d_model];
    for t in 0..seq_len {
        for pair in 0..num_pairs {
            let freq_idx = dim_to_freq[pair];
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * freq_idx as f64 / d_model as f64);
            data[t * d_model + 2 * pair] = freq.sin() as f32;
            data[t * d_model + 2 * pair + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), data).expect("valid PE")
}

// ---------------------------------------------------------------------------
// Tensor-level test helpers (Layer 2)
// ---------------------------------------------------------------------------

/// PyTorch default epsilon for InstanceNorm, GroupNorm, LayerNorm, RMSNorm.
///
/// Use this constant in new tests instead of hardcoding `1e-5_f32`.
/// Existing 48+ occurrences can migrate incrementally.
#[allow(dead_code)]
pub(crate) const DEFAULT_NORM_EPS: f32 = 1e-5;

/// Conv1d output length formula: `(in_len + 2*padding - kernel) / stride + 1`.
///
/// Delegates to canonical `nn_core::conv1d_out_len` (dilation=1).
///
/// Note: parameter order here is `(in_len, kernel_size, stride, padding)` which
/// differs from canonical `(input_len, kernel_size, padding, stride, dilation)`.
/// This wrapper preserves the test-side convention.
#[allow(dead_code)]
pub(crate) fn conv1d_out_len(
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> usize {
    nn_core::conv1d_out_len(in_len, kernel_size, padding, stride, 1)
        .expect("conv1d_out_len: invalid parameters")
}

/// ConvTranspose1d output length: `(in_len - 1) * stride + kernel - 2*padding`.
///
/// Replaces identical copies in compose_four_block_decoder,
/// compose_decoder_conv_transpose, and compose_demucs_decoder_block.
#[allow(dead_code)]
pub(crate) fn conv_transpose_out_len(
    in_len: usize,
    stride: usize,
    kernel_size: usize,
    padding: usize,
) -> usize {
    (in_len - 1) * stride + kernel_size - 2 * padding
}

/// Linear alignment: `f(t) = floor(t * t_enc / t_dec)`.
///
/// Replaces identical copies in causal_attention, multi_head_causal,
/// and softmax_attention helpers.
#[allow(dead_code)]
pub(crate) fn linear_alignment(t: usize, t_dec: usize, t_enc: usize) -> usize {
    (t * t_enc / t_dec).min(t_enc.saturating_sub(1))
}

/// Build a linear causal mask.
///
/// Replaces identical copies in causal_attention, multi_head_causal,
/// and softmax_attention helpers.
#[allow(dead_code)]
pub(crate) fn build_linear_causal_mask(t_dec: usize, t_enc: usize) -> ArrayD<f32> {
    build_causal_mask(t_dec, t_enc, |t| linear_alignment(t, t_dec, t_enc))
}

// ---------------------------------------------------------------------------
// Tensor-level test helpers (Layer 3) — compose test dedup (#844)
// ---------------------------------------------------------------------------

/// Assert all bounds are finite and lower <= upper.
///
/// Replaces the 4-line loop pattern duplicated across 27+ compose test files:
/// ```ignore
/// // NOTE: ignore — code pattern with undefined variables lo/hi
/// for (&l, &u) in lo.iter().zip(hi.iter()) {
///     assert!(l.is_finite() && u.is_finite(), ...);
///     assert!(l <= u, ...);
/// }
/// ```
#[allow(dead_code)]
pub(crate) fn assert_bounds_valid(bounds: &BoundedTensor) {
    let (lo, hi) = bounds.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            l.is_finite() && u.is_finite(),
            "bounds must be finite: got ({l}, {u})"
        );
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// Assert bounds are valid and that CROWN bounds are at least as tight as IBP bounds.
///
/// Verifies the fundamental soundness invariant:
///   crown_lo[i] >= ibp_lo[i] - eps  and  crown_hi[i] <= ibp_hi[i] + eps
///
/// Replaces the 8+ line comparison pattern in 12+ compose test files.
/// Uses `eps` = 1e-4 tolerance for numerical differences between propagation paths.
#[allow(dead_code)]
pub(crate) fn assert_crown_tighter_than_ibp(crown: &BoundedTensor, ibp: &BoundedTensor) {
    assert_bounds_valid(crown);
    assert_bounds_valid(ibp);
    let (crown_lo, crown_hi) = crown.lower_upper();
    let (ibp_lo, ibp_hi) = ibp.lower_upper();
    let eps = 1e-4;
    for (&cl, &il) in crown_lo.iter().zip(ibp_lo.iter()) {
        assert!(
            cl >= il - eps,
            "CROWN lower {cl} should be >= IBP lower {il} (tighter)"
        );
    }
    for (&cu, &iu) in crown_hi.iter().zip(ibp_hi.iter()) {
        assert!(
            cu <= iu + eps,
            "CROWN upper {cu} should be <= IBP upper {iu} (tighter)"
        );
    }
}

/// Assert CROWN tighter-than-IBP when CROWN succeeded (no fallback).
///
/// For complex models where CROWN may fall back to IBP (e.g., due to
/// decomposed GroupNorm or GLU multiplicative interactions), this function
/// runs IBP first, then CROWN with fallback, and only asserts the tightness
/// invariant when CROWN actually succeeded (`PropMethod::Crown`).
///
/// **Warning (#1769):** When CROWN falls back to IBP, the tightness assertion
/// is skipped and only structural validity is checked. This means tests using
/// this helper can silently degrade to IBP-only verification. Always check
/// the returned `method` in the caller if CROWN success is required.
///
/// **Warning (#2715):** Even when CROWN "succeeds" (no fallback), bounds
/// through normalization layers may be FALLBACK_BOUND-capped (~2e10 wide)
/// — vacuously wide and no tighter than IBP. CROWN success is a structural
/// property (the linearization completed), not a tightness guarantee. For
/// chained normalization (e.g., Kokoro 58-layer InstanceNorm), Conservative
/// IBP produces 276M× tighter bounds than CROWN. Check output width in the
/// caller if non-vacuous bounds are required.
///
/// Returns `(method, crown_output, fallback_reason)` for caller logging.
#[allow(dead_code)]
pub(crate) fn assert_crown_tighter_when_not_fallback(
    graph: &nn_verify::GraphNetwork,
    input: &BoundedTensor,
) -> (nn_verify::PropMethod, BoundedTensor, Option<String>) {
    let ibp_output = graph.propagate_ibp(input).expect("IBP baseline");

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(graph, input).expect("propagation");

    assert_bounds_valid(&crown_output);

    if matches!(method, nn_verify::PropMethod::Crown) {
        // #2715: CROWN through normalization layers may produce bounds wider
        // than IBP (vacuously wide). Check per-element tightness and only
        // assert when CROWN is actually tighter on all elements.
        let (crown_lo, crown_hi) = crown_output.lower_upper();
        let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
        let eps = 1e-4;
        let crown_is_tighter = crown_lo
            .iter()
            .zip(ibp_lo.iter())
            .all(|(&cl, &il)| cl >= il - eps)
            && crown_hi
                .iter()
                .zip(ibp_hi.iter())
                .all(|(&cu, &iu)| cu <= iu + eps);
        if crown_is_tighter {
            assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
        } else {
            eprintln!(
                "WARNING (#2715): CROWN succeeded but some bounds are wider than IBP. \
                 Tightness assertion skipped — likely normalization layer vacuous bounds."
            );
        }
    } else {
        // P1-251 (#1769): Explicitly log when CROWN falls back to IBP.
        // This makes the silent degradation visible in test output.
        eprintln!(
            "WARNING: CROWN fell back to IBP (method={method:?}). \
             Tighter-than-IBP assertion SKIPPED. \
             Reason: {}",
            fallback_reason.as_deref().unwrap_or("unknown")
        );
    }

    (method, crown_output, fallback_reason)
}

/// Assert CROWN succeeds (does NOT fall back to IBP) and produces tighter bounds.
///
/// Unlike `assert_crown_tighter_when_not_fallback` which silently accepts IBP
/// fallback, this function panics if CROWN falls back. Use this for tests where
/// CROWN is expected to work (e.g., Kokoro decoder after NY 359a195+).
///
/// Returns the CROWN output bounds for caller use.
#[allow(dead_code)]
pub(crate) fn assert_crown_succeeds(
    graph: &nn_verify::GraphNetwork,
    input: &BoundedTensor,
) -> BoundedTensor {
    let ibp_output = graph.propagate_ibp(input).expect("IBP baseline");

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(graph, input).expect("propagation");

    assert!(
        matches!(method, nn_verify::PropMethod::Crown),
        "CROWN must not fall back to IBP. Fallback reason: {}",
        fallback_reason.as_deref().unwrap_or("unknown")
    );

    assert_bounds_valid(&crown_output);
    assert_crown_tighter_than_ibp(&crown_output, &ibp_output);

    crown_output
}

/// Create uniform BoundedTensor: all lower = -range, all upper = +range.
///
/// Replaces the 4-line `ArrayD::from_elem` + `BoundedTensor::new` pattern
/// duplicated in 12+ `*_input_bounds()` functions across compose test files.
#[allow(dead_code)]
pub(crate) fn uniform_bounds(shape: &[usize], range: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), -range),
        ArrayD::from_elem(IxDyn(shape), range),
    )
    .expect("valid uniform bounds")
}

/// Run `verify_tensor_and_record` and assert standard invariants.
///
/// Loads the workspace `nn_verify_status.json` with an exclusive lock,
/// runs the tensor pipeline, asserts:
///   - output bounds are finite
///   - all lower <= upper
///   - status contains an entry for `status_key`
///
/// Then persists the updated status back to disk (#2221).
///
/// Uses `load_locked()` + `save()` to prevent TOCTOU races (#482).
///
/// Replaces the 10-line pipeline pattern duplicated in 20+ compose test files.
#[allow(dead_code)]
pub(crate) fn verify_and_assert(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input: &BoundedTensor,
    status_key: &str,
) -> TensorPipelineResult {
    let model_path = status_path_for_kernel(status_key);
    let mut locked = VerifyStatus::load_locked(&model_path).expect("load_locked per-model status");
    let result =
        verify_tensor_and_record(&mut locked.status, def, bindings, input, Some(status_key))
            .expect("verify_tensor_and_record pipeline");
    assert!(
        result.verification.is_finite,
        "output bounds must be finite"
    );
    assert_bounds_valid(&result.output_bounds);
    assert!(
        locked.status.kernel(status_key).is_some(),
        "status should contain entry for '{status_key}'"
    );
    if result.verification.soundness_mode == VerificationSoundnessMode::Heuristic {
        locked
            .status
            .set_soundness_justification(
                status_key,
                "NY propagation used heuristic normalization approximation",
            )
            .expect("set justification");
    }
    locked.save().expect("save per-model status");
    result
}

/// Run `verify_tensor_and_record_with_config` with a custom `VerifyConfig`.
///
/// Same invariant assertions as `verify_and_assert` but accepts a config
/// for controlling NormBoundsMode, escalation threshold, etc.
/// Persists results to `nn_verify_status.json` via `load_locked()` + `save()` (#2221).
#[allow(dead_code)]
pub(crate) fn verify_and_assert_with_config(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input: &BoundedTensor,
    status_key: &str,
    config: &VerifyConfig,
) -> TensorPipelineResult {
    let model_path = status_path_for_kernel(status_key);
    let mut locked = VerifyStatus::load_locked(&model_path).expect("load_locked per-model status");
    let result = verify_tensor_and_record_with_config(
        &mut locked.status,
        def,
        bindings,
        input,
        Some(status_key),
        config,
    )
    .expect("verify_tensor_and_record_with_config pipeline");
    assert!(
        result.verification.is_finite,
        "output bounds must be finite"
    );
    assert_bounds_valid(&result.output_bounds);
    assert!(
        locked.status.kernel(status_key).is_some(),
        "status should contain entry for '{status_key}'"
    );
    if result.verification.soundness_mode == VerificationSoundnessMode::Heuristic {
        locked
            .status
            .set_soundness_justification(
                status_key,
                "NY propagation used heuristic normalization approximation",
            )
            .expect("set justification");
    }
    locked.save().expect("save per-model status");
    result
}

/// Extract the global (min lower, max upper) scalar range from a `BoundedTensor`.
///
/// Returns `(lo_min, hi_max)` where `lo_min` is the minimum across all lower-bound
/// elements and `hi_max` is the maximum across all upper-bound elements.
///
/// Replaces the 2-line fold pattern duplicated across 40+ compose test files:
/// ```ignore
/// let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
/// let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
/// ```
#[allow(dead_code)]
pub(crate) fn bounds_min_max(bounds: &BoundedTensor) -> (f32, f32) {
    let (lo, hi) = bounds.lower_upper();
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (lo_min, hi_max)
}

/// Assert that the output bounds width (hi_max - lo_min) is below a threshold.
///
/// This catches vacuously wide IBP results (e.g., `[-1e30, 1e30]`) that would
/// otherwise pass finiteness-only assertions. The `max_width` parameter should be
/// calibrated per pipeline segment based on actual IBP runs.
///
/// Part of #2594: bounds tightness assertions for Kokoro compose tests.
#[allow(dead_code)]
pub(crate) fn assert_bounds_width(bounds: &BoundedTensor, max_width: f32, label: &str) {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "{label}: bounds must be finite, got [{lo_min}, {hi_max}]"
    );
    let width = hi_max - lo_min;
    assert!(
        width < max_width,
        "{label}: bounds width {width} exceeds threshold {max_width} (bounds=[{lo_min}, {hi_max}])"
    );
}

/// Create a 1-element `BoundedTensor` with asymmetric `[lo, hi]` bounds.
///
/// Unlike `uniform_bounds` (which is symmetric ±range), this accepts
/// arbitrary lo/hi values for scalar kernel verification.
///
/// Replaces 3 identical copies in `compose_sequential`, `compose_sequential_dvoice`,
/// and `compose_k2_k4_verification`.
#[allow(dead_code)]
pub(crate) fn scalar_bounds(lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), lo),
        ArrayD::from_elem(IxDyn(&[1]), hi),
    )
    .expect("valid scalar bounds")
}

/// Extract `(lower, upper)` from a scalar (1-element) `BoundedTensor`.
///
/// Replaces 3 identical copies in `compose_sequential`, `compose_sequential_dvoice`,
/// and `compose_k2_k4_verification`.
#[allow(dead_code)]
pub(crate) fn extract_scalar(bt: &BoundedTensor) -> (f32, f32) {
    let (lo, hi) = bt.lower_upper();
    (lo[[0]], hi[[0]])
}

/// Assert that a normalization spatial dimension is non-degenerate.
///
/// InstanceNorm/GroupNorm on a single spatial element is mathematically
/// degenerate: mean equals the value, variance is zero, so the output
/// is always the bias term regardless of input. Verification bounds
/// computed at spatial dim=1 **cannot be extrapolated** to production
/// dimensions because normalization statistics fundamentally depend on
/// the number of elements being reduced over.
///
/// Call this in compose test builders when constructing graphs that
/// include normalization layers. `spatial_size` is the number of elements
/// the norm reduces over (e.g., `T` for InstanceNorm, `C*T` for
/// GroupNorm(g=1)).
///
/// See #2637.
#[allow(dead_code)]
pub(crate) fn assert_norm_spatial_non_degenerate(spatial_size: usize, label: &str) {
    assert!(
        spatial_size > 1,
        "{label}: normalization spatial dimension is {spatial_size} — degenerate. \
         InstanceNorm/GroupNorm on a single element has mean=value, var=0, making \
         the output equal to the bias term regardless of input. Bounds at spatial \
         dim=1 cannot be extrapolated to production dimensions. See #2637.",
    );
}

/// Create high-variance element-wise bounds for any shape.
///
/// Each element gets a different center point spread across `[-spread, +spread]`
/// with a small perturbation radius `r`. This triggers the pathological case for
/// Conservative IBP through normalization layers: element-wise bounds differ
/// significantly, amplifying mean/variance uncertainty.
///
/// ForwardMode anchors to the midpoint, producing ~50-1000x tighter bounds.
#[allow(dead_code)]
pub(crate) fn high_variance_bounds(shape: &[usize], spread: f32, r: f32) -> BoundedTensor {
    let n: usize = shape.iter().product();
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    for i in 0..n {
        // Evenly space centers across [-spread, +spread]
        let center = if n > 1 {
            -spread + 2.0 * spread * (i as f32) / ((n - 1) as f32)
        } else {
            0.0
        };
        lower.push(center - r);
        upper.push(center + r);
    }
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).expect("valid lower"),
        ArrayD::from_shape_vec(IxDyn(shape), upper).expect("valid upper"),
    )
    .expect("valid high-variance bounds")
}
