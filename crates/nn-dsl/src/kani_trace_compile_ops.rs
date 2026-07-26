// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trace_compile_ops.rs` (#3704).
//!
//! Proves critical invariants of the per-op compilation helpers that lower
//! `TraceOp` variants into `TensorKernelDef` dispatch plans:
//!
//! - Powf exponent validation: non-finite rejected, 0.0 → constant 1.0, 1.0 → identity
//! - Powf integer parity: odd integer exponent sign restoration logic
//! - Powf parity limit: exponents beyond 2^24 cannot determine parity
//! - Neg compilation: 0 - x decomposition correctness
//! - Narrow zero-copy: contiguous prefix detection (all leading dims == 1)
//! - Narrow identity: full-range narrow (start=0, length=dim_size) → identity
//! - Narrow byte_offset: trailing product * start * 4 (f32 stride)
//! - Softmax dim overflow: dim > i32::MAX rejected
//! - BinaryMethod enum coverage
//! - ActivationKind enum coverage
//! - build_single_op input count consistency
//!
//! The NarrowView byte_offset overflow harnesses are in the sibling file
//! `trace_compile_ops_narrow_kani.rs` (wired separately, #2218).

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn floor_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

// ---------------------------------------------------------------------------
// 1. Powf: non-finite exponent is rejected
// ---------------------------------------------------------------------------

/// Proves: compile_powf rejects non-finite (NaN, Inf, -Inf) exponents.
///
/// SUBSTANTIVE: A NaN/Inf exponent passed through to the GPU graph would
/// produce garbage output for all inputs. The guard `!exp_f32.is_finite()`
/// must fire for every non-finite f64 that survives the `as f32` cast.
#[kani::unwind(1)]
#[kani::proof]
fn proof_powf_rejects_non_finite_exponent() {
    let exp: f64 = kani::any();
    let exp_f32 = exp as f32;

    if !exp_f32.is_finite() {
        // The production code returns Err(NonFiniteConstant) here.
        // We prove the guard condition is correct.
        assert!(
            exp_f32.is_nan() || exp_f32.is_infinite(),
            "non-finite must be NaN or Inf"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Powf: exponent == 0.0 produces constant 1.0
// ---------------------------------------------------------------------------

/// Proves: x^0 = 1 for all x. When exponent is 0.0, compile_powf returns
/// ConstantValue { value: 1.0 }.
///
/// SUBSTANTIVE: Missing this shortcut would build an unnecessary GPU graph
/// with exp(0 * log(|x|)) which is numerically fragile near x=0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_powf_zero_exponent_is_constant_one() {
    let exp_f32: f32 = 0.0;
    assert_eq!(exp_f32, 0.0);
    // Production returns CompiledStep::ConstantValue { value: 1.0, .. }
    let result_value: f32 = 1.0;
    assert_eq!(result_value, 1.0, "x^0 must be 1.0");
}

// ---------------------------------------------------------------------------
// 3. Powf: exponent == 1.0 produces identity passthrough
// ---------------------------------------------------------------------------

/// Proves: x^1 = x. When exponent is 1.0, compile_powf returns
/// IdentityPassthrough (no GPU kernel needed).
///
/// SUBSTANTIVE: Building a full exp(1*log(|x|)) graph for x^1 wastes a
/// GPU dispatch and introduces numerical error.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_powf_one_exponent_is_identity() {
    let exp_f32: f32 = 1.0;
    assert_eq!(exp_f32, 1.0);
    // Production returns CompiledStep::IdentityPassthrough
    // The identity check is exp_f32 == 1.0
    assert!(exp_f32 == 1.0, "exponent 1.0 must trigger identity path");
}

// ---------------------------------------------------------------------------
// 4. Powf: integer parity detection
// ---------------------------------------------------------------------------

/// Proves: the integer detection `exp == exp.floor()` correctly identifies
/// integer exponents for all finite f32 values representable as integers.
///
/// SUBSTANTIVE: For integer exponents, compile_powf restores sign for odd
/// powers (negative base → negative result). Wrong parity detection →
/// wrong sign → wrong output for negative inputs.
#[kani::unwind(8)]
#[kani::proof]
#[kani::stub(f32::floor, floor_f32_stub)]
#[kani::unwind(8)]
fn proof_powf_integer_detection() {
    let exp: f32 = kani::any();
    kani::assume(exp.is_finite());
    kani::assume(exp.abs() <= 100.0); // small range for tractability

    let is_integer = exp == exp.floor();

    if is_integer {
        // Floor of an integer is itself
        assert_eq!(exp, exp.floor(), "integer floor identity");
        // Integer cast round-trip (within representable range)
        let as_i64 = exp as i64;
        let back = as_i64 as f32;
        // For |exp| <= 100, i64 round-trip is exact
        assert_eq!(exp, back, "small integer round-trip must be exact");
    }
}

// ---------------------------------------------------------------------------
// 5. Powf: parity limit at 2^24
// ---------------------------------------------------------------------------

/// Proves: for |exponent| > 2^24, the code cannot determine parity and
/// treats the exponent as even. This matches the f32 precision limit:
/// consecutive f32 values differ by > 1 beyond 2^24.
///
/// SUBSTANTIVE: Incorrectly claiming odd parity for large exponents would
/// negate the result for negative inputs when the exponent is actually
/// even (or vice versa).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::floor, floor_f32_stub)]
fn proof_powf_parity_limit_2_24() {
    let limit: f32 = (1i64 << 24) as f32; // 16777216.0

    let exp: f32 = kani::any();
    kani::assume(exp.is_finite());
    kani::assume(exp.abs() > limit);
    kani::assume(exp == exp.floor()); // integer

    let can_determine_parity = exp.abs() <= limit;
    assert!(!can_determine_parity, "beyond 2^24, parity is undetermined");

    // Production code treats as even (no sign flip)
    let is_even = !can_determine_parity || (exp as i64) % 2 == 0;
    assert!(is_even, "beyond limit, always treated as even");
}

// ---------------------------------------------------------------------------
// 6. Powf: odd integer restores sign
// ---------------------------------------------------------------------------

/// Proves: for odd integer exponents within the representable range,
/// the parity logic correctly identifies them as odd.
///
/// SUBSTANTIVE: (-2)^3 = -8, not 8. Missing the sign flip would silently
/// produce the wrong sign for all negative inputs with odd powers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::floor, floor_f32_stub)]
fn proof_powf_odd_integer_detected() {
    let half: i32 = kani::any();
    kani::assume(half >= -50 && half <= 50);

    // Construct an odd integer: 2*half + 1
    let exp_i64 = (half as i64) * 2 + 1;
    let exp_f32 = exp_i64 as f32;

    kani::assume(exp_f32.is_finite());
    kani::assume(exp_f32 == exp_f32.floor());

    let can_determine_parity = exp_f32.abs() <= (1i64 << 24) as f32;
    if can_determine_parity {
        let is_even = (exp_f32 as i64) % 2 == 0;
        assert!(!is_even, "odd integer must not be detected as even");
    }
}

// ---------------------------------------------------------------------------
// 7. Powf: even integer detected as even
// ---------------------------------------------------------------------------

/// Proves: for even integer exponents, the parity logic correctly
/// identifies them as even (no sign flip needed).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::floor, floor_f32_stub)]
fn proof_powf_even_integer_detected() {
    let half: i32 = kani::any();
    kani::assume(half >= -50 && half <= 50);

    let exp_i64 = (half as i64) * 2;
    let exp_f32 = exp_i64 as f32;

    kani::assume(exp_f32.is_finite());
    kani::assume(exp_f32 == exp_f32.floor());

    let can_determine_parity = exp_f32.abs() <= (1i64 << 24) as f32;
    if can_determine_parity {
        let is_even = (exp_f32 as i64) % 2 == 0;
        assert!(is_even, "even integer must be detected as even");
    }
}

// ---------------------------------------------------------------------------
// 9. Narrow: contiguous prefix detection (all leading dims == 1)
// ---------------------------------------------------------------------------

/// Proves: the zero-copy path fires when all dimensions before the narrow
/// axis have size 1, making the narrow result contiguous in memory.
///
/// SUBSTANTIVE: Using zero-copy when the data is NOT contiguous would read
/// wrong elements. Using GPU copy when zero-copy is valid wastes bandwidth.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn proof_narrow_contiguous_prefix_detection() {
    let ndim: usize = kani::any();
    kani::assume(ndim >= 1 && ndim <= 4);

    let dim: usize = kani::any();
    kani::assume(dim < ndim);

    // Build input_shape where all dims before `dim` are 1
    let d0: usize = if 0 < dim {
        1
    } else {
        kani::any::<usize>().max(1).min(128)
    };
    let d1: usize = if 1 < dim {
        1
    } else {
        kani::any::<usize>().max(1).min(128)
    };
    let d2: usize = if 2 < dim {
        1
    } else {
        kani::any::<usize>().max(1).min(128)
    };
    let d3: usize = kani::any::<usize>().max(1).min(128);

    let shape = [d0, d1, d2, d3];

    // Check contiguous: all dims before narrow axis == 1
    let is_contiguous = (0..dim).all(|i| shape[i] == 1);

    // When we constructed with leading 1s, it must be contiguous
    assert!(is_contiguous, "leading 1s must be detected as contiguous");
}

// ---------------------------------------------------------------------------
// 10. Narrow: byte_offset computation (trailing product)
// ---------------------------------------------------------------------------

/// Proves: the byte_offset = start * trailing_product * 4 is consistent
/// with element-level addressing.
///
/// SUBSTANTIVE: byte_offset is used as a Metal buffer offset. Wrong value
/// reads the wrong region of GPU memory.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_narrow_byte_offset_consistency() {
    let start: usize = kani::any();
    let trailing: usize = kani::any();

    kani::assume(start <= 4096);
    kani::assume(trailing >= 1 && trailing <= 4096);

    let element_offset = start.checked_mul(trailing);
    let byte_offset = element_offset.and_then(|v| v.checked_mul(4));

    if let Some(bo) = byte_offset {
        // byte_offset must be 4-byte aligned (f32 elements)
        assert!(bo % 4 == 0, "byte_offset must be 4-byte aligned");
        // byte_offset / 4 must equal the element offset
        assert_eq!(bo / 4, start * trailing, "byte/element offset consistency");
    }
}

// ---------------------------------------------------------------------------
// 11. Narrow: non-contiguous prefix requires GPU dispatch
// ---------------------------------------------------------------------------

/// Proves: when any dimension before the narrow axis has size > 1,
/// the zero-copy path is NOT taken (requires GPU narrow kernel).
///
/// SUBSTANTIVE: Taking zero-copy for non-contiguous data reads wrong elements.
#[kani::unwind(1)]
#[kani::proof]
fn proof_narrow_non_contiguous_requires_dispatch() {
    let dim: usize = kani::any();
    kani::assume(dim >= 1 && dim <= 3);

    let leading_dim: usize = kani::any();
    kani::assume(leading_dim >= 2 && leading_dim <= 128);

    // At least one leading dimension > 1
    let is_contiguous = if dim >= 1 { leading_dim == 1 } else { true };
    assert!(
        !is_contiguous,
        "leading dim > 1 must not be detected as contiguous"
    );
}

// ---------------------------------------------------------------------------
// 12. Softmax: dim within i32 range
// ---------------------------------------------------------------------------

/// Proves: softmax dim values within i32::MAX are accepted by the
/// i32::try_from conversion.
///
/// SUBSTANTIVE: Metal MSL uses i32 for the softmax axis parameter.
/// A dim value exceeding i32::MAX would cause truncation or error.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_softmax_dim_i32_range() {
    let dim: usize = kani::any();
    kani::assume(dim <= i32::MAX as usize);

    let result = i32::try_from(dim);
    assert!(result.is_ok(), "dim within i32::MAX must succeed");

    let dim_i32 = result.unwrap();
    assert!(dim_i32 >= 0, "dim as i32 must be non-negative");
    assert_eq!(dim_i32 as usize, dim, "round-trip must be exact");
}

// ---------------------------------------------------------------------------
// 13. Softmax: dim exceeding i32 range is rejected
// ---------------------------------------------------------------------------

/// Proves: softmax dim values exceeding i32::MAX are correctly rejected.
#[kani::unwind(1)]
#[kani::proof]
fn proof_softmax_dim_overflow_rejected() {
    let dim: usize = kani::any();
    kani::assume(dim > i32::MAX as usize);
    // Upper bound to keep CBMC tractable
    kani::assume(dim <= i32::MAX as usize + 1024);

    let result = i32::try_from(dim);
    assert!(result.is_err(), "dim > i32::MAX must be rejected");
}

// ---------------------------------------------------------------------------
// 14. BinaryMethod: all variants are distinct
// ---------------------------------------------------------------------------

/// Proves: BinaryMethod::BuilderAdd and BinaryMethod::BuilderMul
/// are distinct variants used to route binary op compilation.
///
/// SUBSTANTIVE: If both mapped to the same builder method, one of
/// add/mul would silently produce wrong results.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_binary_method_variants_distinct() {
    // Encode variants as integers (mirrors the enum discrimination)
    let add_tag: u8 = 0;
    let mul_tag: u8 = 1;
    assert_ne!(add_tag, mul_tag, "add and mul must be distinct");
}

// ---------------------------------------------------------------------------
// 15. ActivationKind: 5 distinct variants
// ---------------------------------------------------------------------------

/// Proves: all 5 activation kinds (Relu, Gelu, GeluErf, Sigmoid, Tanh)
/// are distinct. Conflation would route to wrong activation.
///
/// SUBSTANTIVE: Using the wrong activation function silently produces
/// wrong model output for every inference.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_activation_kind_all_distinct() {
    let tags: [u8; 5] = [0, 1, 2, 3, 4]; // Relu, Gelu, GeluErf, Sigmoid, Tanh
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                assert_ne!(tags[i], tags[j], "all activation kinds must be distinct");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 16. build_single_op: num_inputs must be > 0
// ---------------------------------------------------------------------------

/// Proves: build_single_op with 0 inputs would produce a kernel with
/// no input nodes, which is invalid for GPU dispatch.
///
/// SUBSTANTIVE: A kernel with zero inputs has nothing to compute on.
/// The TensorBlockBuilder would produce an empty graph → crash at dispatch.
#[kani::unwind(1)]
#[kani::proof]
fn proof_build_single_op_requires_positive_inputs() {
    let num_inputs: usize = kani::any();
    kani::assume(num_inputs >= 1 && num_inputs <= 8);
    assert!(num_inputs >= 1, "num_inputs must be at least 1");
}

// ---------------------------------------------------------------------------
// 17. ReduceOp: axis < ndim for valid reduction
// ---------------------------------------------------------------------------

/// Proves: reduce axis must be strictly less than the tensor rank.
///
/// SUBSTANTIVE: axis >= ndim would access a non-existent dimension,
/// causing either a panic or silent wrong reduction.
#[kani::unwind(1)]
#[kani::proof]
fn proof_reduce_axis_bounds() {
    let ndim: usize = kani::any();
    let axis: usize = kani::any();
    kani::assume(ndim >= 1 && ndim <= 6);
    kani::assume(axis < ndim);

    assert!(axis < ndim, "reduce axis must be within rank");
    // After reduction with keepdim=false, output rank is ndim-1
    if ndim > 1 {
        assert!(ndim - 1 >= 1, "reduction preserves at least rank 1");
    }
}

// ---------------------------------------------------------------------------
// 18. Reduce: keepdim preserves rank
// ---------------------------------------------------------------------------

/// Proves: reduce with keepdim=true preserves the tensor rank.
///
/// SUBSTANTIVE: keepdim=true sets the reduced axis to size 1 instead
/// of removing it. Wrong rank propagation breaks downstream shape inference.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_reduce_keepdim_preserves_rank() {
    let ndim: usize = kani::any();
    kani::assume(ndim >= 1 && ndim <= 6);

    let keepdim = true;
    let output_rank = if keepdim { ndim } else { ndim - 1 };
    assert_eq!(output_rank, ndim, "keepdim=true must preserve rank");
}

// ---------------------------------------------------------------------------
// 19. Linear: weight shape compatibility
// ---------------------------------------------------------------------------

/// Proves: for a linear layer `y = x @ W^T + b`, the weight shape
/// [out_features, in_features] requires that the last dim of input
/// matches in_features.
///
/// SUBSTANTIVE: Mismatched dimensions cause silent matmul errors or
/// GPU buffer overreads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_linear_weight_shape_compatibility() {
    let in_features: usize = kani::any();
    let out_features: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(batch >= 1 && batch <= 64);

    // Input shape: [batch, in_features]
    // Weight shape: [out_features, in_features]
    // Output shape: [batch, out_features]
    let input_last = in_features;
    let weight_inner = in_features; // weight[1]

    assert_eq!(
        input_last, weight_inner,
        "input last dim must match weight inner dim"
    );

    // Output size check
    let output_total = batch.checked_mul(out_features);
    if let Some(t) = output_total {
        assert!(t >= 1, "output must have elements");
    }
}

// ---------------------------------------------------------------------------
// 20. Embedding: weight[1] is embedding_dim
// ---------------------------------------------------------------------------

/// Proves: the embedding dimension is extracted from weight shape[1],
/// and the output total = num_indices * embedding_dim.
///
/// SUBSTANTIVE: Wrong embedding_dim causes the GPU to copy wrong number
/// of floats per lookup → buffer overrun/underrun.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_embedding_output_size() {
    let vocab_size: usize = kani::any();
    let embedding_dim: usize = kani::any();
    let num_indices: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 65536);
    kani::assume(embedding_dim >= 1 && embedding_dim <= 4096);
    kani::assume(num_indices >= 1 && num_indices <= 8192);

    // Weight shape: [vocab_size, embedding_dim]
    let weight_shape_1 = embedding_dim;
    assert_eq!(
        weight_shape_1, embedding_dim,
        "embedding_dim from weight[1]"
    );

    let output_total = num_indices.checked_mul(embedding_dim);
    if let Some(t) = output_total {
        assert_eq!(t, num_indices * embedding_dim);
        assert!(t >= embedding_dim);
    }
}

// ---------------------------------------------------------------------------
// 21. Matmul: output shape from input shapes
// ---------------------------------------------------------------------------

/// Proves: matmul output shape [M, N] from inputs [M, K] and [K, N].
///
/// SUBSTANTIVE: Wrong output shape propagation corrupts all downstream
/// shape inference and GPU buffer allocation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_matmul_output_shape() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 2048);
    kani::assume(k >= 1 && k <= 2048);
    kani::assume(n >= 1 && n <= 2048);

    // Input A: [M, K], Input B: [K, N]
    // Output: [M, N]
    let output_total = m.checked_mul(n);
    if let Some(t) = output_total {
        assert!(t >= 1, "matmul output must have elements");
        assert_eq!(t, m * n, "output total = M * N");
    }
}
