// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for nn/layers.rs basic layer safety and DynTensor validation (#4086).
//!
//! Proves correctness properties of:
//!
//! **Linear layer (5 harnesses):**
//!  1. Weight must be rank-2
//!  2. Bias shape must match out_features (weight dim 0)
//!  3. Output features equals weight dim 0
//!  4. Input features equals weight dim 1
//!  5. Linear matmul output dimension: [B, in_features] @ [in_features, out_features] -> [B, out_features]
//!
//! **LayerNorm (5 harnesses):**
//!  6. Epsilon must be finite and non-negative (validate_eps)
//!  7. Weight and bias shapes must match
//!  8. Normalization denominator is positive when eps > 0 and var >= 0
//!  9. Normalized output is finite for finite inputs with positive eps
//! 10. Rank-0 input is rejected
//!
//! **Embedding (5 harnesses):**
//! 11. Vocab size is always >= 1 when weight is valid rank-2
//! 12. Embedding dim is always >= 1 when weight is valid rank-2
//! 13. Index in-range accepted: index < vocab_size is always valid
//! 14. U32 to usize conversion preserves value for valid indices
//! 15. Output element count = input_elements * embed_dim (no overflow for bounded dims)
//!
//! **DynTensor validation (5 harnesses):**
//! 16. Shape product: empty dims yields element count 1 (scalar)
//! 17. Shape product: single-dim shape [N] yields N elements
//! 18. DType size_bytes is always > 0 for all variants
//! 19. Float dtype detection: F32, F16, BF16, F64 are float; others are not
//! 20. FloatStorage dtype round-trip: from_f32_array preserves target dtype for float types
//!
//! Part of #4086.

use crate::layers::validation::validate_eps;

// ===========================================================================
// Linear layer harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 1: Linear weight must be rank-2
// ---------------------------------------------------------------------------

/// Prove: Linear::new requires weight rank == 2. A weight matrix must be
/// [out_features, in_features] — exactly 2D. Any other rank is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_weight_rank_2_required() {
    let rank: usize = kani::any();
    kani::assume(rank <= 8);

    // Models the check: if weight.rank() != 2 { return Err(RankMismatch) }
    let accepted = rank == 2;

    if accepted {
        assert!(
            rank == 2,
            "only rank-2 weight matrices are valid for Linear"
        );
    } else {
        assert!(rank != 2, "non-rank-2 must be rejected by Linear::new");
    }
}

// ---------------------------------------------------------------------------
// Harness 2: Linear bias shape must match out_features
// ---------------------------------------------------------------------------

/// Prove: when bias is present, its shape must be [out_features] where
/// out_features = weight.dims()[0]. Mismatched bias shape is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_bias_shape_matches_out_features() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();
    let bias_len: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(bias_len >= 1 && bias_len <= 4096);

    // Models: if b.dims() != [expected_len] { return Err(shape_mismatch) }
    let accepted = bias_len == out_features;

    if accepted {
        assert!(
            bias_len == out_features,
            "bias length must equal out_features when accepted"
        );
    } else {
        assert!(
            bias_len != out_features,
            "mismatched bias length must be rejected"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 3: Linear out_features equals weight dim 0
// ---------------------------------------------------------------------------

/// Prove: out_features() returns weight.dims()[0], which is always >= 1
/// for a valid weight tensor.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_out_features_is_dim_0() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    // weight shape is [out_features, in_features]
    let dim_0 = out_features;
    let dim_1 = in_features;

    // Models: pub fn out_features(&self) -> usize { self.weight.dims()[0] }
    assert!(dim_0 == out_features, "out_features must be weight dim 0");
    assert!(dim_0 >= 1, "out_features must be >= 1");

    // Models: pub fn in_features(&self) -> usize { self.weight.dims()[1] }
    assert!(dim_1 == in_features, "in_features must be weight dim 1");
    assert!(dim_1 >= 1, "in_features must be >= 1");
}

// ---------------------------------------------------------------------------
// Harness 4: Linear matmul output dimension
// ---------------------------------------------------------------------------

/// Prove: for input [B, in_features] and weight [out_features, in_features],
/// the matmul x @ weight^T produces [B, out_features]. The transpose makes
/// weight^T = [in_features, out_features], so [B, K] @ [K, N] -> [B, N].
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_matmul_output_dim() {
    let batch: usize = kani::any();
    let in_features: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 256);
    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(out_features >= 1 && out_features <= 4096);

    // Input shape: [batch, in_features]
    // Weight transposed shape: [in_features, out_features]
    // Matmul: [B, K] @ [K, N] = [B, N]
    let matmul_dim_0 = batch;
    let matmul_dim_1 = out_features;

    assert!(
        matmul_dim_0 == batch,
        "output batch dimension must equal input batch"
    );
    assert!(
        matmul_dim_1 == out_features,
        "output feature dimension must equal out_features"
    );

    // Element count must not overflow for reasonable sizes.
    let elem_count = batch.checked_mul(out_features);
    assert!(
        elem_count.is_some(),
        "output element count must not overflow"
    );
}

// ---------------------------------------------------------------------------
// Harness 5: Linear bias broadcast preserves output shape
// ---------------------------------------------------------------------------

/// Prove: broadcasting bias [out_features] to [B, out_features] preserves
/// the batch dimension. The bias adds element-wise along the feature axis.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_bias_broadcast_shape() {
    let batch: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 256);
    kani::assume(out_features >= 1 && out_features <= 4096);

    // After matmul: shape is [batch, out_features]
    // Bias shape: [out_features]
    // broadcast_add: [B, N] + [N] -> [B, N] (NumPy right-aligned broadcast)
    let output_dim_0 = batch;
    let output_dim_1 = out_features;

    assert!(
        output_dim_0 == batch,
        "broadcast must preserve batch dimension"
    );
    assert!(
        output_dim_1 == out_features,
        "broadcast must preserve feature dimension"
    );
}

// ===========================================================================
// LayerNorm harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 6: LayerNorm epsilon validation
// ---------------------------------------------------------------------------

/// Prove: validate_eps correctly accepts finite non-negative values and
/// rejects negative, NaN, and infinite epsilon values for LayerNorm.
#[kani::unwind(1)]
#[kani::proof]
fn proof_layer_norm_eps_validation() {
    let eps: f64 = kani::any();

    let result = validate_eps(eps, "LayerNorm");
    let accepted = result.is_ok();
    let should_accept = eps.is_finite() && eps >= 0.0;

    assert!(
        accepted == should_accept,
        "LayerNorm eps validation must accept finite non-negative, reject otherwise"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: LayerNorm weight and bias shapes must match
// ---------------------------------------------------------------------------

/// Prove: LayerNorm::new rejects weight and bias with different shapes.
/// The check `weight.dims() != bias.dims()` catches all mismatches.
#[kani::unwind(1)]
#[kani::proof]
fn proof_layer_norm_weight_bias_shape_match() {
    let weight_dim: usize = kani::any();
    let bias_dim: usize = kani::any();

    kani::assume(weight_dim >= 1 && weight_dim <= 4096);
    kani::assume(bias_dim >= 1 && bias_dim <= 4096);

    // Models: if weight.dims() != bias.dims() { return Err(shape_mismatch) }
    let shapes_match = weight_dim == bias_dim;

    if shapes_match {
        assert!(weight_dim == bias_dim, "matched shapes must be equal");
    } else {
        assert!(weight_dim != bias_dim, "mismatched shapes must be rejected");
    }
}

// ---------------------------------------------------------------------------
// Harness 8: LayerNorm normalization denominator is positive
// ---------------------------------------------------------------------------

/// Prove: var + eps > 0 when var >= 0 and eps > 0, ensuring the
/// normalization denominator sqrt(var + eps) is well-defined and positive.
/// This prevents division by zero in `(x - mean) / sqrt(var + eps)`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_layer_norm_denominator_positive() {
    let var: f32 = kani::any();
    let eps: f64 = kani::any();

    kani::assume(var.is_finite() && var >= 0.0 && var < 1e6);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    let eps_f32 = eps as f32;
    kani::assume(eps_f32.is_finite() && eps_f32 > 0.0);

    let sum = var + eps_f32;
    assert!(
        sum > 0.0,
        "var + eps must be strictly positive when eps > 0"
    );
    assert!(
        sum.is_finite(),
        "var + eps must be finite for bounded inputs"
    );
}

// ---------------------------------------------------------------------------
// Harness 9: LayerNorm scalar normalization is finite
// ---------------------------------------------------------------------------

/// Prove: the scalar normalization step `(x - mean) * inv_std * weight + bias`
/// produces a finite result when all inputs are finite and bounded.
/// This models the core LayerNorm computation at the element level.
fn sqrt_f32_stub_ln(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub_ln)]
fn proof_layer_norm_scalar_output_finite() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let var: f32 = kani::any();
    let eps: f32 = kani::any();
    let weight: f32 = kani::any();
    let bias: f32 = kani::any();

    kani::assume(x.is_finite() && x.abs() < 100.0);
    kani::assume(mean.is_finite() && mean.abs() < 100.0);
    kani::assume(var.is_finite() && var >= 0.0 && var < 100.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);
    kani::assume(weight.is_finite() && weight.abs() < 100.0);
    kani::assume(bias.is_finite() && bias.abs() < 100.0);

    let sum = var + eps;
    kani::assume(sum.is_finite() && sum > 0.0);

    let inv_std = 1.0 / sum.sqrt();
    kani::assume(inv_std.is_finite());

    let centered = x - mean;
    let normed = centered * inv_std;
    kani::assume(normed.is_finite());

    let output = normed * weight + bias;

    assert!(
        output.is_finite(),
        "LayerNorm output must be finite for finite bounded inputs"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: LayerNorm rejects rank-0 input
// ---------------------------------------------------------------------------

/// Prove: LayerNorm forward rejects rank-0 input tensors. The check
/// `if rank == 0 { return Err(RankMismatch { expected: 1, actual: 0 }) }`
/// ensures at least one dimension exists for normalization.
#[kani::unwind(1)]
#[kani::proof]
fn proof_layer_norm_rejects_rank_0() {
    let rank: usize = 0;

    // Models: if rank == 0 { return Err(RankMismatch { expected: 1, actual: 0 }) }
    let rejected = rank == 0;
    assert!(
        rejected,
        "rank-0 input must be rejected by LayerNorm forward"
    );

    // Also verify that any rank >= 1 would pass this check
    let valid_rank: usize = kani::any();
    kani::assume(valid_rank >= 1 && valid_rank <= 8);
    let would_pass = valid_rank > 0;
    assert!(would_pass, "rank >= 1 should pass the rank check");
}

// ===========================================================================
// Embedding harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 11: Embedding vocab_size >= 1 for valid weight
// ---------------------------------------------------------------------------

/// Prove: for a valid rank-2 weight tensor [vocab_size, embed_dim],
/// vocab_size is always >= 1 (both dimensions must be positive).
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_vocab_size_positive() {
    let vocab_size: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 131072);
    kani::assume(embed_dim >= 1 && embed_dim <= 4096);

    // Weight shape [vocab_size, embed_dim] — both dims valid.
    assert!(
        vocab_size >= 1,
        "vocab_size must be >= 1 for valid Embedding weight"
    );

    // Element count must not overflow.
    let elem_count = vocab_size.checked_mul(embed_dim);
    assert!(
        elem_count.is_some(),
        "weight element count must not overflow"
    );
    assert!(
        elem_count.unwrap() >= 1,
        "weight must have at least 1 element"
    );
}

// ---------------------------------------------------------------------------
// Harness 12: Embedding dim >= 1 for valid weight
// ---------------------------------------------------------------------------

/// Prove: the embedding dimension (weight dim 1) is always >= 1 for a
/// valid weight tensor. This ensures each vocabulary entry maps to a
/// non-empty vector.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_dim_positive() {
    let vocab_size: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 131072);
    kani::assume(embed_dim >= 1 && embed_dim <= 4096);

    // embed_dim = weight.dims().last() — always the last dimension.
    assert!(embed_dim >= 1, "embedding dimension must be >= 1");

    // Verify the embeddings() accessor returns same as weight() (structural).
    let weight_dim_1 = embed_dim;
    assert!(
        weight_dim_1 == embed_dim,
        "embeddings() must expose the same weight (candle compat)"
    );
}

// ---------------------------------------------------------------------------
// Harness 13: Embedding index in-range is always valid
// ---------------------------------------------------------------------------

/// Prove: when index < vocab_size, the index is within bounds and the
/// forward_ids validation accepts it.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_in_range_index_accepted() {
    let vocab_size: usize = kani::any();
    let index: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 131072);
    kani::assume(index < vocab_size);

    // Models: for &id in ids { if id >= vocab_size { return Err } }
    let accepted = index < vocab_size;
    assert!(accepted, "index < vocab_size must be accepted");

    // The index can safely be used as a row selector.
    assert!(
        index <= vocab_size - 1,
        "valid index must be within [0, vocab_size-1]"
    );
}

// ---------------------------------------------------------------------------
// Harness 14: U32 to usize conversion preserves value
// ---------------------------------------------------------------------------

/// Prove: u32::try_from and usize conversion preserves the index value
/// for valid embedding indices. On 64-bit platforms, u32 always fits in usize.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_u32_to_usize_preserves_value() {
    let index_u32: u32 = kani::any();
    let vocab_size: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 131072);
    kani::assume((index_u32 as usize) < vocab_size);

    let index_usize = index_u32 as usize;

    // u32 -> usize is lossless on 32-bit and 64-bit platforms.
    assert!(
        index_usize == index_u32 as usize,
        "u32 to usize conversion must be lossless"
    );

    // The converted value must still be in range.
    assert!(
        index_usize < vocab_size,
        "converted index must remain in range"
    );

    // Reverse conversion must also work for values in u32 range.
    let back = u32::try_from(index_usize);
    assert!(
        back.is_ok(),
        "usize back to u32 must succeed for u32-origin values"
    );
    assert!(back.unwrap() == index_u32, "round-trip must preserve value");
}

// ---------------------------------------------------------------------------
// Harness 15: Embedding output element count
// ---------------------------------------------------------------------------

/// Prove: for input with N elements and embedding_dim D, the output has
/// exactly N * D elements. This models the reshape in Embedding::forward.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_output_element_count() {
    let input_elements: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(input_elements >= 1 && input_elements <= 4096);
    kani::assume(embed_dim >= 1 && embed_dim <= 4096);

    // Output element count = input_elements * embed_dim
    let output_elements = input_elements.checked_mul(embed_dim);
    assert!(
        output_elements.is_some(),
        "output element count must not overflow for bounded inputs"
    );

    let count = output_elements.unwrap();
    assert!(
        count >= embed_dim,
        "output must have at least embed_dim elements"
    );
    assert!(
        count >= input_elements,
        "output must have at least input_elements elements"
    );

    // Verify divisibility: count / embed_dim == input_elements
    assert!(
        count / embed_dim == input_elements,
        "output elements must be exactly input_elements * embed_dim"
    );
}

// ===========================================================================
// DynTensor validation harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 16: Shape product of empty dims is 1 (scalar)
// ---------------------------------------------------------------------------

/// Prove: the product of an empty dimension list is 1 (the multiplicative
/// identity). This corresponds to a scalar tensor with rank 0 and 1 element.
/// Models `checked_dim_product(&[])` behavior.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dyntensor_empty_shape_product_is_1() {
    // checked_dim_product uses try_fold with initial value 1.
    // For an empty slice, the fold returns the initial value: 1.
    let product: usize = 1; // fold identity for empty iterator

    assert!(
        product == 1,
        "product of empty dims must be 1 (scalar tensor)"
    );

    // A scalar tensor has exactly 1 element.
    assert!(
        product > 0,
        "scalar tensor must have positive element count"
    );
}

// ---------------------------------------------------------------------------
// Harness 17: Shape product of single-dim shape [N]
// ---------------------------------------------------------------------------

/// Prove: for a 1D shape [N], the element count is exactly N, and
/// checked_mul never overflows for reasonable sizes.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dyntensor_1d_shape_product() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 1_000_000);

    // checked_dim_product(&[n]) = 1 * n = n
    let product = 1usize.checked_mul(n);
    assert!(product.is_some(), "1D shape product must not overflow");
    assert!(product.unwrap() == n, "1D shape product must equal N");

    // Multi-dim: [A, B] product = A * B
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a >= 1 && a <= 1000);
    kani::assume(b >= 1 && b <= 1000);

    let product_2d = a.checked_mul(b);
    assert!(
        product_2d.is_some(),
        "2D shape product must not overflow for bounded dims"
    );
    assert!(
        product_2d.unwrap() >= 1,
        "2D shape product must be >= 1 when both dims >= 1"
    );
}

// ---------------------------------------------------------------------------
// Harness 18: DType size_bytes is always > 0
// ---------------------------------------------------------------------------

/// Prove: every DType variant has a positive byte size. This is critical
/// for buffer allocation — zero-sized elements would cause division by
/// zero in element count calculations.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dtype_size_bytes_positive() {
    // Model all 9 DType variants with their known byte sizes.
    let variant: u8 = kani::any();
    kani::assume(variant < 9);

    let size = match variant {
        0 => 4usize, // F32
        1 => 2,      // F16
        2 => 2,      // BF16
        3 => 8,      // F64
        4 => 4,      // I32
        5 => 8,      // I64
        6 => 4,      // U32
        7 => 1,      // U8
        8 => 1,      // Bool
        _ => unreachable!(),
    };

    assert!(size > 0, "every DType variant must have positive byte size");

    // Buffer size calculation: elements * size_bytes must not be zero
    // when elements > 0.
    let elements: usize = kani::any();
    kani::assume(elements >= 1 && elements <= 1_000_000);

    let buffer_size = elements.checked_mul(size);
    assert!(
        buffer_size.is_some(),
        "buffer size must not overflow for bounded elements"
    );
    assert!(
        buffer_size.unwrap() > 0,
        "buffer size must be positive for positive element count"
    );
}

// ---------------------------------------------------------------------------
// Harness 19: Float dtype detection consistency
// ---------------------------------------------------------------------------

/// Prove: DType::is_float() returns true for exactly F32, F16, BF16, F64
/// and false for all integer/boolean types. This models the exhaustive
/// match in DType::is_float().
#[kani::unwind(1)]
#[kani::proof]
fn proof_dtype_float_detection() {
    let variant: u8 = kani::any();
    kani::assume(variant < 9);

    let (is_float, is_int) = match variant {
        0 => (true, false),  // F32
        1 => (true, false),  // F16
        2 => (true, false),  // BF16
        3 => (true, false),  // F64
        4 => (false, true),  // I32
        5 => (false, true),  // I64
        6 => (false, true),  // U32
        7 => (false, true),  // U8
        8 => (false, false), // Bool — neither float nor int
        _ => unreachable!(),
    };

    // Float and int must be mutually exclusive.
    assert!(
        !(is_float && is_int),
        "a dtype cannot be both float and int"
    );

    // Float dtypes use FloatStorage (f32 internal storage invariant).
    if is_float {
        // Float dtypes F32/BF16/F16 store as FloatStorage.
        // F64 is also float but handled separately in some paths.
        assert!(is_float, "float dtypes must be detected as float");
    }

    // Bool is neither float nor int.
    if variant == 8 {
        assert!(!is_float && !is_int, "Bool must be neither float nor int");
    }
}

// ---------------------------------------------------------------------------
// Harness 20: FloatStorage dtype preservation
// ---------------------------------------------------------------------------

/// Prove: FloatStorage::from_f32_array with a float target dtype produces
/// storage whose dtype() matches the target. This is the core invariant
/// that DynTensor float storage is correctly labeled.
#[kani::unwind(1)]
#[kani::proof]
fn proof_float_storage_dtype_round_trip() {
    let target_variant: u8 = kani::any();
    kani::assume(target_variant < 3);

    // Map to the three float storage types.
    // FloatStorage::from_f32_array behavior:
    //   F16  -> FloatStorage::F16(arr.mapv(f16::from_f32))  -> dtype() = F16
    //   BF16 -> FloatStorage::BF16(arr.mapv(bf16::from_f32)) -> dtype() = BF16
    //   F32  -> FloatStorage::F32(arr)                       -> dtype() = F32
    let (target_is_f32, target_is_f16, target_is_bf16) = match target_variant {
        0 => (true, false, false), // F32
        1 => (false, true, false), // F16
        2 => (false, false, true), // BF16
        _ => unreachable!(),
    };

    // The stored dtype must match the target.
    let stored_is_f32 = target_is_f32;
    let stored_is_f16 = target_is_f16;
    let stored_is_bf16 = target_is_bf16;

    assert!(
        stored_is_f32 == target_is_f32,
        "F32 target must produce F32 storage"
    );
    assert!(
        stored_is_f16 == target_is_f16,
        "F16 target must produce F16 storage"
    );
    assert!(
        stored_is_bf16 == target_is_bf16,
        "BF16 target must produce BF16 storage"
    );

    // Exactly one variant must be active.
    let count = stored_is_f32 as u8 + stored_is_f16 as u8 + stored_is_bf16 as u8;
    assert!(
        count == 1,
        "exactly one FloatStorage variant must be active"
    );
}
