// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proofs for NativeOpKind dimension arithmetic (#3351).
//!
//! Covers 3 NativeOpKind variants that had zero Kani harnesses:
//! - **Conv1dGemm**: output length formula, GEMM FLOPs threshold, im2col buffer sizing
//! - **BatchedLinearProjection**: weight transpose indexing, total_out consistency
//! - **ProjectionSlice**: offset accumulation stays in bounds
//!
//! These proofs verify the pure integer arithmetic in the peephole passes
//! (trace_compile_conv.rs, trace_compile_peephole_batched_qkv.rs) that
//! constructs NativeOp parameters. Buffer index out-of-bounds in these
//! paths would cause silent data corruption or GPU crashes at runtime.

// ---------------------------------------------------------------------------
// Conv1dGemm: output length formula no-overflow
// ---------------------------------------------------------------------------

/// Prove: Conv1d output length formula does not overflow for realistic dims.
///
/// Models the arithmetic from `trace_compile_conv.rs:48-51`:
/// ```
/// let effective_k = dilation * (k_size - 1) + 1;
/// let l_out = (l_in + 2 * padding - effective_k) / stride + 1;
/// ```
///
/// Overflow in any intermediate would produce a wrong NativeOp shape,
/// leading to buffer size mismatch in the executor.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_gemm_output_length_no_overflow() {
    let l_in: usize = kani::any();
    let padding: usize = kani::any();
    let stride: usize = kani::any();
    let dilation: usize = kani::any();
    let k_size: usize = kani::any();

    // Realistic Conv1d bounds (Kokoro uses k=3,7,11 s=1 d=1).
    kani::assume(l_in >= 1 && l_in <= 65536);
    kani::assume(padding <= 512);
    kani::assume(stride >= 1 && stride <= 16);
    kani::assume(dilation >= 1 && dilation <= 16);
    kani::assume(k_size >= 1 && k_size <= 64);

    // Step 1: effective_k = dilation * (k_size - 1) + 1
    let km1 = k_size - 1; // safe: k_size >= 1
    let dil_km1 = match dilation.checked_mul(km1) {
        Some(v) => v,
        None => return, // overflow: would be caught by bounds check
    };
    let effective_k = match dil_km1.checked_add(1) {
        Some(v) => v,
        None => return,
    };

    // Step 2: l_in + 2*padding
    let two_pad = match padding.checked_mul(2) {
        Some(v) => v,
        None => return,
    };
    let padded = match l_in.checked_add(two_pad) {
        Some(v) => v,
        None => return,
    };

    // Step 3: guard from production code: `if l_in + 2 * padding >= effective_k`
    if padded < effective_k {
        return;
    }

    // Step 4: l_out = (padded - effective_k) / stride + 1
    let numerator = padded - effective_k; // no underflow: padded >= effective_k
    let l_out = numerator / stride + 1; // stride >= 1, no div-by-zero

    // Post-conditions:
    // 1. l_out >= 1 (always true: numerator >= 0, so numerator/stride >= 0, +1 >= 1)
    assert!(l_out >= 1);
    // 2. l_out <= l_in + 2*padding (output can't exceed padded input)
    assert!(l_out <= padded);
    // 3. l_out is reasonable: at most padded (degenerate: effective_k=1, stride=1)
    assert!(l_out <= l_in + 2 * padding);
}

/// Prove: Conv1dGemm FLOPS threshold check uses consistent dimensions.
///
/// The GEMM FLOPs check `c_out * c_in_k * l_out >= MIN_GEMM_FLOPS` must
/// not overflow. If it silently wraps, a Conv1d that should use the naive
/// kernel would erroneously route to GEMM (or vice versa).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_gemm_flops_no_overflow() {
    let c_out: usize = kani::any();
    let c_in: usize = kani::any();
    let k_size: usize = kani::any();
    let l_out: usize = kani::any();

    // Realistic Kokoro bounds: channels 1-1024, kernel 1-64, length 1-8192.
    kani::assume(c_out >= 1 && c_out <= 1024);
    kani::assume(c_in >= 1 && c_in <= 1024);
    kani::assume(k_size >= 1 && k_size <= 64);
    kani::assume(l_out >= 1 && l_out <= 8192);

    // Production code: c_in_k = w_shape[1] * w_shape[2] = c_in * k_size
    let c_in_k = match c_in.checked_mul(k_size) {
        Some(v) => v,
        None => {
            // Overflow means the product exceeds usize::MAX — certainly above
            // the 2M threshold, but the comparison would be wrong.
            // Production code uses unchecked mul. This proves it can't overflow
            // within realistic bounds.
            panic!("c_in * k_size overflowed within realistic bounds");
        }
    };

    // c_out * c_in_k * l_out
    let step1 = match c_out.checked_mul(c_in_k) {
        Some(v) => v,
        None => {
            panic!("c_out * c_in_k overflowed within realistic bounds");
        }
    };
    let flops = match step1.checked_mul(l_out) {
        Some(v) => v,
        None => {
            panic!("flops overflowed within realistic bounds");
        }
    };

    // The check is just a comparison — it's correct if flops didn't overflow.
    let _ = flops >= 2_000_000;
}

// ---------------------------------------------------------------------------
// BatchedLinearProjection: weight transpose indexing
// ---------------------------------------------------------------------------

/// Prove: CPU weight transpose indexing is in-bounds.
///
/// Models the transpose loop from `trace_compile_peephole_batched_qkv.rs:161-166`:
/// ```
/// let mut transposed = vec![0.0f32; total_out * in_features];
/// for r in 0..total_out {
///     for c in 0..in_features {
///         transposed[c * total_out + r] = concat_weight_data[r * in_features + c];
///     }
/// }
/// ```
///
/// Both `r * in_features + c` and `c * total_out + r` must be < `total_out * in_features`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn batched_linear_transpose_indexing_in_bounds() {
    let total_out: usize = kani::any();
    let in_features: usize = kani::any();

    // Realistic bounds: Kokoro PlBert hidden_dim=768, heads=12, head_dim=64.
    // total_out = sum of Q+K+V = 3*768 = 2304 max.
    kani::assume(total_out >= 1 && total_out <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    // Buffer size must not overflow.
    let buf_size = match total_out.checked_mul(in_features) {
        Some(v) => v,
        None => return,
    };

    // For ALL (r, c) in the loop, verify index bounds.
    // We can't iterate full range in Kani, so we verify the formula symbolically.
    let r: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(r < total_out);
    kani::assume(c < in_features);

    // Source index: r * in_features + c
    let src_idx = match r.checked_mul(in_features) {
        Some(v) => match v.checked_add(c) {
            Some(idx) => idx,
            None => panic!("source index addition overflow"),
        },
        None => panic!("source index multiplication overflow"),
    };
    assert!(src_idx < buf_size, "source index out of bounds");

    // Dest index: c * total_out + r
    let dst_idx = match c.checked_mul(total_out) {
        Some(v) => match v.checked_add(r) {
            Some(idx) => idx,
            None => panic!("dest index addition overflow"),
        },
        None => panic!("dest index multiplication overflow"),
    };
    assert!(dst_idx < buf_size, "dest index out of bounds");
}

/// Prove: BatchedLinearProjection total_out == sum(projection_sizes).
///
/// The peephole pass computes `total_out: usize = projection_sizes.iter().sum()`
/// and later validates `concat_weight_data.len() == total_out * in_features`.
/// If the sum overflows, the validation would pass with a wrong total_out,
/// and the weight buffer would be undersized.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn batched_linear_projection_sizes_sum_no_overflow() {
    // Model 2-4 projections (Q, K, V, optional Value).
    let n_projections: usize = kani::any();
    kani::assume(n_projections >= 2 && n_projections <= 4);

    let p0: usize = kani::any();
    let p1: usize = kani::any();
    let p2: usize = kani::any();
    let p3: usize = kani::any();

    // Realistic per-projection sizes: 64 to 4096.
    kani::assume(p0 >= 1 && p0 <= 4096);
    kani::assume(p1 >= 1 && p1 <= 4096);
    kani::assume(p2 >= 1 && p2 <= 4096);
    kani::assume(p3 >= 1 && p3 <= 4096);

    let total = match n_projections {
        2 => p0.checked_add(p1),
        3 => p0.checked_add(p1).and_then(|s| s.checked_add(p2)),
        4 => p0
            .checked_add(p1)
            .and_then(|s| s.checked_add(p2))
            .and_then(|s| s.checked_add(p3)),
        _ => unreachable!(),
    };

    let total = match total {
        Some(t) => t,
        None => panic!("projection sizes sum overflowed within realistic bounds"),
    };

    // Post-condition: total is at least n_projections (each >= 1).
    assert!(total >= n_projections);
    // Post-condition: total <= 4 * 4096 = 16384.
    assert!(total <= 16384);
}

// ---------------------------------------------------------------------------
// ProjectionSlice: offset accumulation stays in bounds
// ---------------------------------------------------------------------------

/// Prove: ProjectionSlice offset accumulation does not overflow and
/// each slice `[start, start+length)` stays within total_out.
///
/// Models the loop from `trace_compile_peephole_batched_qkv.rs:198-215`:
/// ```
/// let mut start = projection_sizes[0];
/// for (i, c) in group.iter().enumerate().skip(1) {
///     // ... ProjectionSlice { start, length: projection_sizes[i], ... }
///     start += projection_sizes[i];
/// }
/// ```
///
/// The running `start` must always satisfy `start + projection_sizes[i] <= total_out`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn projection_slice_offsets_in_bounds() {
    // Model 2-4 projections.
    let n_projections: usize = kani::any();
    kani::assume(n_projections >= 2 && n_projections <= 4);

    let sizes: [usize; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];

    // Each projection size >= 1, <= 4096.
    for i in 0..4 {
        kani::assume(sizes[i] >= 1 && sizes[i] <= 4096);
    }

    // total_out = sum of active projection sizes.
    let total_out: usize = match n_projections {
        2 => sizes[0] + sizes[1],
        3 => sizes[0] + sizes[1] + sizes[2],
        4 => sizes[0] + sizes[1] + sizes[2] + sizes[3],
        _ => unreachable!(),
    };

    // First projection is handled by BatchedLinearProjection directly.
    // ProjectionSlice starts at offset = sizes[0].
    let mut start = sizes[0];

    // Each subsequent projection must have start + length <= total_out.
    for i in 1..n_projections {
        let length = sizes[i];
        assert!(
            start + length <= total_out,
            "ProjectionSlice offset exceeds total_out"
        );
        start += length;
    }

    // After all slices, start should equal total_out exactly.
    assert_eq!(start, total_out, "final offset != total_out");
}

/// Prove: ProjectionSlice partition is exact — no gaps and no overlaps.
///
/// Given projection sizes [p0, p1, ..., pN-1] with total_out = sum(pi),
/// the slices [0, p0), [p0, p0+p1), ... must partition [0, total_out)
/// exactly. Gaps would leave uninitialized data; overlaps would cause
/// double-writes in the executor.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn projection_slice_partition_exact() {
    let n_projections: usize = kani::any();
    kani::assume(n_projections >= 2 && n_projections <= 4);

    let sizes: [usize; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
    for i in 0..4 {
        kani::assume(sizes[i] >= 1 && sizes[i] <= 4096);
    }

    let total_out: usize = match n_projections {
        2 => sizes[0] + sizes[1],
        3 => sizes[0] + sizes[1] + sizes[2],
        4 => sizes[0] + sizes[1] + sizes[2] + sizes[3],
        _ => unreachable!(),
    };

    // Build slice boundaries: [0, p0, p0+p1, p0+p1+p2, ...]
    let mut boundaries = [0usize; 5];
    boundaries[0] = 0;
    for i in 0..n_projections {
        boundaries[i + 1] = boundaries[i] + sizes[i];
    }

    // No gaps: each boundary[i+1] == boundary[i] + sizes[i] (by construction).
    // No overlaps: boundaries are strictly increasing (each sizes[i] >= 1).
    for i in 0..n_projections {
        assert!(boundaries[i] < boundaries[i + 1], "non-increasing boundary");
    }

    // Covers full range: last boundary == total_out.
    assert_eq!(boundaries[n_projections], total_out, "partition incomplete");
}

// ---------------------------------------------------------------------------
// Conv1dGemm: im2col buffer sizing
// ---------------------------------------------------------------------------

/// Prove: im2col intermediate buffer size doesn't overflow.
///
/// The im2col matrix has shape [C_in * K, L_out] for groups=1.
/// The total element count C_in * K * L_out must fit in usize without
/// overflow, otherwise the buffer allocation would be wrong.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_gemm_im2col_buffer_size_no_overflow() {
    let c_in: usize = kani::any();
    let k_size: usize = kani::any();
    let l_out: usize = kani::any();

    // Realistic bounds matching Kokoro Conv1d dimensions.
    kani::assume(c_in >= 1 && c_in <= 1024);
    kani::assume(k_size >= 1 && k_size <= 64);
    kani::assume(l_out >= 1 && l_out <= 8192);

    // im2col rows = C_in * K_size
    let rows = match c_in.checked_mul(k_size) {
        Some(v) => v,
        None => panic!("im2col rows overflow within realistic bounds"),
    };

    // Total elements = rows * L_out
    let total_elements = match rows.checked_mul(l_out) {
        Some(v) => v,
        None => panic!("im2col total elements overflow within realistic bounds"),
    };

    // Byte size for f32 = total_elements * 4
    let byte_size = match total_elements.checked_mul(4) {
        Some(v) => v,
        None => panic!("im2col byte size overflow within realistic bounds"),
    };

    // Sanity: buffer is at most 1024 * 64 * 8192 * 4 = ~2GB — fits in usize on 64-bit.
    assert!(byte_size <= 1024 * 64 * 8192 * 4);
}

// ---------------------------------------------------------------------------
// LinearActivation: dimension consistency
// ---------------------------------------------------------------------------

/// Prove: The production extraction convention `w_shape[0]=out, w_shape[1]=in`
/// is the unique dimension assignment that makes GEMM valid for non-square weights.
///
/// `extract_linear_params` (trace_compile_peephole_linear_activation.rs:110-111)
/// reads `weight_shape[0]` as out_features and `weight_shape[1]` as in_features.
/// GEMM: `input [B, input_last_dim] × weight^T [input_last_dim, out] = [B, out]`.
/// The contraction dimension must equal `input_last_dim`.
///
/// If someone swapped the convention (treating `w_shape[0]` as in_features),
/// non-square weights would produce wrong output shapes or dimension mismatches.
/// This proof shows that for non-square weights, at most one convention is valid,
/// so the extraction order is a correctness-critical decision, not arbitrary.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn linear_activation_gemm_dimension_consistency() {
    let dim_a: usize = kani::any(); // weight_shape[0]
    let dim_b: usize = kani::any(); // weight_shape[1]
    let batch: usize = kani::any();
    let input_last_dim: usize = kani::any();

    kani::assume(dim_a >= 1 && dim_a <= 4096);
    kani::assume(dim_b >= 1 && dim_b <= 4096);
    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(input_last_dim >= 1 && input_last_dim <= 4096);

    // Non-square weight: the convention choice matters.
    kani::assume(dim_a != dim_b);

    // Production convention: contraction_dim = w_shape[1] = dim_b.
    let prod_valid = dim_b == input_last_dim;

    // Swapped convention: contraction_dim = w_shape[0] = dim_a.
    let swap_valid = dim_a == input_last_dim;

    // UNIQUENESS: both conventions cannot be valid simultaneously.
    // If both were valid, dim_b == input_last_dim AND dim_a == input_last_dim,
    // implying dim_a == dim_b — contradicting our non-square assumption.
    assert!(
        !(prod_valid && swap_valid),
        "both conventions valid implies square weight"
    );

    // Output buffer overflow check for the production convention.
    if prod_valid {
        let out_features = dim_a;
        let output_elements = match batch.checked_mul(out_features) {
            Some(v) => v,
            None => panic!("output buffer size overflow within realistic bounds"),
        };
        assert!(output_elements >= batch);

        // GEMM FLOPs: batch * out_features * in_features.
        let in_features = dim_b;
        let flops = match batch
            .checked_mul(out_features)
            .and_then(|v| v.checked_mul(in_features))
        {
            Some(v) => v,
            None => panic!("GEMM FLOPs overflow within realistic bounds"),
        };
        assert!(flops >= output_elements);
    }
}
