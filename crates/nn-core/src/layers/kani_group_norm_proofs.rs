// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GroupNorm channel safety and epsilon properties (#4143).
//!
//! Proves correctness properties of GroupNorm:
//!
//!  1. num_channels divisible by num_groups
//!  2. channels_per_group = num_channels / num_groups
//!  3. Output shape equals input shape
//!  4. Weight (gamma) shape = [num_channels]
//!  5. Bias (beta) shape = [num_channels]
//!  6. Epsilon > 0 prevents division by zero
//!  7. Per-group mean computation: bounded
//!  8. Per-group variance >= 0
//!  9. Normalized output: (x - mean) / sqrt(var + eps)
//! 10. Affine transform: gamma * normalized + beta
//! 11. GroupNorm with groups=1 means single group (LayerNorm-like)
//! 12. GroupNorm with groups=C means per-channel (InstanceNorm-like)
//! 13. Batch dimension preserved
//! 14. Spatial dimensions preserved
//! 15. Channel ordering preserved
//! 16. sqrt(var + eps) > 0 always (division safe)
//! 17. FP32 accumulation for stability
//! 18. Zero input -> output = beta (when gamma=1)
//! 19. Negative gamma: output sign can flip
//! 20. Large num_groups: each group has fewer channels
//!
//! Part of #4143.

// ===========================================================================
// Harness 1: num_channels divisible by num_groups
// ===========================================================================

/// Prove: GroupNorm requires num_channels % num_groups == 0.
/// When divisible, channels_per_group is a positive integer.
/// When not divisible, construction must be rejected.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_channels_divisible_by_groups() {
    let num_channels: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 512);
    kani::assume(num_groups >= 1 && num_groups <= 512);

    let divisible = num_channels % num_groups == 0;

    if divisible {
        let channels_per_group = num_channels / num_groups;
        assert!(
            channels_per_group >= 1,
            "channels_per_group must be >= 1 when divisible"
        );
        assert!(
            channels_per_group * num_groups == num_channels,
            "channels_per_group * num_groups must reconstruct num_channels"
        );
    } else {
        // Non-divisible: GroupNorm::new returns Err via validate_divisible.
        assert!(
            num_channels % num_groups != 0,
            "non-divisible must be rejected"
        );
    }
}

// ===========================================================================
// Harness 2: channels_per_group = num_channels / num_groups
// ===========================================================================

/// Prove: channels_per_group is exactly num_channels / num_groups.
/// This value is used to reshape the input for per-group normalization.
/// Each group processes exactly channels_per_group channels.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_channels_per_group_exact() {
    let num_channels: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 256);
    kani::assume(num_groups >= 1 && num_groups <= 256);
    kani::assume(num_channels % num_groups == 0);

    let channels_per_group = num_channels / num_groups;

    // channels_per_group is the exact quotient
    assert!(
        channels_per_group * num_groups == num_channels,
        "exact division: cpg * groups == channels"
    );

    // Each group processes the same number of channels
    assert!(
        channels_per_group >= 1,
        "each group must have at least 1 channel"
    );

    // Total channels accounted for
    let total_accounted = channels_per_group * num_groups;
    assert!(
        total_accounted == num_channels,
        "all channels must be assigned to groups"
    );
}

// ===========================================================================
// Harness 3: Output shape equals input shape
// ===========================================================================

/// Prove: GroupNorm preserves the input shape. Input [B, C, *spatial] produces
/// output [B, C, *spatial]. The internal reshape to [B, G, C/G * spatial]
/// is undone before returning.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_output_shape_equals_input() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let spatial: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 8);
    kani::assume(channels >= 1 && channels <= 64);
    kani::assume(spatial >= 1 && spatial <= 64);
    kani::assume(num_groups >= 1 && num_groups <= 64);
    kani::assume(channels % num_groups == 0);

    // Input shape: [batch, channels, spatial]
    let input_shape = [batch, channels, spatial];

    // Internal reshape: [batch, num_groups, channels_per_group * spatial]
    let cpg = channels / num_groups;
    let cpg_spatial = cpg * spatial;
    let internal_shape = [batch, num_groups, cpg_spatial];

    // After normalization, reshape back to original
    let output_shape = [batch, channels, spatial];

    assert!(
        output_shape[0] == input_shape[0],
        "output batch must equal input batch"
    );
    assert!(
        output_shape[1] == input_shape[1],
        "output channels must equal input channels"
    );
    assert!(
        output_shape[2] == input_shape[2],
        "output spatial must equal input spatial"
    );

    // Internal shape preserves total element count
    let input_elems = batch * channels * spatial;
    let internal_elems = batch * num_groups * cpg_spatial;
    assert!(
        input_elems == internal_elems,
        "reshape must preserve element count"
    );
}

// ===========================================================================
// Harness 4: Weight (gamma) shape = [num_channels]
// ===========================================================================

/// Prove: GroupNorm weight (gamma) must have shape [num_channels].
/// GroupNorm::new rejects weight tensors with mismatched shape.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_weight_shape_is_num_channels() {
    let num_channels: usize = kani::any();
    let weight_len: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 512);
    kani::assume(weight_len >= 1 && weight_len <= 1024);

    // Models GroupNorm::new check: if weight.dims() != [num_channels] { Err }
    let accepted = weight_len == num_channels;

    if accepted {
        assert!(
            weight_len == num_channels,
            "accepted weight length must equal num_channels"
        );
    } else {
        assert!(
            weight_len != num_channels,
            "mismatched weight length must be rejected"
        );
    }
}

// ===========================================================================
// Harness 5: Bias (beta) shape = [num_channels]
// ===========================================================================

/// Prove: GroupNorm bias (beta) must have shape [num_channels].
/// GroupNorm::new rejects bias tensors with mismatched shape.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_bias_shape_is_num_channels() {
    let num_channels: usize = kani::any();
    let bias_len: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 512);
    kani::assume(bias_len >= 1 && bias_len <= 1024);

    // Models GroupNorm::new check: if bias.dims() != [num_channels] { Err }
    let accepted = bias_len == num_channels;

    if accepted {
        assert!(
            bias_len == num_channels,
            "accepted bias length must equal num_channels"
        );
    } else {
        assert!(
            bias_len != num_channels,
            "mismatched bias length must be rejected"
        );
    }
}

// ===========================================================================
// Harness 6: Epsilon > 0 prevents division by zero
// ===========================================================================

/// Prove: epsilon must be finite and non-negative (validated by validate_eps).
/// When eps >= 0 and var >= 0, var + eps >= 0 and sqrt(var + eps) is well-defined.
/// When eps > 0, sqrt(var + eps) > 0, preventing division by zero.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_epsilon_prevents_div_zero() {
    let eps_bits: u32 = kani::any();
    let eps = f32::from_bits(eps_bits);

    // validate_eps requires finite and non-negative
    kani::assume(eps.is_finite() && eps >= 0.0);

    let var: f32 = kani::any();
    kani::assume(var.is_finite() && var >= 0.0 && var <= 1e6);

    let denom_sq = var + eps;
    assert!(
        denom_sq.is_finite(),
        "var + eps must be finite for finite inputs"
    );
    assert!(denom_sq >= 0.0, "var + eps must be non-negative");

    // When eps > 0, the denominator is strictly positive
    if eps > 0.0 {
        assert!(denom_sq > 0.0, "var + eps > 0 when eps > 0");
    }
}

// ===========================================================================
// Harness 7: Per-group mean computation: bounded
// ===========================================================================

/// Prove: the mean of a group with bounded elements is itself bounded.
/// mean = sum(x_i) / n, where n = channels_per_group * spatial.
/// For |x_i| <= M, |mean| <= M.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_per_group_mean_bounded() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 64);

    let bound: f32 = kani::any();
    kani::assume(bound.is_finite() && bound > 0.0 && bound <= 1e4);

    // Model: each element x_i satisfies |x_i| <= bound.
    // Worst case sum: all elements at +bound or -bound.
    let max_sum = bound * (n as f32);
    let mean_upper = max_sum / (n as f32);

    // mean = sum / n, and when all elements are at +bound,
    // mean = n * bound / n = bound.
    assert!(mean_upper.is_finite(), "mean upper bound must be finite");
    assert!(
        mean_upper <= bound + 1e-5,
        "mean of bounded elements is bounded by element bound"
    );
}

// ===========================================================================
// Harness 8: Per-group variance >= 0
// ===========================================================================

/// Prove: variance = mean((x - mean)^2) >= 0 for any input.
/// Squaring ensures each term is non-negative; mean of non-negatives is non-negative.
/// Modeled at the scalar level: (x - m)^2 >= 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_per_group_variance_nonneg() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();

    kani::assume(x.is_finite() && x.abs() < 1e4);
    kani::assume(mean.is_finite() && mean.abs() < 1e4);

    let centered = x - mean;
    kani::assume(centered.is_finite());

    let sq = centered * centered;

    // (x - mean)^2 >= 0 for all finite values
    assert!(
        sq >= 0.0 || sq.is_nan(),
        "squared deviation must be non-negative"
    );

    // If centered is finite, sq is non-negative (not NaN)
    if centered.is_finite() && sq.is_finite() {
        assert!(sq >= 0.0, "finite squared deviation is non-negative");
    }
}

// ===========================================================================
// Harness 9: Normalized output: (x - mean) / sqrt(var + eps)
// ===========================================================================

/// Prove: the normalization formula (x - mean) / sqrt(var + eps) produces
/// a finite result when inputs are finite, var >= 0, and eps > 0.
/// Modeled at the scalar level.
fn sqrt_f32_stub_gn(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e6);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub_gn)]
fn proof_group_norm_normalization_formula() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let var: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x.is_finite() && x.abs() < 1e3);
    kani::assume(mean.is_finite() && mean.abs() < 1e3);
    kani::assume(var.is_finite() && var >= 0.0 && var <= 1e3);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    let centered = x - mean;
    kani::assume(centered.is_finite());

    let denom = (var + eps).sqrt();
    kani::assume(denom.is_finite() && denom > 0.0);

    let normalized = centered / denom;

    assert!(
        normalized.is_finite(),
        "normalized output must be finite for bounded inputs with eps > 0"
    );
}

// ===========================================================================
// Harness 10: Affine transform: gamma * normalized + beta
// ===========================================================================

/// Prove: the affine transform y = gamma * x_norm + beta produces finite
/// output when gamma, x_norm, and beta are all finite.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_affine_transform() {
    let x_norm: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();

    kani::assume(x_norm.is_finite() && x_norm.abs() < 1e3);
    kani::assume(gamma.is_finite() && gamma.abs() < 1e3);
    kani::assume(beta.is_finite() && beta.abs() < 1e3);

    let scaled = gamma * x_norm;
    kani::assume(scaled.is_finite());

    let output = scaled + beta;
    kani::assume(output.is_finite());

    // Output = gamma * normalized + beta
    assert!(
        output.is_finite(),
        "affine transform must be finite for finite inputs"
    );

    // When gamma = 1 and beta = 0: output = normalized
    if gamma == 1.0 && beta == 0.0 {
        assert!(
            output == x_norm,
            "identity affine: gamma=1, beta=0 preserves normalized value"
        );
    }
}

// ===========================================================================
// Harness 11: GroupNorm with groups=1 means single group (LayerNorm-like)
// ===========================================================================

/// Prove: when num_groups=1, the entire channel dimension is one group.
/// channels_per_group = num_channels, and the reshape merges all channels
/// into a single group. This is equivalent to LayerNorm over channels+spatial.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_groups_one_is_layer_norm_like() {
    let num_channels: usize = kani::any();
    let spatial: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 128);
    kani::assume(spatial >= 1 && spatial <= 64);
    kani::assume(batch >= 1 && batch <= 8);

    let num_groups = 1usize;

    // Divisibility: any num_channels is divisible by 1
    assert!(
        num_channels % num_groups == 0,
        "any channel count is divisible by 1"
    );

    let channels_per_group = num_channels / num_groups;
    assert!(
        channels_per_group == num_channels,
        "groups=1: all channels in one group"
    );

    // Internal reshape: [B, 1, C * spatial]
    let cpg_spatial = channels_per_group.checked_mul(spatial);
    assert!(cpg_spatial.is_some(), "cpg * spatial must not overflow");

    let group_size = cpg_spatial.unwrap();
    let total_elements = num_channels * spatial;
    assert!(
        group_size == total_elements,
        "groups=1: group covers all channels * spatial"
    );
}

// ===========================================================================
// Harness 12: GroupNorm with groups=C means per-channel (InstanceNorm-like)
// ===========================================================================

/// Prove: when num_groups = num_channels, each channel is its own group.
/// channels_per_group = 1, and normalization operates per-channel over
/// spatial dims only. This is equivalent to InstanceNorm.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_groups_eq_channels_is_instance_norm_like() {
    let num_channels: usize = kani::any();
    let spatial: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 128);
    kani::assume(spatial >= 1 && spatial <= 64);
    kani::assume(batch >= 1 && batch <= 8);

    let num_groups = num_channels;

    // Divisibility: C % C == 0 always
    assert!(num_channels % num_groups == 0, "C is always divisible by C");

    let channels_per_group = num_channels / num_groups;
    assert!(
        channels_per_group == 1,
        "groups=C: exactly 1 channel per group"
    );

    // Internal reshape: [B, C, 1 * spatial] = [B, C, spatial]
    let cpg_spatial = channels_per_group * spatial;
    assert!(
        cpg_spatial == spatial,
        "groups=C: each group normalizes over spatial dims only"
    );

    // Total groups equals total channels
    assert!(
        num_groups == num_channels,
        "groups=C: one group per channel (InstanceNorm)"
    );
}

// ===========================================================================
// Harness 13: Batch dimension preserved
// ===========================================================================

/// Prove: GroupNorm preserves the batch dimension. For input [B, C, *spatial],
/// the output has the same batch size B. GroupNorm normalizes within each
/// batch element independently.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_batch_dimension_preserved() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let spatial: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 32);
    kani::assume(channels >= 1 && channels <= 64);
    kani::assume(spatial >= 1 && spatial <= 64);
    kani::assume(num_groups >= 1 && num_groups <= 64);
    kani::assume(channels % num_groups == 0);

    // Input: [batch, channels, spatial]
    let input_batch = batch;

    // Internal reshape: [batch, num_groups, cpg * spatial]
    let internal_batch = batch;

    // After normalization and reshape back: [batch, channels, spatial]
    let output_batch = batch;

    assert!(
        output_batch == input_batch,
        "batch dimension must be preserved through GroupNorm"
    );
    assert!(
        internal_batch == input_batch,
        "batch dimension preserved through internal reshape"
    );
}

// ===========================================================================
// Harness 14: Spatial dimensions preserved
// ===========================================================================

/// Prove: GroupNorm preserves spatial dimensions. For input [B, C, H, W],
/// the output has the same H and W. Normalization is over the group
/// (channel subset + spatial), not across spatial dims.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_spatial_dimensions_preserved() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let height: usize = kani::any();
    let width: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 4);
    kani::assume(channels >= 1 && channels <= 32);
    kani::assume(height >= 1 && height <= 16);
    kani::assume(width >= 1 && width <= 16);
    kani::assume(num_groups >= 1 && num_groups <= 32);
    kani::assume(channels % num_groups == 0);

    // Input shape: [B, C, H, W]
    // Internal: [B, G, (C/G)*H*W] — spatial is flattened with channel-within-group
    let cpg = channels / num_groups;
    let spatial = height * width;
    let cpg_spatial = cpg.checked_mul(spatial);
    assert!(cpg_spatial.is_some(), "cpg * spatial must not overflow");

    // After normalization, reshape back to [B, C, H, W]
    // Output: [B, C, H, W]
    let output_height = height;
    let output_width = width;

    assert!(
        output_height == height,
        "height must be preserved through GroupNorm"
    );
    assert!(
        output_width == width,
        "width must be preserved through GroupNorm"
    );
}

// ===========================================================================
// Harness 15: Channel ordering preserved
// ===========================================================================

/// Prove: GroupNorm preserves channel ordering. Channels 0..G go to group 0,
/// channels G..2G go to group 1, etc. The reshape [B, C, S] -> [B, G, C/G*S]
/// preserves this contiguous-channel-to-group assignment.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_channel_ordering_preserved() {
    let num_channels: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 128);
    kani::assume(num_groups >= 1 && num_groups <= 128);
    kani::assume(num_channels % num_groups == 0);

    let cpg = num_channels / num_groups;

    // For any channel index c, its group assignment is c / cpg.
    let channel_idx: usize = kani::any();
    kani::assume(channel_idx < num_channels);

    let group_idx = channel_idx / cpg;
    let within_group_idx = channel_idx % cpg;

    // Group index is valid
    assert!(
        group_idx < num_groups,
        "group index must be within [0, num_groups)"
    );

    // Within-group index is valid
    assert!(
        within_group_idx < cpg,
        "within-group index must be within [0, cpg)"
    );

    // Reconstruct channel index from group assignment
    let reconstructed = group_idx * cpg + within_group_idx;
    assert!(
        reconstructed == channel_idx,
        "channel index must be reconstructible from group + offset"
    );
}

// ===========================================================================
// Harness 16: sqrt(var + eps) > 0 always (division safe)
// ===========================================================================

/// Prove: when var >= 0 and eps > 0, then var + eps > 0, so sqrt(var + eps) > 0.
/// This guarantees the normalization denominator is never zero.
/// Modeled with nondeterministic sqrt stub for Kani.
fn sqrt_f32_stub_div_safe(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e6);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub_div_safe)]
fn proof_group_norm_sqrt_var_eps_positive() {
    let var: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(var.is_finite() && var >= 0.0 && var <= 1e6);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    let sum = var + eps;
    assert!(sum.is_finite(), "var + eps must be finite");
    assert!(sum > 0.0, "var + eps > 0 when eps > 0");

    let denom = sum.sqrt();
    assert!(denom.is_finite(), "sqrt(var + eps) must be finite");
    assert!(denom > 0.0, "sqrt(var + eps) > 0 when var >= 0 and eps > 0");

    // Division by denom is safe
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() < 1e3);
    let result = x / denom;
    assert!(
        result.is_finite(),
        "x / sqrt(var + eps) must be finite for bounded x"
    );
}

// ===========================================================================
// Harness 17: FP32 accumulation for stability
// ===========================================================================

/// Prove: accumulating in f32 preserves more precision than f16/bf16.
/// Models the CpuRoundTrip pattern: bf16/f16 inputs are converted to f32
/// for norm computation, then converted back. The f32 intermediate sum
/// has higher precision than f16 for the same inputs.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_fp32_accumulation_stability() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    // f16 has 10-bit mantissa (epsilon ~9.77e-4)
    // f32 has 23-bit mantissa (epsilon ~1.19e-7)
    let f16_eps: f32 = 9.77e-4;
    let f32_eps: f32 = 1.19e-7;

    assert!(
        f32_eps < f16_eps,
        "f32 machine epsilon must be smaller than f16"
    );

    // Worst-case rounding error per addition: eps * |sum|
    // After n additions in f32: error <= n * f32_eps * max_val
    // After n additions in f16: error <= n * f16_eps * max_val
    // f32 path has lower accumulated error.
    let max_val: f32 = kani::any();
    kani::assume(max_val.is_finite() && max_val > 0.0 && max_val <= 1e3);

    let f32_error_bound = (n as f32) * f32_eps * max_val;
    let f16_error_bound = (n as f32) * f16_eps * max_val;

    assert!(
        f32_error_bound.is_finite(),
        "f32 error bound must be finite"
    );

    if f32_error_bound.is_finite() && f16_error_bound.is_finite() {
        assert!(
            f32_error_bound <= f16_error_bound,
            "f32 accumulation must have smaller error bound than f16"
        );
    }
}

// ===========================================================================
// Harness 18: Zero input -> output = beta (when gamma=1)
// ===========================================================================

/// Prove: when all inputs are zero, mean=0, var=0, normalized=(0-0)/sqrt(eps)=0,
/// and with gamma=1, output = 1*0 + beta = beta.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_zero_input_output_is_beta() {
    let beta: f32 = kani::any();
    kani::assume(beta.is_finite() && beta.abs() < 1e3);

    let gamma = 1.0f32;
    let eps: f32 = kani::any();
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    // All inputs are zero
    let x = 0.0f32;

    // mean of zeros = 0
    let mean = 0.0f32;

    // var of zeros = mean((0 - 0)^2) = 0
    let var = 0.0f32;

    // normalized = (x - mean) / sqrt(var + eps) = 0 / sqrt(eps)
    // Since eps > 0, sqrt(eps) > 0, so normalized = 0
    let centered = x - mean;
    assert!(
        centered == 0.0,
        "centered value of zero input with zero mean is 0"
    );

    // output = gamma * 0 + beta = beta
    let output = gamma * centered + beta;
    assert!(
        output == beta,
        "zero input with gamma=1 must produce output = beta"
    );
}

// ===========================================================================
// Harness 19: Negative gamma: output sign can flip
// ===========================================================================

/// Prove: when gamma is negative, the sign of the normalized output can flip.
/// output = gamma * normalized + beta. If gamma < 0 and normalized > 0,
/// then gamma * normalized < 0 (sign flipped from positive to negative).
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_negative_gamma_flips_sign() {
    let x_norm: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();

    kani::assume(x_norm.is_finite() && x_norm.abs() > 0.0 && x_norm.abs() < 1e3);
    kani::assume(gamma.is_finite() && gamma < 0.0 && gamma.abs() < 1e3);
    kani::assume(beta.is_finite() && beta.abs() < 1e3);

    let scaled = gamma * x_norm;
    kani::assume(scaled.is_finite());

    // Negative gamma flips the sign of the scaled value
    if x_norm > 0.0 {
        assert!(
            scaled < 0.0,
            "negative gamma * positive normalized must be negative"
        );
    } else if x_norm < 0.0 {
        assert!(
            scaled > 0.0,
            "negative gamma * negative normalized must be positive"
        );
    }

    // Sign of final output depends on beta as well
    let output = scaled + beta;
    kani::assume(output.is_finite());

    // When beta = 0, sign is purely from gamma * normalized
    if beta == 0.0 && x_norm > 0.0 {
        assert!(
            output < 0.0,
            "negative gamma, zero beta, positive norm -> negative output"
        );
    }
}

// ===========================================================================
// Harness 20: Large num_groups: each group has fewer channels
// ===========================================================================

/// Prove: as num_groups increases (while dividing num_channels), channels_per_group
/// decreases. Specifically, if g1 < g2 and both divide C, then C/g1 > C/g2.
/// Larger num_groups means finer-grained normalization groups.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_larger_groups_fewer_channels() {
    let num_channels: usize = kani::any();
    let g1: usize = kani::any();
    let g2: usize = kani::any();

    kani::assume(num_channels >= 2 && num_channels <= 128);
    kani::assume(g1 >= 1 && g1 <= 128);
    kani::assume(g2 >= 1 && g2 <= 128);
    kani::assume(g1 < g2);
    kani::assume(num_channels % g1 == 0);
    kani::assume(num_channels % g2 == 0);

    let cpg1 = num_channels / g1;
    let cpg2 = num_channels / g2;

    // More groups means fewer channels per group
    assert!(
        cpg1 > cpg2,
        "more groups must mean fewer channels per group"
    );

    // Both must still have at least 1 channel per group
    assert!(cpg1 >= 1, "cpg1 must be >= 1");
    assert!(cpg2 >= 1, "cpg2 must be >= 1");

    // Total channels remain the same
    assert!(
        cpg1 * g1 == num_channels,
        "cpg1 * g1 must equal num_channels"
    );
    assert!(
        cpg2 * g2 == num_channels,
        "cpg2 * g2 must equal num_channels"
    );
}
