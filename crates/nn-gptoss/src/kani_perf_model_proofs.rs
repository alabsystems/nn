// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proofs for the performance model module.
//!
//! Proves key invariants of the roofline performance model:
//! - FLOP counts are non-negative (guaranteed by u64)
//! - Arithmetic intensity is positive for non-trivial operations
//! - Predicted latency is positive for non-zero work
//! - Bandwidth utilization is bounded in [0, 1]
//! - Prefill latency >= decode latency for the same model config

use crate::perf_model::{
    profile_attention, profile_full_forward_on, profile_moe_block, ForwardProfile, HardwareProfile,
    OperationProfile,
};

/// Prove that FLOP counts are non-negative for attention profiling.
///
/// Since `flops` is `u64`, it cannot be negative. This proof verifies
/// that no overflow wraps the value to an unexpectedly small number
/// by checking that attention FLOPs are at least as large as the
/// minimal expected contribution.
#[kani::proof]
#[kani::unwind(2)]
fn proof_flops_nonnegative() {
    let seq_len: usize = kani::any();
    let cached_len: usize = kani::any();

    // Bound inputs to prevent trivial overflow saturation from dominating.
    kani::assume(seq_len > 0 && seq_len <= 4096);
    kani::assume(cached_len <= 131_072);

    let cfg = crate::config::GptOssConfig::gptoss_20b();
    let attn_prof = profile_attention(&cfg, seq_len, cached_len);
    let moe_prof = profile_moe_block(&cfg, seq_len);

    // u64 is inherently non-negative. Verify no saturation to 0 for
    // non-zero inputs.
    assert!(
        attn_prof.flops > 0,
        "attention FLOPs must be > 0 for seq_len > 0"
    );
    assert!(moe_prof.flops > 0, "MoE FLOPs must be > 0 for seq_len > 0");

    // Memory bytes must also be positive for non-zero work.
    assert!(attn_prof.memory_bytes > 0);
    assert!(moe_prof.memory_bytes > 0);
}

/// Prove that arithmetic intensity is positive when both flops and bytes
/// are non-zero.
#[kani::proof]
fn proof_arithmetic_intensity_positive() {
    let flops: u64 = kani::any();
    let bytes: u64 = kani::any();

    kani::assume(flops > 0);
    kani::assume(bytes > 0);

    let op = OperationProfile::new(flops, bytes);

    // arithmetic_intensity = flops / bytes, both > 0 -> result > 0
    assert!(
        op.arithmetic_intensity > 0.0,
        "arithmetic intensity must be > 0 when flops > 0 and bytes > 0"
    );
    assert!(
        op.arithmetic_intensity.is_finite(),
        "arithmetic intensity must be finite when bytes > 0"
    );
}

/// Prove that predicted latency is positive for any non-zero work on
/// valid hardware.
#[kani::proof]
fn proof_latency_positive() {
    let flops: u64 = kani::any();
    let bytes: u64 = kani::any();

    // At least one of flops or bytes is non-zero
    kani::assume(flops > 0 || bytes > 0);

    let op = OperationProfile::new(flops, bytes);
    let hw = HardwareProfile::m4_max();

    let latency = op.predicted_latency_us(&hw);
    assert!(latency > 0.0, "latency must be > 0 for non-zero work");
    assert!(
        latency.is_finite(),
        "latency must be finite on valid hardware"
    );
}

/// Prove that memory bandwidth utilization is bounded in [0, 1]
/// for all valid forward profiles.
#[kani::proof]
#[kani::unwind(2)]
fn proof_bandwidth_utilization_bounded() {
    let seq_len: usize = kani::any();
    let cached_len: usize = kani::any();

    kani::assume(seq_len > 0 && seq_len <= 2048);
    kani::assume(cached_len <= 8192);

    let cfg = crate::config::GptOssConfig::gptoss_20b();
    let hw = HardwareProfile::m4_max();
    let prof = profile_full_forward_on(&cfg, seq_len, cached_len, &hw);

    assert!(
        prof.memory_bandwidth_utilization >= 0.0,
        "bandwidth utilization must be >= 0"
    );
    assert!(
        prof.memory_bandwidth_utilization <= 1.0,
        "bandwidth utilization must be <= 1"
    );
    assert!(
        prof.compute_utilization >= 0.0,
        "compute utilization must be >= 0"
    );
    assert!(
        prof.compute_utilization <= 1.0,
        "compute utilization must be <= 1"
    );
}

/// Prove that prefill latency (many tokens, no cache) is greater than or
/// equal to single-token decode latency (1 token, with cache) for the
/// same model config.
#[kani::proof]
#[kani::unwind(2)]
fn proof_prefill_slower_than_decode() {
    let prefill_len: usize = kani::any();
    let cache_len: usize = kani::any();

    // Prefill has multiple tokens; decode is exactly 1 token.
    kani::assume(prefill_len >= 2 && prefill_len <= 1024);
    kani::assume(cache_len <= 8192);

    let cfg = crate::config::GptOssConfig::gptoss_20b();
    let hw = HardwareProfile::m4_max();

    let decode = profile_full_forward_on(&cfg, 1, cache_len, &hw);
    let prefill = profile_full_forward_on(&cfg, prefill_len, 0, &hw);

    assert!(
        prefill.predicted_latency_us >= decode.predicted_latency_us,
        "prefill must be >= decode latency"
    );
    assert!(
        prefill.total_flops >= decode.total_flops,
        "prefill must have >= decode FLOPs"
    );
}
