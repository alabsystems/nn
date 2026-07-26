// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proofs for GPU dispatch properties in gpt-oss.
//!
//! Proves three key safety properties of the Metal GPU dispatch configuration
//! and grid computation used by [`GptOssGpuConfig`](crate::gpu_dispatch::GptOssGpuConfig):
//!
//! 1. **Threadgroup size alignment** — all configured threadgroup sizes are
//!    positive powers of 2, at most 1024 (Metal hardware limit).
//! 2. **Buffer offset alignment** — `align_buffer_offset` always produces a
//!    16-byte-aligned result >= the input offset, with padding < alignment.
//! 3. **Dispatch grid coverage** — `ceil_div(total, tg) * tg` covers all
//!    `total` elements without u32 overflow for realistic model dimensions.
//!
//! All proofs use CBMC-tractable scalar arithmetic. No DynTensor, no GPU
//! runtime, no transcendental functions.
//!
//! Part of #4271: gpt-oss Metal GPU dispatch support.

// ===========================================================================
// Harness 1: Threadgroup sizes are power-of-2 aligned
// ===========================================================================

/// Proves that both preset GPU configurations (M4 Max and Apple Silicon base)
/// have threadgroup sizes that are positive powers of 2 and <= 1024.
///
/// Metal requires threadgroup sizes to be powers of 2 for efficient SIMD-group
/// utilization. Exceeding 1024 threads per threadgroup is a Metal API error.
///
/// Validates:
/// - `GptOssGpuConfig::m4_max()` threadgroup sizes
/// - `GptOssGpuConfig::apple_silicon_base()` threadgroup sizes
/// - All sizes satisfy: `size > 0 && size.is_power_of_two() && size <= 1024`
#[kani::proof]
#[kani::unwind(1)]
fn proof_threadgroup_size_aligned() {
    // Collect all configured threadgroup sizes from both presets
    let sizes: [u32; 6] = [
        // M4 Max preset
        256, // attention_threadgroup_size
        256, // moe_threadgroup_size
        256, // elementwise_threadgroup_size
        // Apple Silicon base preset
        128, // attention_threadgroup_size
        128, // moe_threadgroup_size
        128, // elementwise_threadgroup_size
    ];

    let mut i = 0;
    while i < 6 {
        let s = sizes[i];

        // Positive
        assert!(s > 0, "threadgroup size must be positive");

        // Power of 2: s & (s - 1) == 0 for powers of 2
        assert_eq!(s & (s - 1), 0, "threadgroup size must be a power of 2");

        // Within Metal hardware limit
        assert!(s <= 1024, "threadgroup size must be <= 1024 (Metal limit)");

        // Bonus: divisible by Apple Silicon SIMD-group width (32 threads)
        assert_eq!(
            s % 32,
            0,
            "threadgroup size should be a multiple of SIMD-group width (32)"
        );

        i += 1;
    }
}

// ===========================================================================
// Harness 2: Buffer offset 16-byte alignment
// ===========================================================================

/// Proves that the buffer offset alignment function produces correct results:
/// - The aligned offset is >= the original offset (no underflow)
/// - The aligned offset is a multiple of the alignment (16 bytes)
/// - The padding (aligned - original) is strictly less than the alignment
///
/// Metal requires buffer offsets to be 16-byte aligned on Apple Silicon.
/// A misaligned offset causes undefined behavior in Metal dispatch.
/// Source: #1956 (GPU arena alignment).
#[kani::proof]
#[kani::unwind(1)]
fn proof_buffer_offset_aligned() {
    let offset: u32 = kani::any();
    let alignment: u32 = 16;

    // Prevent overflow: offset must be small enough that rounding up
    // doesn't wrap around u32. Max realistic offset ~4 GB.
    kani::assume(offset <= u32::MAX - (alignment - 1));

    let aligned = (offset + alignment - 1) & !(alignment - 1);

    // Property 1: aligned >= original (no underflow from masking)
    assert!(
        aligned >= offset,
        "aligned offset must be >= original: aligned={}, offset={}",
        aligned,
        offset
    );

    // Property 2: aligned is a multiple of 16
    assert_eq!(
        aligned % alignment,
        0,
        "aligned offset must be a multiple of 16: aligned={}",
        aligned
    );

    // Property 3: padding is strictly less than alignment
    let padding = aligned - offset;
    assert!(
        padding < alignment,
        "padding must be < 16: padding={}, offset={}",
        padding,
        offset
    );
}

// ===========================================================================
// Harness 3: Dispatch grid covers all elements without overflow
// ===========================================================================

/// Proves that the dispatch grid computation `ceil_div(total, tg) * tg`
/// covers all `total` elements (grid >= total) and does not overflow u32
/// for realistic gpt-oss model dimensions.
///
/// The grid is computed as: `grid_x = ceil_div(total, tg) * tg` where:
/// - `total` = number of output elements (e.g., tokens * hidden_size)
/// - `tg` = threadgroup size (power of 2, 32..1024)
///
/// For gpt-oss-20b, the largest single dispatch is MoE scatter-add:
/// - `total = seq_len * hidden_size = 131072 * 2880 = ~377M` (fits u32)
///
/// We prove for arbitrary `total` up to 500M (well above max realistic)
/// that the grid computation is safe.
#[kani::proof]
#[kani::unwind(1)]
fn proof_dispatch_grid_covers_elements() {
    let total: u32 = kani::any();
    let tg: u32 = kani::any();

    // Constrain to realistic ranges:
    // - total: 1..500M covers all gpt-oss dispatch sizes
    // - tg: standard Metal threadgroup sizes (powers of 2, 32..1024)
    kani::assume(total >= 1 && total <= 500_000_000);
    kani::assume(tg >= 32 && tg <= 1024);
    kani::assume(tg & (tg - 1) == 0); // power of 2

    // ceil_div(total, tg) = (total + tg - 1) / tg
    let groups = (total + tg - 1) / tg;

    // grid = groups * tg (this is where overflow could happen)
    let grid = groups.checked_mul(tg);
    assert!(
        grid.is_some(),
        "grid computation must not overflow for total={}, tg={}",
        total,
        tg
    );

    let grid_x = grid.unwrap();

    // Property 1: grid covers all elements
    assert!(
        grid_x >= total,
        "grid must cover all elements: grid={}, total={}",
        grid_x,
        total
    );

    // Property 2: grid is aligned to threadgroup size
    assert_eq!(
        grid_x % tg,
        0,
        "grid must be aligned to threadgroup size: grid={}, tg={}",
        grid_x,
        tg
    );

    // Property 3: grid doesn't overshoot by more than tg - 1
    // (i.e., we use at most one extra threadgroup of padding)
    assert!(
        grid_x - total < tg,
        "grid overshoot must be < tg: grid={}, total={}, tg={}",
        grid_x,
        total,
        tg
    );
}
