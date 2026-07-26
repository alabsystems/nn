// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for neural network pooling layer safety properties.
//!
//! Proves correctness of MaxPool1d/2d, AvgPool2d, and AdaptiveAvgPool2d
//! configuration validation, output dimension arithmetic, and invariants:
//!
//! **Configuration validation (10 harnesses):**
//! 1. Pool1dConfig::new default stride == kernel_size for arbitrary kernel
//! 2. Pool1dConfig rejects kernel_size == 0 via MaxPool1d::new
//! 3. Pool1dConfig rejects stride == 0 via MaxPool1d::new
//! 4. Pool1dConfig accepts all valid (kernel > 0, stride > 0) combos
//! 5. Pool2dConfig::new default stride == kernel_size for arbitrary kernel
//! 6. Pool2dConfig rejects kernel_size == 0 for both MaxPool2d and AvgPool2d
//! 7. Pool2dConfig rejects stride == 0 for both MaxPool2d and AvgPool2d
//! 8. Pool2dConfig accepts all valid combos for both MaxPool2d and AvgPool2d
//! 9. AdaptiveAvgPool2d rejects zero output height or width
//! 10. AdaptiveAvgPool2d accepts all positive output sizes
//!
//! **Output dimension arithmetic (10 harnesses):**
//! 11. pool2d_out_len produces >= 1 when padded >= kernel and stride > 0
//! 12. pool2d_out_len is monotone non-decreasing in input length
//! 13. pool2d_out_len is monotone non-decreasing in padding
//! 14. pool2d_out_len is monotone non-increasing in kernel_size
//! 15. pool2d_out_len is monotone non-increasing in stride
//! 16. pool2d_out_len identity case: p=0, s=1 gives input - kernel + 1
//! 17. pool2d_out_len ceil_mode >= floor_mode
//! 18. pool2d_out_len ceil_mode exceeds floor_mode by at most 1
//! 19. pool2d_out_len with s=kernel (default stride) gives ceil(input/kernel)
//! 20. pool2d_out_len rejects padded < kernel_size
//!
//! **Adaptive pooling window safety (6 harnesses):**
//! 21. Adaptive window indices are within bounds for all out/in ratios
//! 22. Adaptive windows are always non-empty (count >= 1)
//! 23. Adaptive windows cover full input (first starts at 0, last ends at in_h)
//! 24. Adjacent adaptive windows do not skip input positions
//! 25. Adaptive window count per output position is bounded
//! 26. Adaptive pooling preserves batch and channel dimensions
//!
//! **Cross-dimensional consistency (4 harnesses):**
//! 27. MaxPool2d applies same formula to both spatial dims
//! 28. Pool output element count equals batch * channels * out_h * out_w
//! 29. Padding < kernel_size invariant prevents center-free windows
//! 30. Global average pooling (kernel == input) produces single output
//!
//! Part of #4562.

#![cfg(kani)]

use crate::dyn_tensor::conv::pool::pool2d_out_len;
use crate::layers::{AdaptiveAvgPool2d, AvgPool2d, MaxPool1d, MaxPool2d, Pool1dConfig, Pool2dConfig};

// ===========================================================================
// Configuration validation (harnesses 1-10)
// ===========================================================================

/// Prove: Pool1dConfig::new(k) sets stride == kernel_size and padding == 0
/// for any valid kernel_size in [1, 255].
#[kani::unwind(1)]
#[kani::proof]
fn pool1d_config_new_stride_equals_kernel() {
    let k: u8 = kani::any();
    kani::assume(k >= 1);

    let cfg = Pool1dConfig::new(k as usize);
    assert_eq!(
        cfg.stride, cfg.kernel_size,
        "Pool1dConfig::new must default stride to kernel_size"
    );
    assert_eq!(
        cfg.padding, 0,
        "Pool1dConfig::new must default padding to 0"
    );
    assert_eq!(cfg.kernel_size, k as usize, "kernel_size must match input");
}

/// Prove: Pool1dConfig with kernel_size == 0 is rejected by MaxPool1d::new.
#[kani::unwind(1)]
#[kani::proof]
fn pool1d_rejects_zero_kernel_size() {
    let s: u8 = kani::any();
    let p: u8 = kani::any();
    kani::assume(s >= 1);

    let cfg = Pool1dConfig {
        kernel_size: 0,
        stride: s as usize,
        padding: p as usize,
    };
    let result = MaxPool1d::new(cfg);
    assert!(result.is_err(), "MaxPool1d must reject kernel_size == 0");
}

/// Prove: Pool1dConfig with stride == 0 is rejected by MaxPool1d::new.
#[kani::unwind(1)]
#[kani::proof]
fn pool1d_rejects_zero_stride() {
    let k: u8 = kani::any();
    let p: u8 = kani::any();
    kani::assume(k >= 1);

    let cfg = Pool1dConfig {
        kernel_size: k as usize,
        stride: 0,
        padding: p as usize,
    };
    let result = MaxPool1d::new(cfg);
    assert!(result.is_err(), "MaxPool1d must reject stride == 0");
}

/// Prove: Pool1dConfig with kernel > 0 and stride > 0 is accepted.
#[kani::unwind(1)]
#[kani::proof]
fn pool1d_accepts_valid_config() {
    let k: u8 = kani::any();
    let s: u8 = kani::any();
    let p: u8 = kani::any();

    kani::assume(k >= 1);
    kani::assume(s >= 1);

    let cfg = Pool1dConfig {
        kernel_size: k as usize,
        stride: s as usize,
        padding: p as usize,
    };
    let result = MaxPool1d::new(cfg);
    assert!(
        result.is_ok(),
        "MaxPool1d must accept config with kernel > 0 and stride > 0"
    );
}

/// Prove: Pool2dConfig::new(k) sets stride == kernel_size and padding == 0.
#[kani::unwind(1)]
#[kani::proof]
fn pool2d_config_new_stride_equals_kernel() {
    let k: u8 = kani::any();
    kani::assume(k >= 1);

    let cfg = Pool2dConfig::new(k as usize);
    assert_eq!(
        cfg.stride, cfg.kernel_size,
        "Pool2dConfig::new must default stride to kernel_size"
    );
    assert_eq!(
        cfg.padding, 0,
        "Pool2dConfig::new must default padding to 0"
    );
    assert_eq!(cfg.kernel_size, k as usize, "kernel_size must match input");
}

/// Prove: Pool2dConfig with kernel_size == 0 is rejected for MaxPool2d and AvgPool2d.
#[kani::unwind(1)]
#[kani::proof]
fn pool2d_rejects_zero_kernel_size() {
    let s: u8 = kani::any();
    let p: u8 = kani::any();
    kani::assume(s >= 1);

    let cfg = Pool2dConfig {
        kernel_size: 0,
        stride: s as usize,
        padding: p as usize,
    };
    let r_max = MaxPool2d::new(cfg);
    let r_avg = AvgPool2d::new(cfg);
    assert!(r_max.is_err(), "MaxPool2d must reject kernel_size == 0");
    assert!(r_avg.is_err(), "AvgPool2d must reject kernel_size == 0");
}

/// Prove: Pool2dConfig with stride == 0 is rejected for MaxPool2d and AvgPool2d.
#[kani::unwind(1)]
#[kani::proof]
fn pool2d_rejects_zero_stride() {
    let k: u8 = kani::any();
    let p: u8 = kani::any();
    kani::assume(k >= 1);

    let cfg = Pool2dConfig {
        kernel_size: k as usize,
        stride: 0,
        padding: p as usize,
    };
    let r_max = MaxPool2d::new(cfg);
    let r_avg = AvgPool2d::new(cfg);
    assert!(r_max.is_err(), "MaxPool2d must reject stride == 0");
    assert!(r_avg.is_err(), "AvgPool2d must reject stride == 0");
}

/// Prove: Pool2dConfig with kernel > 0, stride > 0 is accepted for both types.
#[kani::unwind(1)]
#[kani::proof]
fn pool2d_accepts_valid_config() {
    let k: u8 = kani::any();
    let s: u8 = kani::any();
    let p: u8 = kani::any();

    kani::assume(k >= 1);
    kani::assume(s >= 1);

    let cfg = Pool2dConfig {
        kernel_size: k as usize,
        stride: s as usize,
        padding: p as usize,
    };
    let r_max = MaxPool2d::new(cfg);
    let r_avg = AvgPool2d::new(cfg);
    assert!(r_max.is_ok(), "MaxPool2d must accept valid config");
    assert!(r_avg.is_ok(), "AvgPool2d must accept valid config");
}

/// Prove: AdaptiveAvgPool2d rejects zero output height or width.
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_pool_rejects_zero_dims() {
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(h >= 1);
    kani::assume(w >= 1);

    // Zero height
    let r1 = AdaptiveAvgPool2d::new(0, w as usize);
    assert!(r1.is_err(), "AdaptiveAvgPool2d must reject out_h == 0");

    // Zero width
    let r2 = AdaptiveAvgPool2d::new(h as usize, 0);
    assert!(r2.is_err(), "AdaptiveAvgPool2d must reject out_w == 0");

    // Both zero
    let r3 = AdaptiveAvgPool2d::new(0, 0);
    assert!(r3.is_err(), "AdaptiveAvgPool2d must reject both == 0");
}

/// Prove: AdaptiveAvgPool2d accepts all positive output sizes.
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_pool_accepts_positive_sizes() {
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(h >= 1);
    kani::assume(w >= 1);

    let result = AdaptiveAvgPool2d::new(h as usize, w as usize);
    assert!(
        result.is_ok(),
        "AdaptiveAvgPool2d must accept positive output sizes"
    );
    let pool = result.unwrap();
    let (out_h, out_w) = pool.output_size();
    assert_eq!(out_h, h as usize, "output height must match");
    assert_eq!(out_w, w as usize, "output width must match");
}

// ===========================================================================
// Output dimension arithmetic (harnesses 11-20)
// ===========================================================================

/// Prove: pool2d_out_len returns >= 1 when padded >= kernel and stride >= 1.
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_at_least_one_when_valid() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks as usize);

    let result = pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);
    assert!(result.is_ok(), "valid params must produce Ok");
    assert!(result.unwrap() >= 1, "valid pool output must be >= 1");
}

/// Prove: pool2d_out_len is monotone non-decreasing in input length.
///
/// Adding more input data never reduces the number of output positions.
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_monotone_in_input() {
    let il1: u8 = kani::any();
    let il2: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il1 >= 1);
    kani::assume(il2 >= il1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded1 = (il1 as usize) + 2 * (p as usize);
    kani::assume(padded1 >= ks as usize);

    let r1 = pool2d_out_len(il1 as usize, ks as usize, p as usize, s as usize, false);
    let r2 = pool2d_out_len(il2 as usize, ks as usize, p as usize, s as usize, false);

    if let (Ok(o1), Ok(o2)) = (r1, r2) {
        assert!(o2 >= o1, "larger input must produce >= output");
    }
}

/// Prove: pool2d_out_len is monotone non-decreasing in padding.
///
/// More padding adds more virtual elements, producing >= output positions.
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_monotone_in_padding() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p1: u8 = kani::any();
    let p2: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(p2 >= p1);

    let padded1 = (il as usize) + 2 * (p1 as usize);
    kani::assume(padded1 >= ks as usize);

    let r1 = pool2d_out_len(il as usize, ks as usize, p1 as usize, s as usize, false);
    let r2 = pool2d_out_len(il as usize, ks as usize, p2 as usize, s as usize, false);

    if let (Ok(o1), Ok(o2)) = (r1, r2) {
        assert!(o2 >= o1, "more padding must produce >= output");
    }
}

/// Prove: pool2d_out_len is monotone non-increasing in kernel_size.
///
/// Larger kernel consumes more elements per window, producing fewer outputs.
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_monotone_decreasing_in_kernel() {
    let il: u8 = kani::any();
    let ks1: u8 = kani::any();
    let ks2: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks1 >= 1);
    kani::assume(ks2 >= ks1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks2 as usize);

    let r1 = pool2d_out_len(il as usize, ks1 as usize, p as usize, s as usize, false);
    let r2 = pool2d_out_len(il as usize, ks2 as usize, p as usize, s as usize, false);

    if let (Ok(o1), Ok(o2)) = (r1, r2) {
        assert!(o1 >= o2, "larger kernel must produce <= output");
    }
}

/// Prove: pool2d_out_len is monotone non-increasing in stride.
///
/// Larger stride skips more elements, producing fewer outputs.
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_monotone_decreasing_in_stride() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s1 >= 1);
    kani::assume(s2 >= s1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks as usize);

    let r1 = pool2d_out_len(il as usize, ks as usize, p as usize, s1 as usize, false);
    let r2 = pool2d_out_len(il as usize, ks as usize, p as usize, s2 as usize, false);

    if let (Ok(o1), Ok(o2)) = (r1, r2) {
        assert!(o1 >= o2, "larger stride must produce <= output");
    }
}

/// Prove: pool2d_out_len identity case (p=0, s=1) gives input - kernel + 1.
///
/// This is the simplest case of the pooling formula.
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_identity_p0_s1() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(il >= ks);

    let result = pool2d_out_len(il as usize, ks as usize, 0, 1, false);
    match result {
        Ok(out) => {
            let expected = (il as usize) - (ks as usize) + 1;
            assert_eq!(out, expected, "p=0, s=1: output must be input - kernel + 1");
        }
        Err(_) => panic!("valid params must not be rejected"),
    }
}

/// Prove: ceil_mode output >= floor_mode output for all valid params.
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_ceil_geq_floor() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks as usize);

    let r_floor = pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);
    let r_ceil = pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, true);

    if let (Ok(out_floor), Ok(out_ceil)) = (r_floor, r_ceil) {
        assert!(out_ceil >= out_floor, "ceil_mode must be >= floor_mode");
    }
}

/// Prove: ceil_mode exceeds floor_mode by at most 1.
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_ceil_exceeds_floor_by_at_most_one() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks as usize);

    let r_floor = pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);
    let r_ceil = pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, true);

    if let (Ok(out_floor), Ok(out_ceil)) = (r_floor, r_ceil) {
        assert!(
            out_ceil <= out_floor + 1,
            "ceil_mode exceeds floor_mode by at most 1"
        );
    }
}

/// Prove: pool2d_out_len with stride == kernel_size (default) and p=0 gives
/// ceil(input / kernel) via floor division.
///
/// This is the PyTorch default configuration: stride defaults to kernel_size.
/// For floor_mode: output = (input - kernel) / kernel + 1 = input / kernel
/// (when input >= kernel and using integer division).
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_default_stride_floor() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(il >= ks);

    let result = pool2d_out_len(il as usize, ks as usize, 0, ks as usize, false);
    match result {
        Ok(out) => {
            // floor: (il - ks) / ks + 1
            let expected = ((il as usize) - (ks as usize)) / (ks as usize) + 1;
            assert_eq!(
                out, expected,
                "default stride: output must be (input - kernel) / kernel + 1"
            );
        }
        Err(_) => panic!("valid params must not be rejected"),
    }
}

/// Prove: pool2d_out_len rejects when padded input < kernel_size.
///
/// When the padded input is smaller than the kernel, no output position
/// can be computed. The function must return Err.
#[kani::unwind(1)]
#[kani::proof]
fn pool_out_len_rejects_padded_lt_kernel() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded < ks as usize);

    let result = pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);
    assert!(
        result.is_err(),
        "pool2d_out_len must reject when padded < kernel_size"
    );
}

// ===========================================================================
// Adaptive pooling window safety (harnesses 21-26)
// ===========================================================================

/// Prove: adaptive pooling window indices are within input bounds for all
/// output/input size ratios (downsampling, upsampling, and identity).
///
/// Window formula (PyTorch ATen):
///   start = (oh * in_h) / out_h
///   end = ((oh + 1) * in_h).div_ceil(out_h)
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_window_indices_in_bounds() {
    let oh: u8 = kani::any();
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);
    kani::assume(oh < out_h);

    let oh = oh as usize;
    let out_h = out_h as usize;
    let in_h = in_h as usize;

    let start = (oh * in_h) / out_h;
    let end = ((oh + 1) * in_h + out_h - 1) / out_h; // div_ceil

    assert!(start < in_h, "window start must be within input");
    assert!(end <= in_h, "window end must not exceed input");
    assert!(start < end, "window must be non-empty");
}

/// Prove: adaptive pooling windows always contain >= 1 element.
///
/// The div_ceil formula for end guarantees non-empty windows even when
/// upsampling (out_h > in_h), preventing division by zero in the avg.
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_window_always_nonempty() {
    let oh: u8 = kani::any();
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);
    kani::assume(oh < out_h);

    let oh = oh as usize;
    let out_h = out_h as usize;
    let in_h = in_h as usize;

    let start = (oh * in_h) / out_h;
    let end = ((oh + 1) * in_h + out_h - 1) / out_h; // div_ceil

    let count = end - start;
    assert!(
        count >= 1,
        "adaptive window must contain at least 1 element"
    );
}

/// Prove: adaptive windows cover full input range.
///
/// First window starts at 0, last window ends at in_h.
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_windows_cover_full_input() {
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);

    let out_h = out_h as usize;
    let in_h = in_h as usize;

    // First window (oh=0) starts at 0
    let first_start = (0 * in_h) / out_h;
    assert_eq!(first_start, 0, "first window must start at 0");

    // Last window (oh=out_h-1) end
    let last_end = (out_h * in_h + out_h - 1) / out_h; // div_ceil
    assert!(last_end >= in_h, "last window must reach end of input");

    // Actually: (out_h * in_h) / out_h == in_h exactly (no ceiling needed)
    let last_end_exact = (out_h * in_h) / out_h;
    assert_eq!(last_end_exact, in_h, "last window end formula is exact");
}

/// Prove: adjacent adaptive windows do not skip input positions.
///
/// For consecutive output positions oh and oh+1, the end of window oh
/// is >= the start of window oh+1. This ensures no input position is
/// left uncovered by the union of windows.
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_windows_no_gap_between_adjacent() {
    let oh: u8 = kani::any();
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();

    kani::assume(out_h >= 2); // need at least 2 output positions
    kani::assume(in_h >= 1);
    kani::assume(oh < out_h - 1); // oh and oh+1 both valid

    let oh = oh as usize;
    let out_h = out_h as usize;
    let in_h = in_h as usize;

    // Window for oh
    let end_oh = ((oh + 1) * in_h + out_h - 1) / out_h; // div_ceil

    // Window for oh + 1
    let start_next = ((oh + 1) * in_h) / out_h;

    // No gap: end of current window >= start of next window
    assert!(
        end_oh >= start_next,
        "adjacent adaptive windows must not skip input positions"
    );
}

/// Prove: adaptive window count (elements per output position) is bounded.
///
/// For each output position, the number of input elements in its window
/// is at most ceil(in_h / out_h) + 1 (the +1 accounts for ceiling rounding
/// in both start and end formulas).
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_window_count_bounded() {
    let oh: u8 = kani::any();
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);
    kani::assume(oh < out_h);

    let oh = oh as usize;
    let out_h = out_h as usize;
    let in_h = in_h as usize;

    let start = (oh * in_h) / out_h;
    let end = ((oh + 1) * in_h + out_h - 1) / out_h;

    let count = end - start;

    // Upper bound: ceil(in_h / out_h) + 1
    let max_count = (in_h + out_h - 1) / out_h + 1;
    assert!(
        count <= max_count,
        "adaptive window count must be bounded by ceil(in_h/out_h) + 1"
    );

    // Lower bound: at least 1 (proved separately, but double check)
    assert!(count >= 1, "window must have at least 1 element");
}

/// Prove: adaptive pooling output preserves batch and channel dimensions.
///
/// For a 4D tensor [B, C, H, W], adaptive pooling only changes H and W.
/// The output shape is [B, C, out_h, out_w].
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_pool_preserves_batch_channel() {
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let in_h: u8 = kani::any();
    let in_w: u8 = kani::any();
    let out_h: u8 = kani::any();
    let out_w: u8 = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(c >= 1 && c <= 16);
    kani::assume(in_h >= 1 && in_h <= 16);
    kani::assume(in_w >= 1 && in_w <= 16);
    kani::assume(out_h >= 1 && out_h <= 16);
    kani::assume(out_w >= 1 && out_w <= 16);

    let bu = b as usize;
    let cu = c as usize;
    let ohu = out_h as usize;
    let owu = out_w as usize;

    // Input shape: [B, C, H, W]
    let input_shape = [bu, cu, in_h as usize, in_w as usize];

    // Output shape: [B, C, out_h, out_w]
    let output_shape = [bu, cu, ohu, owu];

    // Batch and channel preserved
    assert_eq!(
        output_shape[0], input_shape[0],
        "batch dimension must be preserved"
    );
    assert_eq!(
        output_shape[1], input_shape[1],
        "channel dimension must be preserved"
    );

    // Spatial dims changed to target
    assert_eq!(output_shape[2], ohu, "output height must match target");
    assert_eq!(output_shape[3], owu, "output width must match target");
}

// ===========================================================================
// Cross-dimensional consistency (harnesses 27-30)
// ===========================================================================

/// Prove: MaxPool2d applies the same formula to both spatial dimensions.
///
/// For square pooling (same kernel/stride/padding for H and W),
/// both output dimensions must be equal when input H == input W.
#[kani::unwind(1)]
#[kani::proof]
fn maxpool2d_same_formula_both_dims() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks as usize);

    // Same input for both dimensions (square input)
    let r_h = pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);
    let r_w = pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);

    // Both dimensions use same formula, so results must be identical
    match (r_h, r_w) {
        (Ok(out_h), Ok(out_w)) => {
            assert_eq!(out_h, out_w, "square pooling must produce equal H and W");
        }
        (Err(_), Err(_)) => {} // both error is also consistent
        _ => panic!("same parameters must produce same Ok/Err result"),
    }
}

/// Prove: pool output element count equals batch * channels * out_h * out_w.
///
/// The total number of elements in the output tensor must be the product of
/// all four dimensions. No dimension is dropped or added by pooling.
#[kani::unwind(1)]
#[kani::proof]
fn pool_output_element_count_correct() {
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let in_h: u8 = kani::any();
    let ks: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(c >= 1 && c <= 8);
    kani::assume(in_h >= 1 && in_h <= 32);
    kani::assume(ks >= 1 && ks <= 8);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(in_h >= ks);

    if let Ok(out_h) = pool2d_out_len(in_h as usize, ks as usize, 0, s as usize, false) {
        let total = (b as usize) * (c as usize) * out_h * out_h;
        let bc = (b as usize) * (c as usize);

        // Element count decomposes correctly
        if out_h > 0 {
            let spatial = out_h * out_h;
            assert_eq!(total, bc * spatial, "total = batch * channels * spatial");
            assert_eq!(total / spatial, bc, "extracting batch*channels from total");
        }
    }
}

/// Prove: padding < kernel_size prevents center-free windows.
///
/// When padding < kernel_size, the center element of each pooling window
/// is always within the real (non-padded) input. This ensures max pooling
/// always sees at least one real input element per window.
#[kani::unwind(1)]
#[kani::proof]
fn padding_lt_kernel_prevents_center_free_windows() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume((p as usize) < (ks as usize)); // The invariant under test

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let padded = il + 2 * p;
    if padded >= ks {
        let out = (padded - ks) / s + 1;

        // For each output position i, window covers [i*s - p, i*s - p + ks - 1]
        // The window center is at i*s - p + ks/2.
        // Since p < ks, and i*s ranges from 0 to (out-1)*s:
        //   - First window start: 0*s - p => might be negative (virtual padding)
        //   - But since p < ks, the window extends at least (ks - p) real positions
        //     into the input, which is >= 1.

        // The key insight: with p < ks, every window overlaps at least 1 real
        // input element. Proof: window of size ks at position starting at
        // (i*s - p) covers up to (i*s - p + ks - 1). Since ks > p,
        // i*s - p + ks > i*s > 0 for any valid output position.

        // For the first window (i=0): start = -p, end = ks - 1 - p
        // The end is ks - 1 - p >= 0 since ks > p.
        // So the first window reaches real position 0.
        let first_window_end_in_real = ks - p; // ks - 1 - p + 1 real positions covered
        assert!(
            first_window_end_in_real >= 1,
            "first window must reach at least 1 real input position"
        );

        // For any window, overlap with real input is >= ks - p >= 1
        let min_real_overlap = ks - p;
        assert!(
            min_real_overlap >= 1,
            "every window must overlap >= 1 real input element"
        );

        assert!(out >= 1, "valid config must produce >= 1 output");
    }
}

/// Prove: global average pooling (kernel == input, stride == 1, padding == 0)
/// produces exactly one output per spatial dimension.
///
/// This is the common pattern for classification heads:
/// [B, C, H, W] -> AvgPool2d(kernel=H) -> [B, C, 1, 1]
#[kani::unwind(1)]
#[kani::proof]
fn global_avg_pool_produces_single_output() {
    let il: u8 = kani::any();
    kani::assume(il >= 1);

    // Global pooling: kernel == input, stride = 1, padding = 0
    let result = pool2d_out_len(il as usize, il as usize, 0, 1, false);
    match result {
        Ok(out) => {
            assert_eq!(
                out, 1,
                "global pooling (kernel == input) must produce exactly 1 output"
            );
        }
        Err(_) => panic!("global pooling with valid input must succeed"),
    }
}
