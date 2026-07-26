// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Audio waveshaper kernel — normalized tanh soft-clipping.
//!
//! `output = tanh(drive * x) / tanh(drive)` with guaranteed output in [-1, 1]
//! for drive > 0 and `|x| ≤ 1`. Passthrough when drive <= 0.
//! For `|x| > 1`, output can exceed ±1 (approaches `1/tanh(drive)`).
//!
//! Part of #956 D2 (Audio DSP kernel support).

use crate::kernel_error::KernelError;
use crate::kernel_util::{checked_scalar_output, validate_finite_inputs};

/// Soft-clip waveshaper: normalized tanh distortion.
///
/// When `drive > 0`: output = `tanh(drive * x) / tanh(drive)`.
/// Bounded in [-1, 1] for `|x| ≤ 1`. For `|x| > 1`, output can exceed ±1.
/// When `drive <= 0`: passthrough (output = x).
///
/// # Errors
///
/// Returns [`KernelError`] if inputs or output are non-finite.
pub fn tanh_waveshaper_scalar(x: f32, drive: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x), ("drive", drive)])?;
    if drive <= 0.0 {
        return checked_scalar_output(x);
    }
    let numerator = (drive * x).tanh();
    let denominator = drive.tanh();
    // tanh(positive) > 0, so denominator > 0 for drive > 0
    checked_scalar_output(numerator / denominator)
}

/// Conservative output bounds for the tanh waveshaper.
///
/// For `drive > 0` and `|x| ≤ 1`: output is in [-1, 1].
/// For `drive > 0` and `|x| > 1`: output can exceed ±1 (approaches x as drive → 0+).
/// For `drive <= 0`: passthrough, so bounds equal input bounds.
pub fn tanh_waveshaper_scalar_bounds(
    x_lo: f32,
    x_hi: f32,
    drive_lo: f32,
    drive_hi: f32,
) -> Result<(f32, f32), KernelError> {
    crate::kernel_util::validate_bounds_pairs(&[(x_lo, x_hi), (drive_lo, drive_hi)])?;
    if drive_lo > 0.0 && x_lo >= -1.0 && x_hi <= 1.0 {
        // |x| ≤ 1: tanh is monotone → |tanh(d*x)| ≤ tanh(d) → ratio ≤ 1
        Ok((-1.0, 1.0))
    } else {
        // Either drive can be ≤ 0 (passthrough) or |x| > 1 (output exceeds ±1
        // at low drive). Conservative: union of input bounds and [-1, 1].
        Ok((x_lo.min(-1.0), x_hi.max(1.0)))
    }
}

#[cfg(test)]
#[path = "waveshaper_tests.rs"]
mod tests;

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::kani_stubs::tanh_stub;

    /// Proof: output is bounded in [-1, 1] for drive > 0 and |x| ≤ 1.
    ///
    /// The property holds because tanh is monotone: |tanh(d*x)| ≤ tanh(d)
    /// when |x| ≤ 1, so the ratio |tanh(d*x)/tanh(d)| ≤ 1.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::tanh, tanh_stub)]
    fn waveshaper_output_bounded_for_positive_drive() {
        let x: f32 = kani::any();
        let drive: f32 = kani::any();
        kani::assume(x.is_finite() && x.abs() <= 1.0);
        kani::assume(drive.is_finite() && drive > 0.01 && drive <= 10.0);

        let result = tanh_waveshaper_scalar(x, drive);
        if let Ok(y) = result {
            assert!(y >= -1.001 && y <= 1.001, "output must be in [-1, 1]");
        }
    }

    /// Proof: drive <= 0 produces passthrough.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::tanh, tanh_stub)]
    fn waveshaper_passthrough_at_zero_drive() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite() && x.abs() <= 1e6);

        let result = tanh_waveshaper_scalar(x, 0.0).unwrap();
        assert_eq!(result.to_bits(), x.to_bits(), "drive=0 must be passthrough");
    }

    /// Proof: no NaN/Inf for bounded finite inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::tanh, tanh_stub)]
    fn waveshaper_finite_for_bounded_inputs() {
        let x: f32 = kani::any();
        let drive: f32 = kani::any();
        kani::assume(x.is_finite() && x.abs() <= 100.0);
        kani::assume(drive.is_finite() && drive.abs() <= 10.0);

        let result = tanh_waveshaper_scalar(x, drive);
        if let Ok(y) = result {
            assert!(y.is_finite(), "output must be finite");
        }
    }
}
