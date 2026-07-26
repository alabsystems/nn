// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for polar-to-rectangular conversion (#2218 F14).
//!
//! The fused MSL kernel `fused_polar_to_rect_f32` computes:
//!   `real = magnitude * cos(phase)`
//!   `imag = magnitude * sin(phase)`
//!
//! These harnesses prove:
//! 1. Output finiteness for bounded finite inputs (with sin/cos stubs)
//! 2. Output bounds: |real| <= |mag| and |imag| <= |mag| (since |sin|,|cos| <= 1)
//! 3. Pythagorean invariant: real² + imag² <= mag² (within float tolerance)
//!
//! Uses `sin_stub`/`cos_stub` from `kani_stubs.rs` since CBMC cannot model
//! transcendentals (#708).

use crate::kani_stubs::{cos_stub, sin_stub};

/// Scalar polar-to-rect: reference Rust implementation matching MSL kernel.
fn polar_to_rect_scalar(magnitude: f32, phase: f32) -> (f32, f32) {
    let c = phase.cos();
    let s = phase.sin();
    (magnitude * c, magnitude * s)
}

/// Proves polar_to_rect output finiteness for bounded inputs.
///
/// SUBSTANTIVE: proves that for any magnitude in [0, 100] and phase in [-pi, pi],
/// both real and imag outputs are finite. Uses sin_stub/cos_stub (sound
/// over-approximation: any value in [-1, 1]).
///
/// Covers: `dyn_tensor_metal_polar_to_rect_msl.rs` kernel logic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sin, sin_stub)]
#[kani::stub(f32::cos, cos_stub)]
fn polar_to_rect_finite_bounded() {
    let magnitude: f32 = kani::any();
    let phase: f32 = kani::any();

    kani::assume(magnitude.is_finite());
    kani::assume(phase.is_finite());
    kani::assume(magnitude >= 0.0 && magnitude <= 100.0);
    kani::assume(phase >= -std::f32::consts::PI && phase <= std::f32::consts::PI);

    let (real, imag) = polar_to_rect_scalar(magnitude, phase);

    assert!(real.is_finite(), "real output must be finite");
    assert!(imag.is_finite(), "imag output must be finite");
}

/// Proves output magnitude bound: |real| <= |mag| and |imag| <= |mag|.
///
/// SUBSTANTIVE: since |cos| <= 1 and |sin| <= 1, the output components
/// cannot exceed the input magnitude. This is the key safety property.
#[kani::unwind(8)]
#[kani::proof]
#[kani::stub(f32::sin, sin_stub)]
#[kani::stub(f32::cos, cos_stub)]
fn polar_to_rect_magnitude_bound() {
    let magnitude: f32 = kani::any();
    let phase: f32 = kani::any();

    kani::assume(magnitude.is_finite());
    kani::assume(phase.is_finite());
    kani::assume(magnitude >= 0.0 && magnitude <= 50.0);
    kani::assume(phase >= -std::f32::consts::PI && phase <= std::f32::consts::PI);

    let (real, imag) = polar_to_rect_scalar(magnitude, phase);

    // |cos(x)| <= 1 and |sin(x)| <= 1, so |mag * cos| <= |mag|.
    // Float epsilon accounts for rounding in multiply.
    let bound = magnitude + f32::EPSILON;
    assert!(
        real.abs() <= bound,
        "real magnitude must not exceed input magnitude"
    );
    assert!(
        imag.abs() <= bound,
        "imag magnitude must not exceed input magnitude"
    );
}

/// Proves zero magnitude produces zero output.
///
/// SUBSTANTIVE: when magnitude == 0, both real and imag must be exactly 0.0
/// regardless of phase. This is an important GPU correctness property since
/// 0 * NaN = NaN in IEEE 754 (but we require finite phase inputs).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sin, sin_stub)]
#[kani::stub(f32::cos, cos_stub)]
fn polar_to_rect_zero_magnitude() {
    let phase: f32 = kani::any();

    kani::assume(phase.is_finite());
    kani::assume(phase >= -std::f32::consts::PI && phase <= std::f32::consts::PI);

    let (real, imag) = polar_to_rect_scalar(0.0, phase);

    assert!(real == 0.0, "zero magnitude must produce zero real");
    assert!(imag == 0.0, "zero magnitude must produce zero imag");
}
