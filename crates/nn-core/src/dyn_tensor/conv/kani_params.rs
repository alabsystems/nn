// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for convolution parameter structs (`params.rs`).
//!
//! Proves correctness of `validate()` methods on Conv1dParams, Conv2dParams,
//! ConvTranspose1dParams, and ConvTranspose2dParams. Covers:
//!
//! - Default params always pass validation
//! - Non-zero params always pass validation
//! - Zero stride/dilation/groups always rejected
//! - ConvTranspose: output_padding >= stride always rejected
//! - ConvTranspose2d: per-dimension validation correctness
//! - Round-trip: validate() accepts iff params satisfy the documented invariants

#![cfg(kani)]

use super::params::{Conv1dParams, Conv2dParams, ConvTranspose1dParams, ConvTranspose2dParams};

// ---------------------------------------------------------------------------
// Conv1dParams
// ---------------------------------------------------------------------------

/// Prove: Conv1dParams::default() always passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_params_default_valid() {
    let p = Conv1dParams::default();
    assert!(p.validate().is_ok(), "default Conv1dParams must be valid");
}

/// Prove: Conv1dParams with all non-zero fields passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_params_nonzero_valid() {
    let padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let dilation: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride >= 1);
    kani::assume(dilation >= 1);
    kani::assume(groups >= 1);

    let p = Conv1dParams {
        padding: padding as usize,
        stride: stride as usize,
        dilation: dilation as usize,
        groups: groups as usize,
    };
    assert!(
        p.validate().is_ok(),
        "Conv1dParams with non-zero stride/dilation/groups must be valid"
    );
}

/// Prove: Conv1dParams with zero stride always fails validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_params_zero_stride_rejected() {
    let padding: u8 = kani::any();
    let dilation: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(dilation >= 1);
    kani::assume(groups >= 1);

    let p = Conv1dParams {
        padding: padding as usize,
        stride: 0,
        dilation: dilation as usize,
        groups: groups as usize,
    };
    assert!(
        p.validate().is_err(),
        "Conv1dParams with stride=0 must fail validation"
    );
}

/// Prove: Conv1dParams with zero dilation always fails validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_params_zero_dilation_rejected() {
    let padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride >= 1);
    kani::assume(groups >= 1);

    let p = Conv1dParams {
        padding: padding as usize,
        stride: stride as usize,
        dilation: 0,
        groups: groups as usize,
    };
    assert!(
        p.validate().is_err(),
        "Conv1dParams with dilation=0 must fail validation"
    );
}

/// Prove: Conv1dParams with zero groups always fails validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_params_zero_groups_rejected() {
    let padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let dilation: u8 = kani::any();

    kani::assume(stride >= 1);
    kani::assume(dilation >= 1);

    let p = Conv1dParams {
        padding: padding as usize,
        stride: stride as usize,
        dilation: dilation as usize,
        groups: 0,
    };
    assert!(
        p.validate().is_err(),
        "Conv1dParams with groups=0 must fail validation"
    );
}

/// Prove: Conv1dParams validate() is an iff — passes exactly when
/// stride >= 1 AND dilation >= 1 AND groups >= 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_params_validate_iff() {
    let padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let dilation: u8 = kani::any();
    let groups: u8 = kani::any();

    let p = Conv1dParams {
        padding: padding as usize,
        stride: stride as usize,
        dilation: dilation as usize,
        groups: groups as usize,
    };

    let should_pass = stride >= 1 && dilation >= 1 && groups >= 1;
    let did_pass = p.validate().is_ok();
    assert_eq!(
        should_pass, did_pass,
        "Conv1dParams::validate must pass iff stride>0 AND dilation>0 AND groups>0"
    );
}

// ---------------------------------------------------------------------------
// Conv2dParams
// ---------------------------------------------------------------------------

/// Prove: Conv2dParams::default() always passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv2d_params_default_valid() {
    let p = Conv2dParams::default();
    assert!(p.validate().is_ok(), "default Conv2dParams must be valid");
}

/// Prove: Conv2dParams with all non-zero fields passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv2d_params_nonzero_valid() {
    let padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let dilation: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride >= 1);
    kani::assume(dilation >= 1);
    kani::assume(groups >= 1);

    let p = Conv2dParams {
        padding: padding as usize,
        stride: stride as usize,
        dilation: dilation as usize,
        groups: groups as usize,
    };
    assert!(
        p.validate().is_ok(),
        "Conv2dParams with non-zero stride/dilation/groups must be valid"
    );
}

/// Prove: Conv2dParams validate() is an iff.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv2d_params_validate_iff() {
    let padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let dilation: u8 = kani::any();
    let groups: u8 = kani::any();

    let p = Conv2dParams {
        padding: padding as usize,
        stride: stride as usize,
        dilation: dilation as usize,
        groups: groups as usize,
    };

    let should_pass = stride >= 1 && dilation >= 1 && groups >= 1;
    let did_pass = p.validate().is_ok();
    assert_eq!(
        should_pass, did_pass,
        "Conv2dParams::validate must pass iff stride>0 AND dilation>0 AND groups>0"
    );
}

// ---------------------------------------------------------------------------
// ConvTranspose1dParams
// ---------------------------------------------------------------------------

/// Prove: ConvTranspose1dParams::default() always passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose1d_params_default_valid() {
    let p = ConvTranspose1dParams::default();
    assert!(
        p.validate().is_ok(),
        "default ConvTranspose1dParams must be valid"
    );
}

/// Prove: ConvTranspose1dParams with valid fields passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose1d_params_valid_accepted() {
    let padding: u8 = kani::any();
    let output_padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let dilation: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride >= 1);
    kani::assume(dilation >= 1);
    kani::assume(groups >= 1);
    kani::assume(output_padding < stride);

    let p = ConvTranspose1dParams {
        padding: padding as usize,
        output_padding: output_padding as usize,
        stride: stride as usize,
        dilation: dilation as usize,
        groups: groups as usize,
    };
    assert!(
        p.validate().is_ok(),
        "ConvTranspose1dParams with valid fields must pass"
    );
}

/// Prove: ConvTranspose1dParams with output_padding >= stride is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose1d_params_output_padding_ge_stride_rejected() {
    let padding: u8 = kani::any();
    let output_padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let dilation: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride >= 1);
    kani::assume(dilation >= 1);
    kani::assume(groups >= 1);
    kani::assume(output_padding >= stride);

    let p = ConvTranspose1dParams {
        padding: padding as usize,
        output_padding: output_padding as usize,
        stride: stride as usize,
        dilation: dilation as usize,
        groups: groups as usize,
    };
    assert!(
        p.validate().is_err(),
        "ConvTranspose1dParams with output_padding >= stride must fail"
    );
}

/// Prove: ConvTranspose1dParams validate() iff.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose1d_params_validate_iff() {
    let padding: u8 = kani::any();
    let output_padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let dilation: u8 = kani::any();
    let groups: u8 = kani::any();

    let p = ConvTranspose1dParams {
        padding: padding as usize,
        output_padding: output_padding as usize,
        stride: stride as usize,
        dilation: dilation as usize,
        groups: groups as usize,
    };

    let should_pass = stride >= 1 && dilation >= 1 && groups >= 1 && output_padding < stride;
    let did_pass = p.validate().is_ok();
    assert_eq!(
        should_pass, did_pass,
        "ConvTranspose1dParams::validate must pass iff invariants hold"
    );
}

// ---------------------------------------------------------------------------
// ConvTranspose2dParams
// ---------------------------------------------------------------------------

/// Prove: ConvTranspose2dParams::default() always passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose2d_params_default_valid() {
    let p = ConvTranspose2dParams::default();
    assert!(
        p.validate().is_ok(),
        "default ConvTranspose2dParams must be valid"
    );
}

/// Prove: ConvTranspose2dParams with valid fields passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose2d_params_valid_accepted() {
    let pad_h: u8 = kani::any();
    let pad_w: u8 = kani::any();
    let opad_h: u8 = kani::any();
    let opad_w: u8 = kani::any();
    let stride_h: u8 = kani::any();
    let stride_w: u8 = kani::any();
    let dil_h: u8 = kani::any();
    let dil_w: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride_h >= 1);
    kani::assume(stride_w >= 1);
    kani::assume(dil_h >= 1);
    kani::assume(dil_w >= 1);
    kani::assume(groups >= 1);
    kani::assume(opad_h < stride_h);
    kani::assume(opad_w < stride_w);

    let p = ConvTranspose2dParams {
        padding: [pad_h as usize, pad_w as usize],
        output_padding: [opad_h as usize, opad_w as usize],
        stride: [stride_h as usize, stride_w as usize],
        dilation: [dil_h as usize, dil_w as usize],
        groups: groups as usize,
    };
    assert!(
        p.validate().is_ok(),
        "ConvTranspose2dParams with valid fields must pass"
    );
}

/// Prove: ConvTranspose2dParams with zero stride_h is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose2d_params_zero_stride_h_rejected() {
    let stride_w: u8 = kani::any();
    let dil_h: u8 = kani::any();
    let dil_w: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride_w >= 1);
    kani::assume(dil_h >= 1);
    kani::assume(dil_w >= 1);
    kani::assume(groups >= 1);

    let p = ConvTranspose2dParams {
        padding: [0, 0],
        output_padding: [0, 0],
        stride: [0, stride_w as usize],
        dilation: [dil_h as usize, dil_w as usize],
        groups: groups as usize,
    };
    assert!(
        p.validate().is_err(),
        "ConvTranspose2dParams with stride[0]=0 must fail"
    );
}

/// Prove: ConvTranspose2dParams with zero stride_w is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose2d_params_zero_stride_w_rejected() {
    let stride_h: u8 = kani::any();
    let dil_h: u8 = kani::any();
    let dil_w: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride_h >= 1);
    kani::assume(dil_h >= 1);
    kani::assume(dil_w >= 1);
    kani::assume(groups >= 1);

    let p = ConvTranspose2dParams {
        padding: [0, 0],
        output_padding: [0, 0],
        stride: [stride_h as usize, 0],
        dilation: [dil_h as usize, dil_w as usize],
        groups: groups as usize,
    };
    assert!(
        p.validate().is_err(),
        "ConvTranspose2dParams with stride[1]=0 must fail"
    );
}

/// Prove: ConvTranspose2dParams with output_padding[0] >= stride[0] is rejected
/// (even when dimension 1 is valid).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose2d_params_opad_h_ge_stride_h_rejected() {
    let opad_h: u8 = kani::any();
    let stride_h: u8 = kani::any();
    let stride_w: u8 = kani::any();
    let dil_h: u8 = kani::any();
    let dil_w: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride_h >= 1);
    kani::assume(stride_w >= 1);
    kani::assume(dil_h >= 1);
    kani::assume(dil_w >= 1);
    kani::assume(groups >= 1);
    kani::assume(opad_h >= stride_h);

    let p = ConvTranspose2dParams {
        padding: [0, 0],
        output_padding: [opad_h as usize, 0],
        stride: [stride_h as usize, stride_w as usize],
        dilation: [dil_h as usize, dil_w as usize],
        groups: groups as usize,
    };
    assert!(
        p.validate().is_err(),
        "ConvTranspose2dParams with output_padding[0] >= stride[0] must fail"
    );
}

/// Prove: ConvTranspose2dParams with output_padding[1] >= stride[1] is rejected
/// (even when dimension 0 is valid).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose2d_params_opad_w_ge_stride_w_rejected() {
    let opad_w: u8 = kani::any();
    let stride_h: u8 = kani::any();
    let stride_w: u8 = kani::any();
    let dil_h: u8 = kani::any();
    let dil_w: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(stride_h >= 1);
    kani::assume(stride_w >= 1);
    kani::assume(dil_h >= 1);
    kani::assume(dil_w >= 1);
    kani::assume(groups >= 1);
    kani::assume(opad_w >= stride_w);

    let p = ConvTranspose2dParams {
        padding: [0, 0],
        output_padding: [0, opad_w as usize],
        stride: [stride_h as usize, stride_w as usize],
        dilation: [dil_h as usize, dil_w as usize],
        groups: groups as usize,
    };
    assert!(
        p.validate().is_err(),
        "ConvTranspose2dParams with output_padding[1] >= stride[1] must fail"
    );
}

/// Prove: ConvTranspose2dParams validate() iff all invariants hold.
///
/// This is the strongest harness: exhaustive characterization of validate().
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose2d_params_validate_iff() {
    let pad_h: u8 = kani::any();
    let pad_w: u8 = kani::any();
    let opad_h: u8 = kani::any();
    let opad_w: u8 = kani::any();
    let stride_h: u8 = kani::any();
    let stride_w: u8 = kani::any();
    let dil_h: u8 = kani::any();
    let dil_w: u8 = kani::any();
    let groups: u8 = kani::any();

    let p = ConvTranspose2dParams {
        padding: [pad_h as usize, pad_w as usize],
        output_padding: [opad_h as usize, opad_w as usize],
        stride: [stride_h as usize, stride_w as usize],
        dilation: [dil_h as usize, dil_w as usize],
        groups: groups as usize,
    };

    let should_pass = stride_h >= 1
        && stride_w >= 1
        && dil_h >= 1
        && dil_w >= 1
        && groups >= 1
        && opad_h < stride_h
        && opad_w < stride_w;
    let did_pass = p.validate().is_ok();
    assert_eq!(
        should_pass, did_pass,
        "ConvTranspose2dParams::validate must pass iff all invariants hold"
    );
}
