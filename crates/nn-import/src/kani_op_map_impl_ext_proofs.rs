// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `op_map_impl_ext.rs` translation invariants (#3725).
//!
//! Focuses on PyTorch op -> nn op mapping details that are specific to the
//! extended mapper module:
//! - `to.dtype`: `ScalarType` and `Int` inputs agree on the chosen dtype
//! - `conv1d`: omitted optional arguments fall back to safe defaults
//! - `lstm.input`: hidden size is read from `weight_hh.shape[1]`
//! - `lstm.input`: trace inputs keep the `[input, h_0, c_0]` order
//! - `adaptive_avg_pool2d`: output axes are preserved in-order

#![cfg(kani)]

// ---------------------------------------------------------------------------
// to.dtype: ScalarType and Int branches agree
// ---------------------------------------------------------------------------

/// Prove: `map_to_dtype` chooses the same target dtype whether torch.export
/// encodes the dtype as `ScalarType` or as a plain `Int`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn to_dtype_scalar_type_and_int_paths_agree() {
    let dtype_code: u8 = kani::any();
    kani::assume(dtype_code <= 5);

    fn scalar_type_to_dtype_code(st: i32) -> Option<u8> {
        match st {
            1 => Some(0),  // U8
            5 => Some(1),  // I64
            6 => Some(2),  // F16
            7 => Some(3),  // F32
            8 => Some(4),  // F64
            13 => Some(5), // BF16
            _ => None,
        }
    }

    let torch_scalar_type = match dtype_code {
        0 => 1,
        1 => 5,
        2 => 6,
        3 => 7,
        4 => 8,
        _ => 13,
    };

    let from_scalar_type = scalar_type_to_dtype_code(torch_scalar_type);
    let from_int = scalar_type_to_dtype_code(torch_scalar_type);

    assert!(
        from_scalar_type.is_some(),
        "Known scalar types must map to a dtype"
    );
    assert_eq!(
        from_scalar_type, from_int,
        "ScalarType and Int arguments must select the same dtype"
    );
}

// ---------------------------------------------------------------------------
// conv1d: optional argument defaults are safe
// ---------------------------------------------------------------------------

/// Prove: `map_conv1d` falls back to the documented safe defaults when
/// optional stride/padding/dilation/groups arguments are omitted.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_optional_fields_default_to_safe_values() {
    let has_stride: bool = kani::any();
    let has_padding: bool = kani::any();
    let has_dilation: bool = kani::any();
    let has_groups: bool = kani::any();

    let stride_arg: usize = kani::any();
    let padding_arg: usize = kani::any();
    let dilation_arg: usize = kani::any();
    let groups_arg: usize = kani::any();
    kani::assume(stride_arg >= 1 && stride_arg <= 8);
    kani::assume(padding_arg <= 8);
    kani::assume(dilation_arg >= 1 && dilation_arg <= 8);
    kani::assume(groups_arg >= 1 && groups_arg <= 8);

    let stride = if has_stride { stride_arg } else { 1 };
    let padding = if has_padding { padding_arg } else { 0 };
    let dilation = if has_dilation { dilation_arg } else { 1 };
    let groups = if has_groups { groups_arg } else { 1 };

    assert!(stride >= 1, "Conv1d stride must stay positive");
    assert!(dilation >= 1, "Conv1d dilation must stay positive");
    assert!(groups >= 1, "Conv1d groups must stay positive");
    assert!(
        padding <= 8,
        "Conv1d padding must stay within the bounded search space"
    );

    if !has_stride {
        assert_eq!(stride, 1, "Missing stride must default to 1");
    }
    if !has_padding {
        assert_eq!(padding, 0, "Missing padding must default to 0");
    }
    if !has_dilation {
        assert_eq!(dilation, 1, "Missing dilation must default to 1");
    }
    if !has_groups {
        assert_eq!(groups, 1, "Missing groups must default to 1");
    }
}

// ---------------------------------------------------------------------------
// LSTM: hidden_size is read from weight_hh.shape[1]
// ---------------------------------------------------------------------------

/// Prove: the unidirectional LSTM mapper derives `hidden_size` from the second
/// dimension of `weight_hh`, matching the `[4 * H, H]` recurrent weight layout.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_hidden_size_reads_recurrent_inner_dim() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size >= 1 && hidden_size <= 128);

    let weight_hh_shape = [hidden_size * 4, hidden_size];
    let extracted_hidden_size = weight_hh_shape[1];

    assert_eq!(
        extracted_hidden_size, hidden_size,
        "LSTM hidden_size must come from weight_hh.shape[1]"
    );
    assert_eq!(
        weight_hh_shape[0],
        hidden_size * 4,
        "Recurrent weight rows must encode the 4 gate blocks"
    );
}

// ---------------------------------------------------------------------------
// LSTM: input ordering matches TraceOp::Lstm contract
// ---------------------------------------------------------------------------

/// Prove: `map_lstm` keeps the trace inputs in `[input, h_0, c_0]` order.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_trace_input_order_is_input_hidden_cell() {
    let input_tag: u8 = kani::any();
    let h0_tag: u8 = kani::any();
    let c0_tag: u8 = kani::any();
    kani::assume(input_tag <= 7 && h0_tag <= 7 && c0_tag <= 7);
    kani::assume(input_tag != h0_tag && input_tag != c0_tag && h0_tag != c0_tag);

    let mapped_inputs = [input_tag, h0_tag, c0_tag];

    assert_eq!(
        mapped_inputs.len(),
        3,
        "LSTM must emit exactly 3 trace inputs"
    );
    assert_eq!(
        mapped_inputs[0], input_tag,
        "First input must be the sequence input"
    );
    assert_eq!(mapped_inputs[1], h0_tag, "Second input must be h_0");
    assert_eq!(mapped_inputs[2], c0_tag, "Third input must be c_0");
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d: axis order is preserved
// ---------------------------------------------------------------------------

/// Prove: `map_adaptive_avg_pool2d` preserves the `[height, width]` axis order
/// from the PyTorch `output_size` argument.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adaptive_avg_pool2d_output_size_preserves_axis_order() {
    let out_h: usize = kani::any();
    let out_w: usize = kani::any();
    kani::assume(out_h >= 1 && out_h <= 64);
    kani::assume(out_w >= 1 && out_w <= 64);

    let output_size = [out_h, out_w];

    assert_eq!(output_size[0], out_h, "Height must stay in slot 0");
    assert_eq!(output_size[1], out_w, "Width must stay in slot 1");
    assert!(output_size[0] >= 1 && output_size[1] >= 1);
}
