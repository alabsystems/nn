// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for RoPE (K6) kernel: builders, reference, and tensor IR.

use super::*;

// --- Scalar kernel builders ---

#[test]
fn test_rope_cos_kernel_builds() {
    let k = build_rope_cos_kernel().expect("build must succeed");
    assert_eq!(k.name, "rope_cos");
    assert_eq!(k.params.len(), 3, "rope_cos takes (x0, x1, freq)");
    k.validate().expect("IR must validate");
}

#[test]
fn test_rope_sin_kernel_builds() {
    let k = build_rope_sin_kernel().expect("build must succeed");
    assert_eq!(k.name, "rope_sin");
    assert_eq!(k.params.len(), 3, "rope_sin takes (x0, x1, freq)");
    k.validate().expect("IR must validate");
}

// --- Scalar reference: known values ---

#[test]
fn test_rope_cos_scalar_zero_freq() {
    let y = rope_cos_scalar(3.0, 7.0, 0.0).expect("must succeed");
    assert!(
        (y - 3.0).abs() < 1e-6,
        "zero freq: y0 should equal x0, got {y}"
    );
}

#[test]
fn test_rope_sin_scalar_zero_freq() {
    let y = rope_sin_scalar(3.0, 7.0, 0.0).expect("must succeed");
    assert!(
        (y - 7.0).abs() < 1e-6,
        "zero freq: y1 should equal x1, got {y}"
    );
}

#[test]
fn test_rope_cos_scalar_pi_half() {
    let freq = std::f32::consts::FRAC_PI_2;
    let y = rope_cos_scalar(3.0, 7.0, freq).expect("must succeed");
    assert!((y - (-7.0)).abs() < 1e-5, "π/2 freq: y0 ≈ -x1, got {y}");
}

#[test]
fn test_rope_sin_scalar_pi_half() {
    let freq = std::f32::consts::FRAC_PI_2;
    let y = rope_sin_scalar(3.0, 7.0, freq).expect("must succeed");
    assert!((y - 3.0).abs() < 1e-5, "π/2 freq: y1 ≈ x0, got {y}");
}

#[test]
fn test_rope_rotation_preserves_norm_known() {
    let x0 = 3.0_f32;
    let x1 = 4.0_f32;
    let freq = 1.5_f32;

    let y0 = rope_cos_scalar(x0, x1, freq).expect("must succeed");
    let y1 = rope_sin_scalar(x0, x1, freq).expect("must succeed");

    let input_norm = x0 * x0 + x1 * x1;
    let output_norm = y0 * y0 + y1 * y1;

    assert!(
        (output_norm - input_norm).abs() < 1e-3,
        "norm should be preserved: input={input_norm}, output={output_norm}"
    );
}

// --- Tensor kernel builder ---

#[test]
fn test_rope_rotate_validates() {
    let k = build_rope_rotate_kernel(4, 8, 6).expect("build must succeed");
    k.validate().expect("K6 RoPE IR must validate");
}

#[test]
fn test_rope_rotate_zero_dim_returns_err() {
    assert!(build_rope_rotate_kernel(0, 8, 6).is_err(), "zero bh");
    assert!(build_rope_rotate_kernel(4, 0, 6).is_err(), "zero seq_len");
    assert!(build_rope_rotate_kernel(4, 8, 0).is_err(), "zero head_dim");
}

#[test]
fn test_rope_rotate_odd_head_dim_returns_err() {
    assert!(
        build_rope_rotate_kernel(4, 8, 7).is_err(),
        "odd head_dim must return Err"
    );
}

#[test]
fn test_rope_rotate_node_count() {
    let k = build_rope_rotate_kernel(4, 8, 6).expect("build must succeed");
    assert_eq!(
        k.nodes.len(),
        10,
        "2 inputs + 1 reshape + 2 axis_select + 1 broadcast + 2 elementwise + 1 stack + 1 reshape = 10"
    );
}

#[test]
fn test_rope_rotate_output_shape() {
    let k = build_rope_rotate_kernel(4, 8, 6).expect("build must succeed");
    let output_shape = &k.nodes[k.output.index()].shape;
    assert_eq!(
        output_shape,
        &[4, 8, 6],
        "output shape must match input [BH, S, D]"
    );
}

#[test]
fn test_rope_rotate_pretty_print() {
    let k = build_rope_rotate_kernel(2, 4, 6).expect("build must succeed");
    let ir = crate::tensor_ir::tensor_ir_pretty_print(&k);
    assert!(
        ir.contains("tensor_kernel rope_rotate"),
        "should contain kernel name"
    );
    assert!(ir.contains("reshape"), "should contain reshape ops");
    assert!(ir.contains("axis_select"), "should contain axis_select ops");
    assert!(ir.contains("stack"), "should contain stack op");
    assert!(
        ir.contains("elementwise(rope_cos"),
        "should contain rope_cos"
    );
    assert!(
        ir.contains("elementwise(rope_sin"),
        "should contain rope_sin"
    );
    assert!(ir.contains("broadcast"), "should contain broadcast");
}

// --- Reference implementation ---

#[test]
fn test_rope_rotate_ref_zero_freq() {
    let bh = 1;
    let seq_len = 2;
    let head_dim = 4;
    let half_dim = head_dim / 2;

    let x: Vec<f32> = (0..bh * seq_len * head_dim)
        .map(|i| i as f32 * 0.1)
        .collect();
    let freqs = vec![0.0f32; seq_len * half_dim];

    let out = rope_rotate_ref(&x, &freqs, bh, seq_len, head_dim).expect("ref must succeed");

    for (i, (&got, &exp)) in out.iter().zip(x.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "zero freq should be identity at index {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_rope_rotate_ref_preserves_norm() {
    let bh = 2;
    let seq_len = 3;
    let head_dim = 6;
    let half_dim = head_dim / 2;

    let x: Vec<f32> = (0..bh * seq_len * head_dim)
        .map(|i| i as f32 * 0.01 - 0.2)
        .collect();
    let freqs: Vec<f32> = (0..seq_len * half_dim)
        .map(|i| i as f32 * 0.3 - 0.5)
        .collect();

    let out = rope_rotate_ref(&x, &freqs, bh, seq_len, head_dim).expect("ref must succeed");

    for b in 0..bh {
        for s in 0..seq_len {
            for p in 0..half_dim {
                let base = b * seq_len * head_dim + s * head_dim + p * 2;
                let in_norm = x[base] * x[base] + x[base + 1] * x[base + 1];
                let out_norm = out[base] * out[base] + out[base + 1] * out[base + 1];
                assert!(
                    (out_norm - in_norm).abs() < 1e-4,
                    "norm mismatch at b={b},s={s},p={p}: in={in_norm}, out={out_norm}"
                );
            }
        }
    }
}

#[test]
fn test_rope_rotate_ref_matches_manual_computation() {
    let bh = 1;
    let seq_len = 1;
    let head_dim = 4;

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let freqs = vec![0.5, 1.0];

    let out = rope_rotate_ref(&x, &freqs, bh, seq_len, head_dim).expect("ref must succeed");

    let c0 = 0.5_f32.cos();
    let s0 = 0.5_f32.sin();
    let exp_0 = 1.0 * c0 - 2.0 * s0;
    let exp_1 = 1.0 * s0 + 2.0 * c0;

    let c1 = 1.0_f32.cos();
    let s1 = 1.0_f32.sin();
    let exp_2 = 3.0 * c1 - 4.0 * s1;
    let exp_3 = 3.0 * s1 + 4.0 * c1;

    assert!(
        (out[0] - exp_0).abs() < 1e-6,
        "pair 0 even: got {}, exp {exp_0}",
        out[0]
    );
    assert!(
        (out[1] - exp_1).abs() < 1e-6,
        "pair 0 odd: got {}, exp {exp_1}",
        out[1]
    );
    assert!(
        (out[2] - exp_2).abs() < 1e-6,
        "pair 1 even: got {}, exp {exp_2}",
        out[2]
    );
    assert!(
        (out[3] - exp_3).abs() < 1e-6,
        "pair 1 odd: got {}, exp {exp_3}",
        out[3]
    );
}

// --- Error cases ---

#[test]
fn test_rope_rotate_ref_wrong_x_length() {
    assert!(
        rope_rotate_ref(&[1.0; 5], &[0.0; 2], 1, 2, 4).is_err(),
        "wrong x length must return Err"
    );
}

#[test]
fn test_rope_rotate_ref_wrong_freqs_length() {
    assert!(
        rope_rotate_ref(&[1.0; 8], &[0.0; 3], 1, 2, 4).is_err(),
        "wrong freqs length must return Err"
    );
}

#[test]
fn test_rope_rotate_ref_odd_head_dim() {
    assert!(
        rope_rotate_ref(&[1.0; 7], &[0.0; 3], 1, 1, 7).is_err(),
        "odd head_dim must return Err"
    );
}

// --- Differential: builder matches reference ---

#[test]
fn test_rope_rotate_head_dim_2_minimum_boundary() {
    let k = build_rope_rotate_kernel(1, 1, 2).expect("head_dim=2 must succeed");
    k.validate().expect("must validate");
    assert_eq!(k.nodes[k.output.index()].shape, vec![1, 1, 2]);

    let x = vec![1.0_f32, 0.0];
    let freqs = vec![0.0_f32];
    let result = rope_rotate_ref(&x, &freqs, 1, 1, 2);
    assert!(result.is_ok());
    let out = result.unwrap();
    assert!((out[0] - 1.0).abs() < 1e-6, "even element should be x0");
    assert!((out[1] - 0.0).abs() < 1e-6, "odd element should be x1");
}

#[test]
fn test_rope_rotate_builder_matches_ref_semantics() {
    let bh = 2;
    let seq_len = 3;
    let head_dim = 6;

    let k = build_rope_rotate_kernel(bh, seq_len, head_dim).expect("build must succeed");
    k.validate().expect("must validate");

    let input_x = &k.nodes[0];
    assert_eq!(input_x.shape, vec![bh, seq_len, head_dim]);

    let input_freqs = &k.nodes[1];
    assert_eq!(input_freqs.shape, vec![seq_len, head_dim / 2]);

    let output = &k.nodes[k.output.index()];
    assert_eq!(output.shape, vec![bh, seq_len, head_dim]);
}

#[test]
fn test_rope_rotate_ref_nan_x_rejected() {
    // Shape: bh=1, seq_len=1, head_dim=2 → x len = 2, freqs len = 1
    let x = &[f32::NAN, 1.0];
    let freqs = &[0.5];
    let err = rope_rotate_ref(x, freqs, 1, 1, 2).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteSliceElement {
                name: "x",
                index: 0,
                ..
            }
        ),
        "NaN at x[0] should be caught, got: {err}"
    );
}

#[test]
fn test_rope_rotate_ref_inf_freqs_rejected() {
    let x = &[1.0, 2.0];
    let freqs = &[f32::INFINITY];
    let err = rope_rotate_ref(x, freqs, 1, 1, 2).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteSliceElement {
                name: "freqs",
                index: 0,
                ..
            }
        ),
        "Inf at freqs[0] should be caught, got: {err}"
    );
}

// --- MSL codegen tests ---

#[test]
fn test_rope_cos_kernel_msl_codegen() {
    let kernel = build_rope_cos_kernel().expect("build must succeed");
    let msl = crate::emit_msl(&kernel).expect("MSL codegen");
    assert!(msl.contains("rope_cos"), "MSL should contain kernel name");
    assert!(msl.contains("cos("), "MSL should use cos for RoPE rotation");
    assert!(
        msl.contains("[[kernel]]"),
        "MSL should have kernel attribute",
    );
}

#[test]
fn test_rope_sin_kernel_msl_codegen() {
    let kernel = build_rope_sin_kernel().expect("build must succeed");
    let msl = crate::emit_msl(&kernel).expect("MSL codegen");
    assert!(msl.contains("rope_sin"), "MSL should contain kernel name");
    assert!(msl.contains("sin("), "MSL should use sin for RoPE rotation");
    assert!(
        msl.contains("[[kernel]]"),
        "MSL should have kernel attribute",
    );
}
