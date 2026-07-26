// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for weight initialization and parameter distribution bounds.
//!
//! Verifies IBP and CROWN bound propagation through linear and embedding layers
//! with different weight initialization strategies. Weight magnitude directly
//! controls output bound width — smaller weights produce tighter bounds.
//!
//! 1.  **Xavier uniform weight range -> output bound width** (IBP)
//! 2.  **Kaiming normal weight range -> output bound width** (IBP)
//! 3.  **Small weight initialization -> tighter output bounds** (IBP)
//! 4.  **Large weight initialization -> wider output bounds** (IBP)
//! 5.  **Bias initialization effect on output shift** (IBP)
//! 6.  **Zero-initialized bias preserves symmetry** (IBP)
//! 7.  **Weight scale factor effect on bound width** (IBP)
//! 8.  **Embedding weight range -> lookup bound width** (IBP)
//! 9.  **Normalization weight (gamma near 1) -> output bound width** (IBP)
//! 10. **Weight magnitude vs output bound width correlation** (IBP)
//! 11. **CROWN tightness with different weight ranges** (CROWN)
//! 12. **Weight sparsity effect on bounds** (IBP)
//! 13. **Tied weights vs independent weights bound comparison** (IBP)
//! 14. **Weight clipping effect on output bounds** (IBP)
//! 15. **Full model: initialized weights -> forward -> output bounds** (IBP + CROWN)
//!
//! Weight initialization references:
//! - Xavier/Glorot (Glorot & Bengio, 2010): U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
//! - Kaiming/He (He et al., 2015): N(0, sqrt(2/fan_in))
//! - LeCun (LeCun et al., 1998): N(0, sqrt(1/fan_in))
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128
//!
//! Part of #4048: Compose tests for weight initialization and parameter distribution bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 64;
const FFN_DIM: usize = 128;
const VOCAB_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a single linear layer kernel: y = x @ W^T (+ optional bias).
fn build_linear_kernel(
    name: &str,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    with_bias: bool,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[seq_len, in_dim]);
    let w = b.add_input("w", &[out_dim, in_dim]);
    let bias = if with_bias {
        Some(b.add_input("bias", &[out_dim]))
    } else {
        None
    };
    let out = b.add_linear(input, w, bias, &[seq_len, out_dim]);
    b.build(out).expect("valid linear kernel")
}

/// Build bindings for a linear layer with uniform-magnitude weights.
fn linear_bindings(
    in_dim: usize,
    out_dim: usize,
    weight_mag: f32,
    bias_val: Option<f32>,
) -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_dim, in_dim]),
            weight_mag,
        )),
    ];
    if let Some(bv) = bias_val {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_dim]),
            bv,
        )));
    }
    bindings
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

// ===========================================================================
// 1. Xavier uniform weight range -> output bound width (IBP)
// ===========================================================================

/// Xavier uniform initialization: U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))].
/// For fan_in=64, fan_out=128: bound = sqrt(6/192) ≈ 0.1768.
#[test]
fn test_xavier_uniform_weight_range_ibp() {
    let fan_in = HIDDEN_DIM;
    let fan_out = FFN_DIM;
    let xavier_bound = (6.0f32 / (fan_in + fan_out) as f32).sqrt();

    let def = build_linear_kernel("dpdf_weight_init_xavier", SEQ_LEN, fan_in, fan_out, false);
    let bindings = linear_bindings(fan_in, fan_out, xavier_bound, None);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, fan_in], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!(
        "Xavier uniform (fan_in={fan_in}, fan_out={fan_out}) IBP: xavier_bound={xavier_bound:.6}, width={width:.6}"
    );
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 2. Kaiming normal weight range -> output bound width (IBP)
// ===========================================================================

/// Kaiming/He initialization: N(0, sqrt(2/fan_in)).
/// For fan_in=64: std = sqrt(2/64) ≈ 0.1768. Use 2*std as magnitude bound.
#[test]
fn test_kaiming_normal_weight_range_ibp() {
    let fan_in = HIDDEN_DIM;
    let kaiming_std = (2.0f32 / fan_in as f32).sqrt();
    let weight_mag = 2.0 * kaiming_std; // 2-sigma bound

    let def = build_linear_kernel("dpdf_weight_init_kaiming", SEQ_LEN, fan_in, FFN_DIM, false);
    let bindings = linear_bindings(fan_in, FFN_DIM, weight_mag, None);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, fan_in], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!(
        "Kaiming normal (fan_in={fan_in}) IBP: std={kaiming_std:.6}, mag={weight_mag:.6}, width={width:.6}"
    );
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 3. Small weight initialization -> tighter output bounds (IBP)
// ===========================================================================

#[test]
fn test_small_weight_init_tighter_bounds_ibp() {
    let small_mag = 0.001f32;
    let normal_mag = 0.02f32;

    let def = build_linear_kernel(
        "dpdf_weight_init_small",
        SEQ_LEN,
        HIDDEN_DIM,
        FFN_DIM,
        false,
    );

    // Small weights
    let small_bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, small_mag, None);
    let graph_small = tensor_kernel_to_graph(&def, &small_bindings).expect("small graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let small_output = graph_small.propagate_ibp(&input).expect("small IBP");
    assert_bounds_valid(&small_output);
    let small_width = bound_width(&small_output);

    // Normal weights
    let normal_bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, normal_mag, None);
    let graph_normal = tensor_kernel_to_graph(&def, &normal_bindings).expect("normal graph");
    let normal_output = graph_normal.propagate_ibp(&input).expect("normal IBP");
    assert_bounds_valid(&normal_output);
    let normal_width = bound_width(&normal_output);

    eprintln!(
        "Small vs normal init IBP: small_width={small_width:.6}, normal_width={normal_width:.6}"
    );
    assert!(
        small_width <= normal_width + 1e-6,
        "smaller weights must produce tighter bounds: small={small_width}, normal={normal_width}"
    );
}

// ===========================================================================
// 4. Large weight initialization -> wider output bounds (IBP)
// ===========================================================================

#[test]
fn test_large_weight_init_wider_bounds_ibp() {
    let normal_mag = 0.02f32;
    let large_mag = 0.5f32;

    let def = build_linear_kernel(
        "dpdf_weight_init_large",
        SEQ_LEN,
        HIDDEN_DIM,
        FFN_DIM,
        false,
    );

    // Normal weights
    let normal_bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, normal_mag, None);
    let graph_normal = tensor_kernel_to_graph(&def, &normal_bindings).expect("normal graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let normal_output = graph_normal.propagate_ibp(&input).expect("normal IBP");
    assert_bounds_valid(&normal_output);
    let normal_width = bound_width(&normal_output);

    // Large weights
    let large_bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, large_mag, None);
    let graph_large = tensor_kernel_to_graph(&def, &large_bindings).expect("large graph");
    let large_output = graph_large.propagate_ibp(&input).expect("large IBP");
    assert_bounds_valid(&large_output);
    let large_width = bound_width(&large_output);

    eprintln!(
        "Normal vs large init IBP: normal_width={normal_width:.6}, large_width={large_width:.6}"
    );
    assert!(
        large_width >= normal_width - 1e-6,
        "larger weights must produce wider bounds: large={large_width}, normal={normal_width}"
    );
}

// ===========================================================================
// 5. Bias initialization effect on output shift (IBP)
// ===========================================================================

#[test]
fn test_bias_init_shifts_output_bounds_ibp() {
    let weight_mag = 0.02f32;
    let bias_val = 1.0f32;

    // Without bias
    let def_no_bias = build_linear_kernel(
        "dpdf_weight_init_no_bias",
        SEQ_LEN,
        HIDDEN_DIM,
        FFN_DIM,
        false,
    );
    let bindings_no_bias = linear_bindings(HIDDEN_DIM, FFN_DIM, weight_mag, None);
    let graph_no_bias =
        tensor_kernel_to_graph(&def_no_bias, &bindings_no_bias).expect("no-bias graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let no_bias_output = graph_no_bias.propagate_ibp(&input).expect("no-bias IBP");
    assert_bounds_valid(&no_bias_output);
    let (no_bias_lo, no_bias_hi) = bounds_min_max(&no_bias_output);

    // With positive bias
    let def_bias = build_linear_kernel("dpdf_weight_init_bias", SEQ_LEN, HIDDEN_DIM, FFN_DIM, true);
    let bindings_bias = linear_bindings(HIDDEN_DIM, FFN_DIM, weight_mag, Some(bias_val));
    let graph_bias = tensor_kernel_to_graph(&def_bias, &bindings_bias).expect("bias graph");
    let bias_output = graph_bias.propagate_ibp(&input).expect("bias IBP");
    assert_bounds_valid(&bias_output);
    let (bias_lo, bias_hi) = bounds_min_max(&bias_output);

    eprintln!(
        "Bias effect IBP: no_bias=[{no_bias_lo:.6}, {no_bias_hi:.6}], bias=[{bias_lo:.6}, {bias_hi:.6}]"
    );
    // Positive bias shifts bounds upward
    let tol = 1e-4;
    assert!(
        bias_lo >= no_bias_lo + bias_val - tol,
        "bias should shift lower bound up: bias_lo={bias_lo}, no_bias_lo={no_bias_lo}, shift={bias_val}"
    );
    assert!(
        bias_hi >= no_bias_hi + bias_val - tol,
        "bias should shift upper bound up: bias_hi={bias_hi}, no_bias_hi={no_bias_hi}, shift={bias_val}"
    );
}

// ===========================================================================
// 6. Zero-initialized bias preserves symmetry (IBP)
// ===========================================================================

#[test]
fn test_zero_bias_preserves_symmetry_ibp() {
    let weight_mag = 0.02f32;

    let def = build_linear_kernel(
        "dpdf_weight_init_zero_bias",
        SEQ_LEN,
        HIDDEN_DIM,
        FFN_DIM,
        true,
    );
    let bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, weight_mag, Some(0.0));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Zero bias IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // With symmetric input and uniform positive weights, zero bias preserves
    // approximate symmetry around zero (IBP with uniform weights is symmetric).
    let tol = 1e-4;
    assert!(
        (lo_min + hi_max).abs() < tol,
        "zero bias should preserve symmetry: lo_min={lo_min}, hi_max={hi_max}, sum={}",
        lo_min + hi_max
    );
}

// ===========================================================================
// 7. Weight scale factor effect on bound width (IBP)
// ===========================================================================

#[test]
fn test_weight_scale_factor_bound_width_ibp() {
    let base_mag = 0.01f32;
    let def = build_linear_kernel(
        "dpdf_weight_init_scale_factor",
        SEQ_LEN,
        HIDDEN_DIM,
        FFN_DIM,
        false,
    );
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let mut prev_width = 0.0f32;
    for scale in [1.0, 2.0, 4.0, 8.0] {
        let mag = base_mag * scale;
        let bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, mag, None);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let width = bound_width(&output);

        eprintln!("Scale factor {scale}: mag={mag:.6}, width={width:.6}");
        assert!(width.is_finite(), "width must be finite at scale {scale}");
        if scale > 1.0 {
            assert!(
                width >= prev_width - 1e-6,
                "bound width should increase with scale: scale={scale}, width={width}, prev={prev_width}"
            );
        }
        prev_width = width;
    }
}

// ===========================================================================
// 8. Embedding weight range -> lookup bound width (IBP)
// ===========================================================================

#[test]
fn test_embedding_weight_range_lookup_width_ibp() {
    let emb_mag = 0.02f32;

    let mut b = TensorBlockBuilder::new("dpdf_weight_init_embedding");
    let indices = b.add_input("indices", &[SEQ_LEN]);
    let emb_w = b.add_input("emb_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let out = b.add_embedding(indices, emb_w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid embedding kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            emb_mag,
        )),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Embedding input: token indices bounded in [0, VOCAB_SIZE-1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid index bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Embedding weight range IBP: emb_mag={emb_mag:.6}, width={width:.6}");
    assert!(width.is_finite(), "embedding output width must be finite");
}

// ===========================================================================
// 9. Normalization weight (gamma near 1) -> output bound width (IBP)
// ===========================================================================

#[test]
fn test_norm_gamma_near_one_bound_width_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_weight_init_norm_gamma");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let normed = b.add_rms_norm(input, eps, 1, gamma, &shape);
    let def = b.build(normed).expect("valid RMSNorm kernel");

    // gamma near 1.0 (standard initialization)
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("RMSNorm gamma=1.0 IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 10. Weight magnitude vs output bound width correlation (IBP)
// ===========================================================================

#[test]
fn test_weight_magnitude_bound_width_correlation_ibp() {
    let magnitudes = [0.001f32, 0.01, 0.05, 0.1, 0.5];
    let def = build_linear_kernel(
        "dpdf_weight_init_correlation",
        SEQ_LEN,
        HIDDEN_DIM,
        FFN_DIM,
        false,
    );
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let mut widths = Vec::new();
    for &mag in &magnitudes {
        let bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, mag, None);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let width = bound_width(&output);
        widths.push(width);
        eprintln!("Magnitude {mag:.4}: width={width:.6}");
    }

    // Bound width should be monotonically non-decreasing with weight magnitude
    for i in 1..widths.len() {
        assert!(
            widths[i] >= widths[i - 1] - 1e-6,
            "width must increase with magnitude: mag[{}]={}, width[{}]={}, mag[{}]={}, width[{}]={}",
            i,
            magnitudes[i],
            i,
            widths[i],
            i - 1,
            magnitudes[i - 1],
            i - 1,
            widths[i - 1]
        );
    }
}

// ===========================================================================
// 11. CROWN tightness with different weight ranges (CROWN)
// ===========================================================================

#[test]
fn test_crown_tightness_different_weight_ranges() {
    for &weight_mag in &[0.01f32, 0.05, 0.1] {
        let mut b = TensorBlockBuilder::new(&format!("dpdf_weight_init_crown_{weight_mag:.0e}"));
        let input_node = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let w = b.add_input("w", &[FFN_DIM, HIDDEN_DIM]);
        let h = b.add_linear(input_node, w, None, &[SEQ_LEN, FFN_DIM]);
        // Add nonlinearity so CROWN linearization has something to do
        let h = add_silu(&mut b, h, &[SEQ_LEN, FFN_DIM]);
        let w2 = b.add_input("w2", &[HIDDEN_DIM, FFN_DIM]);
        let out = b.add_linear(h, w2, None, &[SEQ_LEN, HIDDEN_DIM]);
        let def = b.build(out).expect("valid CROWN kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[FFN_DIM, HIDDEN_DIM]),
                weight_mag,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[HIDDEN_DIM, FFN_DIM]),
                weight_mag,
            )),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

        let (method, output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);

        assert_bounds_valid(&output);
        let width = bound_width(&output);
        eprintln!("CROWN weight_mag={weight_mag:.3}: method={method:?}, width={width:.6}");
        if let Some(reason) = &fallback_reason {
            eprintln!("Fallback reason: {reason}");
        }
    }
}

// ===========================================================================
// 12. Weight sparsity effect on bounds (IBP)
// ===========================================================================

/// Sparse weights (many zeros) should produce tighter bounds than dense
/// weights of the same magnitude, because zero entries contribute nothing
/// to the interval sum.
#[test]
fn test_weight_sparsity_effect_on_bounds_ibp() {
    let weight_mag = 0.1f32;

    // Dense weights: all elements = weight_mag
    let def = build_linear_kernel(
        "dpdf_weight_init_sparsity",
        SEQ_LEN,
        HIDDEN_DIM,
        FFN_DIM,
        false,
    );
    let dense_bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, weight_mag, None);
    let graph_dense = tensor_kernel_to_graph(&def, &dense_bindings).expect("dense graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let dense_output = graph_dense.propagate_ibp(&input).expect("dense IBP");
    assert_bounds_valid(&dense_output);
    let dense_width = bound_width(&dense_output);

    // Sparse weights: 75% zeros, 25% = weight_mag
    let mut sparse_data = vec![0.0f32; FFN_DIM * HIDDEN_DIM];
    for (i, val) in sparse_data.iter_mut().enumerate() {
        if i % 4 == 0 {
            *val = weight_mag;
        }
    }
    let sparse_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[FFN_DIM, HIDDEN_DIM]), sparse_data)
                .expect("valid sparse weights"),
        ),
    ];
    let graph_sparse = tensor_kernel_to_graph(&def, &sparse_bindings).expect("sparse graph");
    let sparse_output = graph_sparse.propagate_ibp(&input).expect("sparse IBP");
    assert_bounds_valid(&sparse_output);
    let sparse_width = bound_width(&sparse_output);

    eprintln!("Sparsity effect IBP: dense_width={dense_width:.6}, sparse_width={sparse_width:.6}");
    assert!(
        sparse_width <= dense_width + 1e-4,
        "sparse weights should produce tighter bounds: sparse={sparse_width}, dense={dense_width}"
    );
}

// ===========================================================================
// 13. Tied weights vs independent weights bound comparison (IBP)
// ===========================================================================

/// Tied weights (W_2 = W_1^T) should produce the same bounds as a single
/// projection round-trip. Independent W_1, W_2 may differ.
#[test]
fn test_tied_vs_independent_weights_ibp() {
    let weight_mag = 0.02f32;

    // Tied: project hidden -> ffn -> hidden using the same weight matrix (transposed)
    let mut b_tied = TensorBlockBuilder::new("dpdf_weight_init_tied");
    let input = b_tied.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b_tied.add_input("w", &[FFN_DIM, HIDDEN_DIM]);
    let w_t = b_tied.add_input("w_t", &[HIDDEN_DIM, FFN_DIM]);
    let h = b_tied.add_linear(input, w, None, &[SEQ_LEN, FFN_DIM]);
    let out = b_tied.add_linear(h, w_t, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def_tied = b_tied.build(out).expect("valid tied kernel");

    let tied_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            weight_mag,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            weight_mag,
        )),
    ];

    let graph_tied = tensor_kernel_to_graph(&def_tied, &tied_bindings).expect("tied graph");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let tied_output = graph_tied.propagate_ibp(&input_bounds).expect("tied IBP");
    assert_bounds_valid(&tied_output);
    let tied_width = bound_width(&tied_output);

    // Independent: different weight magnitudes for each projection
    let indep_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            weight_mag,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            weight_mag * 2.0,
        )),
    ];
    let graph_indep = tensor_kernel_to_graph(&def_tied, &indep_bindings).expect("indep graph");
    let indep_output = graph_indep.propagate_ibp(&input_bounds).expect("indep IBP");
    assert_bounds_valid(&indep_output);
    let indep_width = bound_width(&indep_output);

    eprintln!("Tied vs independent IBP: tied_width={tied_width:.6}, indep_width={indep_width:.6}");
    // Independent with 2x weight in second layer should be wider
    assert!(
        indep_width >= tied_width - 1e-4,
        "independent 2x weight should produce wider bounds: indep={indep_width}, tied={tied_width}"
    );
}

// ===========================================================================
// 14. Weight clipping effect on output bounds (IBP)
// ===========================================================================

/// Clipping weights to a smaller range should produce tighter output bounds.
#[test]
fn test_weight_clipping_effect_on_bounds_ibp() {
    let original_mag = 0.1f32;
    let clipped_mag = 0.05f32;

    let def = build_linear_kernel(
        "dpdf_weight_init_clipping",
        SEQ_LEN,
        HIDDEN_DIM,
        FFN_DIM,
        false,
    );
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Original (unclipped) weights
    let orig_bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, original_mag, None);
    let graph_orig = tensor_kernel_to_graph(&def, &orig_bindings).expect("original graph");
    let orig_output = graph_orig.propagate_ibp(&input).expect("original IBP");
    assert_bounds_valid(&orig_output);
    let orig_width = bound_width(&orig_output);

    // Clipped weights
    let clip_bindings = linear_bindings(HIDDEN_DIM, FFN_DIM, clipped_mag, None);
    let graph_clip = tensor_kernel_to_graph(&def, &clip_bindings).expect("clipped graph");
    let clip_output = graph_clip.propagate_ibp(&input).expect("clipped IBP");
    assert_bounds_valid(&clip_output);
    let clip_width = bound_width(&clip_output);

    eprintln!("Weight clipping IBP: original_width={orig_width:.6}, clipped_width={clip_width:.6}");
    assert!(
        clip_width <= orig_width + 1e-4,
        "clipped weights must produce tighter bounds: clip={clip_width}, orig={orig_width}"
    );
}

// ===========================================================================
// 15. Full model: initialized weights -> forward -> output bounds (IBP + CROWN)
// ===========================================================================

/// Full model: Linear -> SiLU -> Linear -> RMSNorm -> Linear (output projection).
/// Tests that standard initialization produces finite, reasonable bounds
/// through a multi-layer pipeline.
fn build_full_model_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_weight_init_full_model");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape_h = [SEQ_LEN, FFN_DIM];
    let shape_out = [SEQ_LEN, HIDDEN_DIM];

    // Layer 1: Linear -> SiLU
    let w1 = b.add_input("w1", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(input, w1, None, &shape_h);
    let h = add_silu(&mut b, h, &shape_h);

    // Layer 2: Linear (down projection)
    let w2 = b.add_input("w2", &[HIDDEN_DIM, FFN_DIM]);
    let h = b.add_linear(h, w2, None, &shape_out);

    // RMSNorm
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[HIDDEN_DIM]);
    let h = b.add_rms_norm(h, eps, 1, gamma, &shape_out);

    // Output projection
    let w3 = b.add_input("w3", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(h, w3, None, &shape_out);

    b.build(out).expect("valid full model kernel")
}

fn full_model_bindings(weight_mag: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // w1: [FFN_DIM, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            weight_mag,
        )),
        // w2: [HIDDEN_DIM, FFN_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            weight_mag,
        )),
        // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        // gamma (RMSNorm weight, initialized to 1.0)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        // w3: [HIDDEN_DIM, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            weight_mag,
        )),
    ]
}

#[test]
fn test_full_model_initialized_weights_ibp() {
    let weight_mag = 0.02f32;
    let def = build_full_model_kernel();
    let bindings = full_model_bindings(weight_mag);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("Full model IBP: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_full_model_initialized_weights_crown() {
    let weight_mag = 0.02f32;
    let def = build_full_model_kernel();
    let bindings = full_model_bindings(weight_mag);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full model CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
