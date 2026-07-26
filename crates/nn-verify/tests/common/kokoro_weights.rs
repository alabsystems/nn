// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared Kokoro weight construction helpers for compose tests.
//!
//! Extracted from `compose_kokoro_trace_full.rs` and `compose_kokoro_traced.rs`
//! to eliminate 6 duplicated helper functions (#2404).
//!
//! Part of #2404.
//! Part of #2218.

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_models::kokoro_decoder::Generator;
use nn_models::KokoroConfig;
use nn_verify::{BoundedTensor, GraphNetwork, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// -- Weight insertion helpers -------------------------------------------------

/// Insert a weight tensor with the given fill value.
///
/// `fill = 0.0` for structure-only tracing, `fill = 0.01` for meaningful IBP bounds.
pub(crate) fn z_fill(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize], fill: f64) {
    let t = if fill == 0.0 {
        DynTensor::zeros(shape, DType::F32, &cpu()).unwrap()
    } else {
        DynTensor::full(shape, fill, DType::F32, &cpu()).unwrap()
    };
    m.insert(name.to_string(), t);
}

/// Insert Conv1d weight `[out_ch, in_ch, k]` and bias `[out_ch]` at `prefix`.
pub(crate) fn conv1d_w(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    fill: f64,
) {
    z_fill(m, &format!("{prefix}.weight"), &[out_ch, in_ch, k], fill);
    z_fill(m, &format!("{prefix}.bias"), &[out_ch], fill);
}

/// Insert ConvTranspose1d weight `[in_ch, out_ch, k]` and bias `[out_ch]` at `prefix`.
pub(crate) fn conv_tr_w(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    fill: f64,
) {
    z_fill(m, &format!("{prefix}.weight"), &[in_ch, out_ch, k], fill);
    z_fill(m, &format!("{prefix}.bias"), &[out_ch], fill);
}

/// Insert ResBlock weights for one block with given dilations.
///
/// Unified from `compose_kokoro_trace_full.rs` (parameterized) and
/// `compose_kokoro_traced.rs` (hardcoded single dilation). Use `dilations = &[1]`
/// for the simple single-dilation case.
pub(crate) fn resblock_w(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    ch: usize,
    k: usize,
    style_dim: usize,
    dilations: &[usize],
    fill: f64,
) {
    for i in 0..dilations.len() {
        conv1d_w(m, &format!("{prefix}.convs1.{i}"), ch, ch, k, fill);
        conv1d_w(m, &format!("{prefix}.convs2.{i}"), ch, ch, k, fill);
        z_fill(
            m,
            &format!("{prefix}.adain1.{i}.fc.weight"),
            &[2 * ch, style_dim],
            fill,
        );
        z_fill(m, &format!("{prefix}.adain1.{i}.fc.bias"), &[2 * ch], fill);
        z_fill(
            m,
            &format!("{prefix}.adain2.{i}.fc.weight"),
            &[2 * ch, style_dim],
            fill,
        );
        z_fill(m, &format!("{prefix}.adain2.{i}.fc.bias"), &[2 * ch], fill);
        // Alpha must be nonzero to avoid division by near-zero in snake.
        m.insert(
            format!("{prefix}.alpha1.{i}"),
            DynTensor::full(&[1, ch, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
        m.insert(
            format!("{prefix}.alpha2.{i}"),
            DynTensor::full(&[1, ch, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
    }
}

// -- Model-level weight builders ----------------------------------------------

/// Build minimal weights for bert_encoder (Linear: hidden_dim → d_en).
pub(crate) fn bert_encoder_weights(
    d_en: usize,
    hidden: usize,
    fill: f64,
) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    z_fill(&mut m, "weight", &[d_en, hidden], fill);
    z_fill(&mut m, "bias", &[d_en], fill);
    m
}

/// Build minimal weights for TextEncoder (Embedding + Conv + LayerNorm + BiLSTM + Linear).
pub(crate) fn text_encoder_weights(
    vocab_size: usize,
    d_en: usize,
    fill: f64,
) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let bilstm_hidden = d_en / 2;

    // Embedding(vocab_size, d_en)
    z_fill(&mut m, "embedding.weight", &[vocab_size, d_en], fill);

    // 3x Conv1d(d_en, d_en, k=5) + LayerNorm(d_en)
    for i in 0..3 {
        z_fill(&mut m, &format!("convs.{i}.weight"), &[d_en, d_en, 5], fill);
        z_fill(&mut m, &format!("convs.{i}.bias"), &[d_en], fill);
        // LayerNorm weight=1.0 for stable normalization
        m.insert(
            format!("norms.{i}.weight"),
            DynTensor::full(&[d_en], 1.0, DType::F32, &cpu()).unwrap(),
        );
        z_fill(&mut m, &format!("norms.{i}.bias"), &[d_en], fill);
    }

    // BiLSTM
    let p = "lstm";
    z_fill(
        &mut m,
        &format!("{p}.weight_ih_l0"),
        &[4 * bilstm_hidden, d_en],
        fill,
    );
    z_fill(
        &mut m,
        &format!("{p}.weight_hh_l0"),
        &[4 * bilstm_hidden, bilstm_hidden],
        fill,
    );
    z_fill(
        &mut m,
        &format!("{p}.bias_ih_l0"),
        &[4 * bilstm_hidden],
        fill,
    );
    z_fill(
        &mut m,
        &format!("{p}.bias_hh_l0"),
        &[4 * bilstm_hidden],
        fill,
    );
    z_fill(
        &mut m,
        &format!("{p}.weight_ih_l0_reverse"),
        &[4 * bilstm_hidden, d_en],
        fill,
    );
    z_fill(
        &mut m,
        &format!("{p}.weight_hh_l0_reverse"),
        &[4 * bilstm_hidden, bilstm_hidden],
        fill,
    );
    z_fill(
        &mut m,
        &format!("{p}.bias_ih_l0_reverse"),
        &[4 * bilstm_hidden],
        fill,
    );
    z_fill(
        &mut m,
        &format!("{p}.bias_hh_l0_reverse"),
        &[4 * bilstm_hidden],
        fill,
    );
    z_fill(&mut m, &format!("{p}.linear.weight"), &[d_en, d_en], fill);
    z_fill(&mut m, &format!("{p}.linear.bias"), &[d_en], fill);
    m
}

// -- Bounds helpers -----------------------------------------------------------

/// Create a `BoundedTensor` with uniform asymmetric `[lo, hi]` bounds.
///
/// Unlike `common::uniform_bounds()` (symmetric ±range), this accepts arbitrary
/// lo/hi values. Unlike `common::scalar_bounds()` (1-element only), this accepts
/// any shape.
pub(crate) fn asymmetric_bounds(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(shape), lo);
    let upper = ArrayD::from_elem(IxDyn(shape), hi);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

/// Build stacked multi-input bounds and propagate IBP.
///
/// Each entry in `inputs` is `(shape, (lower_bound, upper_bound))`.
/// Shapes are flattened and concatenated into a single 1D `BoundedTensor`.
pub(crate) fn propagate_multi_input_ibp(
    gn: &GraphNetwork,
    inputs: &[(&[usize], (f32, f32))],
) -> BoundedTensor {
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for &(shape, (lo, hi)) in inputs {
        let flat: usize = shape.iter().product();
        lower.extend(vec![lo; flat]);
        upper.extend(vec![hi; flat]);
    }
    let total = lower.len();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid bounds");
    gn.propagate_ibp(&input_bounds).expect("IBP propagation")
}

// -- Generator constants & weights --------------------------------------------

/// Generator test dimensions — shared between `compose_kokoro_trace_full.rs`
/// and `compose_kokoro_traced.rs`.
pub(crate) const GEN_CH: usize = 8; // initial_channels
pub(crate) const GEN_NEXT_CH: usize = 4; // ch / 2 after one upsample stage
pub(crate) const GEN_N_FFT: usize = 4; // n_bins = n_fft/2 + 1 = 3
pub(crate) const GEN_N_BINS: usize = GEN_N_FFT / 2 + 1; // 3
pub(crate) const GEN_KERNEL: usize = 3; // resblock kernel size

/// Build minimal weights for Generator: 1 upsample stage, 1 resblock, 1 dilation.
///
/// `fill` controls whether weights are zero (structure-only tracing) or non-zero
/// (meaningful IBP bounds). `style_dim` is the style embedding dimension.
pub(crate) fn generator_weights(fill: f64, style_dim: usize) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let resblock_dilations = [1usize];
    let noise_dilations = [1usize, 3, 5];
    conv1d_w(&mut m, "conv_pre", GEN_CH, GEN_CH, 7, fill);
    conv_tr_w(&mut m, "ups.0", GEN_CH, GEN_NEXT_CH, 4, fill);
    conv1d_w(
        &mut m,
        "noise_convs.0",
        GEN_NEXT_CH,
        2 * GEN_N_BINS,
        1,
        fill,
    );
    resblock_w(
        &mut m,
        "noise_res.0",
        GEN_NEXT_CH,
        11,
        style_dim,
        &noise_dilations,
        fill,
    );
    resblock_w(
        &mut m,
        "resblocks.0",
        GEN_NEXT_CH,
        GEN_KERNEL,
        style_dim,
        &resblock_dilations,
        fill,
    );
    conv1d_w(&mut m, "conv_post", 2 * GEN_N_BINS, GEN_NEXT_CH, 7, fill);
    m
}

/// Build minimal weights for Generator at arbitrary channel dimensions.
///
/// Like `generator_weights()` but parameterized for scaled pipeline tests.
/// `gen_ch` is the initial channel count; `gen_next_ch` = `gen_ch / 2`.
/// Uses 1 upsample stage, 1 resblock, 1 dilation — same structure, different scale.
///
/// Part of #2593.
pub(crate) fn generator_weights_scaled(
    gen_ch: usize,
    n_fft: usize,
    resblock_kernel: usize,
    fill: f64,
    style_dim: usize,
) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let gen_next_ch = gen_ch / 2;
    let n_bins = n_fft / 2 + 1;
    let resblock_dilations = [1usize];
    let noise_dilations = [1usize, 3, 5];

    conv1d_w(&mut m, "conv_pre", gen_ch, gen_ch, 7, fill);
    conv_tr_w(&mut m, "ups.0", gen_ch, gen_next_ch, 4, fill);
    conv1d_w(&mut m, "noise_convs.0", gen_next_ch, 2 * n_bins, 1, fill);
    resblock_w(
        &mut m,
        "noise_res.0",
        gen_next_ch,
        11,
        style_dim,
        &noise_dilations,
        fill,
    );
    resblock_w(
        &mut m,
        "resblocks.0",
        gen_next_ch,
        resblock_kernel,
        style_dim,
        &resblock_dilations,
        fill,
    );
    conv1d_w(&mut m, "conv_post", 2 * n_bins, gen_next_ch, 7, fill);
    m
}

// -- Convenience wrappers -----------------------------------------------------

/// Non-zero fill convenience wrapper (fill = 0.01 for meaningful IBP bounds).
///
/// Shared between `compose_kokoro_traced.rs` and `compose_kokoro_f0_plbert.rs`.
pub(crate) fn z(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    z_fill(m, name, shape, 0.01);
}

/// Create a `BoundedTensor` with uniform asymmetric `[lo, hi]` bounds.
///
/// Thin wrapper over `asymmetric_bounds()` — shared between
/// `compose_kokoro_layerwise_d128.rs` and `compose_kokoro_layerwise_d512.rs`.
pub(crate) fn uniform_bt(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
    asymmetric_bounds(shape, lo, hi)
}

// -- Signed weight helpers (#2615) --------------------------------------------

/// Apply alternating signs to multi-dimensional weight tensors in bindings.
///
/// Leaves 1D tensors (biases, norm params), scalars, and Variable bindings
/// unchanged. Only multi-dimensional `ConstantTensor` bindings (weight matrices)
/// get sign-alternated. This transforms uniform positive weights into mixed-sign
/// weights that enable CROWN tightening over IBP.
///
/// Part of #2615.
pub(crate) fn sign_alternate_weight_bindings(bindings: &mut [TensorParamBinding]) {
    for binding in bindings {
        if let TensorParamBinding::ConstantTensor(arr) = binding {
            if arr.ndim() > 1 {
                for (i, v) in arr.iter_mut().enumerate() {
                    if i % 2 == 1 {
                        *v = -*v;
                    }
                }
            }
        }
    }
}

// -- Bounds width helpers -----------------------------------------------------

/// Extract element-wise max width from a `BoundedTensor`.
///
/// Computes `max(upper) - min(lower)` across all elements.
/// Shared between `compose_kokoro_layerwise_grouped.rs` (as `bt_max_width`)
/// and `compose_kokoro_layerwise_deep.rs` (as `bounds_max_width`).
///
/// Part of #2633.
pub(crate) fn bt_max_width(bt: &BoundedTensor) -> f32 {
    let (lo, hi) = bt.lower_upper();
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    hi_max - lo_min
}

// -- Assertion helpers --------------------------------------------------------

/// Assert all bounds in output are finite and ordered, with label for diagnostics.
///
/// Uses the more thorough version from `compose_kokoro_traced.rs` which also
/// checks `lo <= hi` ordering (catches inverted bounds from negative-weight mishandling).
pub(crate) fn assert_all_finite(output: &BoundedTensor, label: &str) {
    let (lo, hi) = output.lower_upper();
    for (idx, (&lo_val, &hi_val)) in lo.iter().zip(hi.iter()).enumerate() {
        assert!(
            lo_val.is_finite(),
            "{label} IBP lower at {idx} not finite: {lo_val}"
        );
        assert!(
            hi_val.is_finite(),
            "{label} IBP upper at {idx} not finite: {hi_val}"
        );
        assert!(
            lo_val <= hi_val,
            "{label} IBP bounds inverted at {idx}: lo={lo_val} > hi={hi_val}"
        );
    }
}

// -- BiLSTM weight builder ----------------------------------------------------

/// Insert BiLSTM weights (forward + reverse) at `prefix`.
///
/// Consolidates 2 identical `bilstm_weights()` in `compose_kokoro_pipeline_traced.rs`
/// and `compose_kokoro_prosody_traced.rs`. `hidden_dim` is the LSTM hidden size
/// (typically `d_en / 2` for BiLSTM).
///
/// Part of #2633.
pub(crate) fn bilstm_weights(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    input_dim: usize,
    hidden_dim: usize,
) {
    z_fill(
        m,
        &format!("{prefix}.weight_ih_l0"),
        &[4 * hidden_dim, input_dim],
        0.01,
    );
    z_fill(
        m,
        &format!("{prefix}.weight_hh_l0"),
        &[4 * hidden_dim, hidden_dim],
        0.01,
    );
    z_fill(m, &format!("{prefix}.bias_ih_l0"), &[4 * hidden_dim], 0.01);
    z_fill(m, &format!("{prefix}.bias_hh_l0"), &[4 * hidden_dim], 0.01);
    z_fill(
        m,
        &format!("{prefix}.weight_ih_l0_reverse"),
        &[4 * hidden_dim, input_dim],
        0.01,
    );
    z_fill(
        m,
        &format!("{prefix}.weight_hh_l0_reverse"),
        &[4 * hidden_dim, hidden_dim],
        0.01,
    );
    z_fill(
        m,
        &format!("{prefix}.bias_ih_l0_reverse"),
        &[4 * hidden_dim],
        0.01,
    );
    z_fill(
        m,
        &format!("{prefix}.bias_hh_l0_reverse"),
        &[4 * hidden_dim],
        0.01,
    );
}

// -- Generator construction ---------------------------------------------------

/// Build a test `Generator` with minimal weights and standard test config.
///
/// Consolidates 5 identical `build_test_generator()` functions scattered across
/// compose_kokoro test helpers. All use the same config (1 upsample stage,
/// 1 resblock, 1 dilation) — only `fill` varies:
/// - `fill = 0.0`: structure-only tracing (graph topology without meaningful bounds)
/// - `fill = 0.01`: meaningful IBP bounds
///
/// Part of #2633.
pub(crate) fn build_test_generator(fill: f64, style_dim: usize) -> Generator {
    let weights = generator_weights(fill, style_dim);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let mut config = KokoroConfig::default();
    config.upsample_rates = vec![2];
    config.upsample_kernel_sizes = vec![4];
    config.resblock_kernel_sizes = vec![GEN_KERNEL];
    config.resblock_dilations = vec![vec![1]];
    config.gen_initial_channels = GEN_CH;
    config.style_dim = style_dim;
    config.n_fft = GEN_N_FFT;
    Generator::load(&vb, &config).expect("Generator::load")
}
