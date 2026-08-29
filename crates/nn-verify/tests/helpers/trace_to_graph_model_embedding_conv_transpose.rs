// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Embedding and ConvTranspose2d trace-to-graph translation
//! via the `trace_to_graph_model` (LayerSpec → build_graph_network) path.
//!
//! Mirrors `trace_to_graph_embedding_conv_transpose.rs` (old `trace_to_graph_network` path)
//! to ensure equivalent coverage on the new path.

use super::common::assert_bounds_valid;
use ndarray::{ArrayD, IxDyn};
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig, Embedding,
    Module,
};
use nn_core::{DType, Device};
use nn_verify::{propagate_with_crown_fallback, trace_to_graph_model, BoundedTensor, PropMethod};

fn cpu() -> Device {
    Device::Cpu
}

/// Count nodes carrying a given NY layer type.
fn count_layer_type(graph: &nn_verify::GraphNetwork, layer_type: &str) -> usize {
    graph
        .node_names()
        .iter()
        .filter(|name| {
            graph
                .node(name)
                .is_some_and(|n| n.layer().layer_type() == layer_type)
        })
        .count()
}

/// Interval image of `w * [lo, hi]` in f64 (sign-aware endpoint swap).
fn scale_interval(w: f64, lo: f64, hi: f64) -> (f64, f64) {
    if w >= 0.0 {
        (w * lo, w * hi)
    } else {
        (w * hi, w * lo)
    }
}

/// Assert `[ibp_lo, ibp_hi]` encloses the independently computed `[lo, hi]`
/// (soundness) without being materially wider than it (tightness), and that it
/// also contains `truth` — the value nn's own eager kernel produced.
fn assert_encloses(
    label: &str,
    index: usize,
    ibp_lo: f32,
    ibp_hi: f32,
    lo: f64,
    hi: f64,
    truth: f32,
) {
    const EPS: f64 = 1e-5;
    assert!(
        ibp_lo.is_finite() && ibp_hi.is_finite(),
        "{label}[{index}] bounds must be finite, got [{ibp_lo}, {ibp_hi}]"
    );
    assert!(
        ibp_lo <= ibp_hi,
        "{label}[{index}] lower {ibp_lo} must be <= upper {ibp_hi}"
    );
    assert!(
        f64::from(ibp_lo) <= lo + EPS && f64::from(ibp_hi) >= hi - EPS,
        "{label}[{index}] bounds [{ibp_lo}, {ibp_hi}] must ENCLOSE the reachable range \
         [{lo}, {hi}]"
    );
    assert!(
        f64::from(ibp_lo) >= lo - EPS && f64::from(ibp_hi) <= hi + EPS,
        "{label}[{index}] bounds [{ibp_lo}, {ibp_hi}] are looser than the exact reachable \
         range [{lo}, {hi}]"
    );
    assert!(
        ibp_lo <= truth && truth <= ibp_hi,
        "{label}[{index}] bounds [{ibp_lo}, {ibp_hi}] must contain the eager value {truth}"
    );
}

// -- Embedding trace → graph translation with IBP bounds -----------------------

#[test]
fn test_model_trace_embedding_ibp() {
    // 3 vocabulary entries × 4 embedding dims with known values
    let weight_data = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        3.0, 4.0, 5.0, 6.0, // row 2
    ];
    let weight = DynTensor::new(&weight_data, &[3, 4], &cpu()).unwrap();
    let embedding = Embedding::new(weight).unwrap();

    // Input: single token index [1] (batch=1, seq=1)
    let x = DynTensor::from_vec_u32(vec![1], &[1, 1], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1], DType::U32).unwrap();
        x.set_trace_id(id);
        let y = embedding.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Embedding translation should succeed")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1]), 1.0_f32),
    )
    .expect("valid bounds");

    let (_method, output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.len(), 4, "expected 4 embedding dims in output");

    for &v in lo.iter() {
        assert!(v >= -10.0, "embedding lower bound unreasonably low: {v}");
        assert!(v.is_finite(), "lower bound must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 20.0, "embedding upper bound unreasonably high: {v}");
        assert!(v.is_finite(), "upper bound must be finite, got {v}");
    }
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// -- Embedding non-finite weight rejection ------------------------------------

/// A NaN embedding weight is refused by the bridge's `weight_f32` validation
/// with `NyError::NumericalInstability` naming the offending weight and index.
/// (Pinned after the ny 25dee0c5e hardening: the Embedding lowering itself now
/// succeeds via the integer-index carve-out + vocabulary-hull surrogate, so
/// this weight validation is the guard that must keep firing.)
#[test]
fn test_model_trace_embedding_non_finite_weight_rejected() {
    let weight_data = vec![
        1.0,
        2.0,
        3.0,
        f32::NAN, // row 0 has NaN
        5.0,
        6.0,
        7.0,
        8.0, // row 1
    ];
    let weight = DynTensor::new(&weight_data, &[2, 4], &cpu()).unwrap();
    let embedding = Embedding::new(weight).unwrap();

    let x = DynTensor::from_vec_u32(vec![0], &[1, 1], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1], DType::U32).unwrap();
        x.set_trace_id(id);
        let y = embedding.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let err = trace_to_graph_model(&graph).unwrap_err();
    match err {
        nn_verify::VerifyError::Ny(ny_api::NyError::NumericalInstability(msg)) => {
            assert!(
                msg.contains("Embedding weight") && msg.contains("non-finite"),
                "refusal must name the embedding weight and the non-finite value, got: {msg}"
            );
        }
        other => panic!("expected VerifyError::Ny(NumericalInstability), got: {other:?}"),
    }
}

// -- ConvTranspose2d IBP propagation ------------------------------------------

#[test]
fn test_model_trace_conv_transpose2d_ibp() {
    let weight_data = vec![1.0, 0.5, -0.5, 0.3];
    let weight = DynTensor::new(&weight_data, &[1, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default()
        .with_stride(1)
        .with_padding(0);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("ConvTranspose2d translation")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 1.0_f32),
    )
    .expect("valid bounds");

    let (_method, output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -10.0, "conv_t lower bound unreasonably low: {v}");
        assert!(v.is_finite(), "lower bound must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 10.0, "conv_t upper bound unreasonably high: {v}");
        assert!(v.is_finite(), "upper bound must be finite, got {v}");
    }
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
        assert!(l <= 0.01, "lower bound should contain 0, got {l}");
        assert!(u >= -0.01, "upper bound should contain 0, got {u}");
    }
}

// -- ConvTranspose2d CROWN propagation ----------------------------------------

#[test]
fn test_model_trace_conv_transpose2d_crown_succeeds() {
    let weight_data = vec![1.0, 0.5, -0.5, 0.3];
    let weight = DynTensor::new(&weight_data, &[1, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default()
        .with_stride(1)
        .with_padding(0);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 1.0_f32),
    )
    .expect("valid bounds");

    let (method, output, crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");

    assert_eq!(
        method,
        PropMethod::Crown,
        "Expected CROWN success, but got IBP fallback. CROWN error: {crown_err:?}"
    );

    assert_bounds_valid(&output);
}

// -- ConvTranspose2d unsupported parameters -----------------------------------

/// Grouped ConvTranspose2d is the ONE ConvTranspose2d attribute still refused
/// fail-closed. `dilation` and `output_padding` were stale gates and now
/// translate (see the two acceptance tests below, ny 3f91a42f2); grouping is a
/// real gap: ny's `ConvTranspose2dLayer` models only the dense groups=1 kernel
/// layout, so lowering a grouped `[C_in, C_out/groups, kH, kW]` kernel onto it
/// would reinterpret it as dense and produce non-enclosing bounds.
///
/// The refusal surfaces as `VerifyError::Ny(UnsupportedOp)` — the variant
/// changed with the accurate-reason rewrite — and its message states that
/// kernel-layout reason rather than a generic "unsupported configuration".
#[test]
fn test_model_trace_conv_transpose2d_groups_refused_fail_closed() {
    let weight_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let weight = DynTensor::new(&weight_data, &[2, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default().with_groups(2);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let x = DynTensor::new(&x_data, &[1, 2, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let err = trace_to_graph_model(&graph)
        .expect_err("grouped ConvTranspose2d must be refused fail-closed");
    match err {
        nn_verify::VerifyError::Ny(ny_api::NyError::UnsupportedOp(msg)) => {
            assert!(
                msg.contains("ConvTranspose2d groups=2 not supported")
                    && msg.contains("dense groups=1 kernel layout")
                    && msg.contains("not enclosing"),
                "refusal must name the group count and the dense-kernel-layout reason, got: {msg}"
            );
        }
        other => panic!("expected VerifyError::Ny(UnsupportedOp), got: {other:?}"),
    }
}

/// ConvTranspose2d `dilation` TRANSLATES (ny 3f91a42f2 removed a stale gate;
/// the layer and its converter supported dilation end to end all along).
///
/// The lowering is one native `ConvTranspose2d` layer carrying `dilation`, not
/// a decomposition — so the test pins the layer type, the dilated output
/// geometry, and IBP bounds against an f64 reference computed here.
///
/// With `stride=1, padding=0, dilation=2` and a 2x2 kernel over a 2x2 input,
/// `oh = ih + 2*kh` is a BIJECTION onto `0..4`, so every output element is a
/// single product `x[oh%2][ow%2] * k[oh/2][ow/2]` — an exact expected interval
/// with no accumulation, which makes a tightness assertion meaningful too.
#[test]
fn test_model_trace_conv_transpose2d_dilation_ibp() {
    // kernel[kh][kw], matching the [1, 1, 2, 2] weight below.
    const K: [[f64; 2]; 2] = [[1.0, 0.5], [-0.5, 0.3]];
    // x[ih][iw], the eager input the trace is recorded on.
    const X: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    const RADIUS: f64 = 0.25;

    let weight = DynTensor::new(&[1.0, 0.5, -0.5, 0.3], &[1, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default().with_dilation(2);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Dilated geometry: (2-1)*1 - 0 + 2*(2-1) + 1 = 4.
    assert_eq!(result.dims(), &[1, 1, 4, 4]);
    let eager = result.to_flat_vec::<f32>().unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("dilated ConvTranspose2d must translate")
        .graph;
    assert_eq!(
        count_layer_type(&gn, "ConvTranspose2d"),
        1,
        "dilation must ride ONE native ConvTranspose2d layer, not a decomposition"
    );

    // Input box centred on the traced point, so the eager output is reachable.
    let lower = ArrayD::from_shape_fn(IxDyn(&[1, 1, 2, 2]), |ix| (X[ix[2]][ix[3]] - RADIUS) as f32);
    let upper = ArrayD::from_shape_fn(IxDyn(&[1, 1, 2, 2]), |ix| (X[ix[2]][ix[3]] + RADIUS) as f32);
    let input_bounds = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let ibp_lo: Vec<f32> = lo.iter().copied().collect();
    let ibp_hi: Vec<f32> = hi.iter().copied().collect();
    assert_eq!(ibp_lo.len(), 16, "expected a 4x4 dilated output");

    for oh in 0..4 {
        for ow in 0..4 {
            let (ih, kh) = (oh % 2, oh / 2);
            let (iw, kw) = (ow % 2, ow / 2);
            let w = K[kh][kw];
            let (want_lo, want_hi) = scale_interval(w, X[ih][iw] - RADIUS, X[ih][iw] + RADIUS);
            let flat = oh * 4 + ow;
            assert_encloses(
                "conv_transpose2d dilation",
                flat,
                ibp_lo[flat],
                ibp_hi[flat],
                want_lo,
                want_hi,
                eager[flat],
            );
        }
    }
}

/// ConvTranspose2d `output_padding` TRANSLATES (ny 3f91a42f2). Only
/// `output_padding >= stride` — which is not a representable geometry — stays
/// refused; `stride=2, output_padding=1` is emitted as a layer attribute.
///
/// `oh = 2*ih + kh` covers `0..4` exactly once, so the interior is again a
/// single product per element, and the trailing output_padding row/column has
/// NO contributor at all — its bounds must be exactly `[0, 0]`, which is the
/// assertion that catches an output_padding silently dropped or misplaced.
#[test]
fn test_model_trace_conv_transpose2d_output_padding_ibp() {
    const K: [[f64; 2]; 2] = [[1.0, 0.5], [-0.5, 0.3]];
    const X: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    const RADIUS: f64 = 0.25;

    let weight = DynTensor::new(&[1.0, 0.5, -0.5, 0.3], &[1, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default()
        .with_stride(2)
        .with_output_padding(1);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // (2-1)*2 - 0 + 1*(2-1) + 1 + output_padding(1) = 5.
    assert_eq!(result.dims(), &[1, 1, 5, 5]);
    let eager = result.to_flat_vec::<f32>().unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("ConvTranspose2d with output_padding must translate")
        .graph;
    assert_eq!(
        count_layer_type(&gn, "ConvTranspose2d"),
        1,
        "output_padding must ride ONE native ConvTranspose2d layer"
    );

    let lower = ArrayD::from_shape_fn(IxDyn(&[1, 1, 2, 2]), |ix| (X[ix[2]][ix[3]] - RADIUS) as f32);
    let upper = ArrayD::from_shape_fn(IxDyn(&[1, 1, 2, 2]), |ix| (X[ix[2]][ix[3]] + RADIUS) as f32);
    let input_bounds = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let ibp_lo: Vec<f32> = lo.iter().copied().collect();
    let ibp_hi: Vec<f32> = hi.iter().copied().collect();
    assert_eq!(ibp_lo.len(), 25, "expected a 5x5 output_padded output");

    for oh in 0..5 {
        for ow in 0..5 {
            let flat = oh * 5 + ow;
            // The output_padding row/column has no (ih, kh) contributor: the
            // true value there is exactly 0.
            let (want_lo, want_hi) = if oh == 4 || ow == 4 {
                (0.0, 0.0)
            } else {
                let w = K[oh % 2][ow % 2];
                scale_interval(w, X[oh / 2][ow / 2] - RADIUS, X[oh / 2][ow / 2] + RADIUS)
            };
            assert_encloses(
                "conv_transpose2d output_padding",
                flat,
                ibp_lo[flat],
                ibp_hi[flat],
                want_lo,
                want_hi,
                eager[flat],
            );
        }
    }
}

// -- ConvTranspose1d with output_padding decomposition (#2558) ----------------

#[test]
fn test_model_trace_conv_transpose1d_output_padding_decomposed() {
    // ConvTranspose1d: in_ch=1, out_ch=1, kernel=3, stride=2, output_padding=1
    // Input: [1, 1, 4] → T_mid = (4-1)*2 + 3 - 0 = 9, T_out = 9 + 1 = 10
    let weight_data = vec![0.5, 1.0, -0.5];
    let weight = DynTensor::new(&weight_data, &[1, 1, 3], &cpu()).unwrap();
    let bias_data = vec![0.1];
    let bias = DynTensor::new(&bias_data, &[1], &cpu()).unwrap();

    let config = ConvTranspose1dConfig::new(0, 2, 1).with_output_padding(1);
    let conv_t = ConvTranspose1d::new(weight, Some(bias), config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Translation should succeed (no longer rejected).
    let gn = trace_to_graph_model(&graph)
        .expect("ConvTranspose1d with output_padding should translate (#2558).graph")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 4]), 1.0_f32),
    )
    .expect("valid bounds");

    let (_method, output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    // T_out = 10 elements
    assert_eq!(
        lo.len(),
        10,
        "expected 10 output elements (T_mid=9 + output_padding=1), got {}",
        lo.len()
    );

    for &v in lo.iter() {
        assert!(v.is_finite(), "lower bound must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "upper bound must be finite, got {v}");
    }
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    // The last element (from output_padding) should have bounds [0, 0]
    // since the zero-pad linear layer maps it from a zero row.
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();
    let last_lo = lo_flat[lo_flat.len() - 1];
    let last_hi = hi_flat[hi_flat.len() - 1];
    assert!(
        last_lo.abs() < 0.2 && last_hi.abs() < 0.2,
        "output_padding element should have near-zero bounds, got [{last_lo}, {last_hi}]"
    );
}
