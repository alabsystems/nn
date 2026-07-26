// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Traced FullDecoder Stage 1 NY verification tests.
//!
//! Verifies the Stage1ResBlk + encode/decode pipeline via IBP propagation.
//! Stage 1 ops: AdaIn (InstanceNorm + Linear + affine), LeakyReLU(0.2),
//! Conv1d, Cat (skip connections), Add, MulScalar(1/sqrt(2)).
//!
//! The upsample block (ConvTranspose1d with groups>1 + output_padding=1) and
//! Generator are excluded — they have their own verification tests.
//! Grouped ConvTranspose1d and output_padding are now supported (#2989, #2558).
//!
//! Part of #2520.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, Module};
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_models::kokoro_full_decoder::Stage1ResBlk;
use nn_verify::{trace_to_graph_model_multi_input, BoundedTensor};
use std::collections::HashMap;

use super::common::kokoro_recording::record_ibp_result;
use super::common::kokoro_weights::{assert_all_finite, conv1d_w, propagate_multi_input_ibp, z};

// -- Test-sized dimensions ----------------------------------------------------

const D_EN: usize = 8;
const STYLE_DIM: usize = 4;
const HIDDEN: usize = 2 * D_EN;
const ASR_RES_CH: usize = D_EN / 8;
const ENCODE_IN: usize = D_EN + 2;
const DECODE_IN: usize = HIDDEN + ASR_RES_CH + 2;
const T: usize = 4;

// -- Weight helpers -----------------------------------------------------------

/// Insert Stage1ResBlk weights at `prefix` (non-upsample only).
fn stage1_resblk_w(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
) {
    conv1d_w(m, &format!("{prefix}.conv1"), dim_out, dim_in, 3, 0.01);
    conv1d_w(m, &format!("{prefix}.conv2"), dim_out, dim_out, 3, 0.01);
    z(
        m,
        &format!("{prefix}.norm1.style_linear.weight"),
        &[2 * dim_in, style_dim],
    );
    z(
        m,
        &format!("{prefix}.norm1.style_linear.bias"),
        &[2 * dim_in],
    );
    z(
        m,
        &format!("{prefix}.norm2.style_linear.weight"),
        &[2 * dim_out, style_dim],
    );
    z(
        m,
        &format!("{prefix}.norm2.style_linear.bias"),
        &[2 * dim_out],
    );
    if dim_in != dim_out {
        conv1d_w(m, &format!("{prefix}.conv1x1"), dim_out, dim_in, 1, 0.01);
    }
}

fn trace_input(t: &DynTensor) -> DynTensor {
    let mut out = t.clone();
    out.set_trace_id(record_input(out.dims(), DType::F32).expect("tracing active"));
    out
}

/// Record a traced segment's IBP result to the per-model status file.
///
/// Uses `record_pipeline` with `load_locked` + `save` for concurrent safety.
fn assert_bounds_non_vacuous(output: &BoundedTensor, label: &str) {
    assert_all_finite(output, label);
    let (lo, hi) = output.lower_upper();
    let max_width: f32 = hi
        .iter()
        .zip(lo.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width < 1e6,
        "{label}: max bound width {max_width} exceeds 1e6 (vacuously wide)"
    );
    assert!(
        max_width > 0.0,
        "{label}: zero-width bounds suggest degenerate model"
    );
    eprintln!(
        "{label} IBP: {} output elements, max_width={max_width:.4}",
        lo.len()
    );
}

// -- Test 1: Single Stage1ResBlk block ----------------------------------------

#[test]
fn test_stage1_resblk_single_block_ibp() {
    let mut weights = HashMap::new();
    stage1_resblk_w(&mut weights, "blk", ENCODE_IN, HIDDEN, STYLE_DIM);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let blk = Stage1ResBlk::load(vb.pp("blk"), ENCODE_IN, HIDDEN, STYLE_DIM, false)
        .expect("Stage1ResBlk load");

    let x_shape = [1, ENCODE_IN, T];
    let style_shape = [1, STYLE_DIM];
    let x = DynTensor::full(&x_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_out, graph) = trace_graph(|| {
        let xin = trace_input(&x);
        let sin = trace_input(&style);
        blk.forward(&xin, &sin)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))
    })
    .expect("Stage1ResBlk trace");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("trace_to_graph")
        .graph;
    let output = propagate_multi_input_ibp(
        &gn,
        &[(&x_shape[..], (-1.0, 1.0)), (&style_shape[..], (-0.5, 0.5))],
    );
    assert_bounds_non_vacuous(&output, "Stage1ResBlk");
}

// -- Test 2: Stage 1 encode + decode pipeline ---------------------------------

/// Stage 1 components loaded from weights (extracted for function-size limit).
struct Stage1Pipeline {
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res_conv: Conv1d,
    encode: Stage1ResBlk,
    decode_blocks: Vec<Stage1ResBlk>,
}

/// Build weights and load all Stage 1 components (encode + 3 decode blocks).
fn build_stage1_pipeline() -> Stage1Pipeline {
    let mut weights = HashMap::new();
    conv1d_w(&mut weights, "F0_conv", 1, 1, 3, 0.01);
    conv1d_w(&mut weights, "N_conv", 1, 1, 3, 0.01);
    conv1d_w(&mut weights, "asr_res", ASR_RES_CH, D_EN, 1, 0.01);
    stage1_resblk_w(&mut weights, "encode", ENCODE_IN, HIDDEN, STYLE_DIM);
    for i in 0..3 {
        stage1_resblk_w(
            &mut weights,
            &format!("decode.{i}"),
            DECODE_IN,
            HIDDEN,
            STYLE_DIM,
        );
    }

    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let f0n_cfg = Conv1dConfig::default().with_padding(1).with_stride(2);
    let f0_conv = Conv1d::new(
        vb.get(&[1, 1, 3], "F0_conv.weight").unwrap(),
        Some(vb.get(&[1], "F0_conv.bias").unwrap()),
        f0n_cfg,
    )
    .unwrap();
    let n_conv = Conv1d::new(
        vb.get(&[1, 1, 3], "N_conv.weight").unwrap(),
        Some(vb.get(&[1], "N_conv.bias").unwrap()),
        f0n_cfg,
    )
    .unwrap();
    let asr_res_conv = Conv1d::new(
        vb.get(&[ASR_RES_CH, D_EN, 1], "asr_res.weight").unwrap(),
        Some(vb.get(&[ASR_RES_CH], "asr_res.bias").unwrap()),
        Conv1dConfig::default(),
    )
    .unwrap();
    let encode = Stage1ResBlk::load(vb.pp("encode"), ENCODE_IN, HIDDEN, STYLE_DIM, false).unwrap();
    let mut decode_blocks = Vec::new();
    for i in 0..3 {
        decode_blocks.push(
            Stage1ResBlk::load(
                vb.pp(format!("decode.{i}")),
                DECODE_IN,
                HIDDEN,
                STYLE_DIM,
                false,
            )
            .unwrap(),
        );
    }
    Stage1Pipeline {
        f0_conv,
        n_conv,
        asr_res_conv,
        encode,
        decode_blocks,
    }
}

/// Trace Stage 1 pipeline and verify IBP bounds are finite and non-vacuous.
///
/// Full Stage 1 data-flow: F0/N downsample → cat → encode → skip → 3 decode.
/// Excludes decode.3 upsample (ConvTranspose1d groups>1) and Generator.
#[test]
fn test_full_decoder_stage1_pipeline_ibp() {
    let p = build_stage1_pipeline();

    let asr_shape = [1, D_EN, T];
    let f0_shape = [1, 1, 2 * T];
    let n_shape = [1, 1, 2 * T];
    let style_shape = [1, STYLE_DIM];

    let asr = DynTensor::full(&asr_shape, 0.1, DType::F32, &cpu()).unwrap();
    let f0_in = DynTensor::full(&f0_shape, 0.05, DType::F32, &cpu()).unwrap();
    let n_in = DynTensor::full(&n_shape, 0.02, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_out, graph) = trace_graph(|| {
        let (asr_t, f0_t, n_t, style_t) = (
            trace_input(&asr),
            trace_input(&f0_in),
            trace_input(&n_in),
            trace_input(&style),
        );
        let f0 = p.f0_conv.forward(&f0_t)?;
        let n = p.n_conv.forward(&n_t)?;
        let enc_input = DynTensor::cat(&[&asr_t, &f0, &n], 1)?;
        let mut x = p
            .encode
            .forward(&enc_input, &style_t)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        let asr_skip = p.asr_res_conv.forward(&asr_t)?;
        for blk in &p.decode_blocks {
            let skip_input = DynTensor::cat(&[&x, &asr_skip, &f0, &n], 1)?;
            x = blk
                .forward(&skip_input, &style_t)
                .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        }
        Ok(x)
    })
    .expect("Stage1 pipeline trace");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("trace_to_graph")
        .graph;
    let input_specs: &[(&[usize], (f32, f32))] = &[
        (&asr_shape[..], (-1.0, 1.0)),
        (&f0_shape[..], (0.0, 0.5)),
        (&n_shape[..], (0.0, 0.3)),
        (&style_shape[..], (-0.5, 0.5)),
    ];
    let output = propagate_multi_input_ibp(&gn, input_specs);
    assert_bounds_non_vacuous(&output, "FullDecoder_Stage1");

    // Build combined input bounds and record to status file.
    // This clears the stale `kokoro_full_decoder_stage1` entry (#2591).
    let mut in_lo = Vec::new();
    let mut in_hi = Vec::new();
    for &(shape, (lo, hi)) in input_specs {
        let flat: usize = shape.iter().product();
        in_lo.extend(vec![lo; flat]);
        in_hi.extend(vec![hi; flat]);
    }
    let total = in_lo.len();
    let input_bounds = BoundedTensor::new(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[total]), in_lo).unwrap(),
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[total]), in_hi).unwrap(),
    )
    .expect("valid input bounds");
    record_ibp_result("kokoro_full_decoder_stage1", &input_bounds, &output);
}
