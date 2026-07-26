// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs temporal pipeline composition.
//!
//! Two levels of composition in a single test binary:
//!
//! ## `branch` — Encoder → Decoder (no transformer)
//!
//! ```text
//! Encoder: audio [IN_CH, T] → Conv1d(stride) → GELU → DConv(×1) → Rewrite → GLU
//! Decoder: skip_add → Rewrite(Conv1d k=3) → GLU → DConv(×1) → ConvTranspose1d → GELU
//! ```
//!
//! ## `full_pipeline` — Encoder → Transformer → Decoder
//!
//! ```text
//! Encoder:     audio [IN_CH, T] → Conv1d(stride) → GELU → DConv(×1) → Rewrite → GLU
//! Transformer: Transpose [C,T]→[T,C] → LN → MHA → Residual → LN → FFN → Residual → Transpose
//! Decoder:     skip_add → Rewrite(Conv1d k=3) → GLU → DConv(×1) → ConvTranspose1d → GELU
//! ```
//!
//! Consolidates `compose_demucs_temporal_branch.rs` and
//! `compose_demucs_temporal_full.rs` into a single binary.
//!
//! Part of #779 Phase E — temporal pipeline composition.
//! Part of #1982 — test binary consolidation.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    conv_transpose_out_len, uniform_bounds, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Shared parameters
// ---------------------------------------------------------------------------

const IN_CH: usize = 4;
const ENC_CH: usize = 8;
const T_IN: usize = 16;
const ENC_KERNEL: usize = 8;
const ENC_STRIDE: usize = 4;
const ENC_PADDING: usize = ENC_KERNEL / 4;
const DCONV_COMPRESS_RATIO: usize = 4;
const DCONV_KERNEL: usize = 3;
const DCONV_DEPTH: usize = 1;
const DEC_REWRITE_KERNEL: usize = 3;
const DEC_REWRITE_PADDING: usize = DEC_REWRITE_KERNEL / 2;
const CT_KERNEL: usize = 8;
const CT_STRIDE: usize = 4;
const CT_PADDING: usize = ENC_PADDING;
const WEIGHT_MAG: f32 = 0.001;

// ---------------------------------------------------------------------------
// Shared DConv sub-layer
// ---------------------------------------------------------------------------

struct DConvInputs {
    cw: TensorNodeId,
    cb: TensorNodeId,
    ng: TensorNodeId,
    nb: TensorNodeId,
    ew: TensorNodeId,
    eb: TensorNodeId,
    eng: TensorNodeId,
    enb: TensorNodeId,
    ls: TensorNodeId,
    eps1: TensorNodeId,
    eps2: TensorNodeId,
    dilation: usize,
}

impl DConvInputs {
    fn add(b: &mut TensorBlockBuilder, pfx: &str, k: usize, ch: usize, comp: usize) -> Self {
        let d = ch * 2;
        Self {
            cw: b.add_input(&format!("{pfx}_dc{k}_cw"), &[comp, ch, DCONV_KERNEL]),
            cb: b.add_input(&format!("{pfx}_dc{k}_cb"), &[comp]),
            ng: b.add_input(&format!("{pfx}_dc{k}_ng"), &[comp]),
            nb: b.add_input(&format!("{pfx}_dc{k}_nb"), &[comp]),
            ew: b.add_input(&format!("{pfx}_dc{k}_ew"), &[d, comp, 1]),
            eb: b.add_input(&format!("{pfx}_dc{k}_eb"), &[d]),
            eng: b.add_input(&format!("{pfx}_dc{k}_eng"), &[d]),
            enb: b.add_input(&format!("{pfx}_dc{k}_enb"), &[d]),
            ls: b.add_input(&format!("{pfx}_dc{k}_ls"), &[ch]),
            eps1: b.add_input(&format!("{pfx}_dc{k}_eps"), &[1]),
            eps2: b.add_input(&format!("{pfx}_dc{k}_eps2"), &[1]),
            dilation: 1 << k,
        }
    }
}

/// Conv1d(dilated) → GN(G=1) → GELU → Conv1d(1×1) → GN(G=1) → GLU → LayerScale → residual
fn build_dconv(
    b: &mut TensorBlockBuilder,
    x: TensorNodeId,
    dc: &DConvInputs,
    ch: usize,
    comp: usize,
    t: usize,
) -> TensorNodeId {
    let d = ch * 2;
    let pad = dc.dilation * (DCONV_KERNEL - 1) / 2;
    let c1 = b.add_conv1d_full(x, dc.cw, Some(dc.cb), 1, pad, dc.dilation, 1, &[comp, t]);
    let n1 = b.add_group_norm_g1(c1, dc.eps1, Some(dc.ng), Some(dc.nb), comp, t);
    let g1 = b.add_gelu(n1, &[comp, t]);
    let c2 = b.add_conv1d(g1, dc.ew, Some(dc.eb), 1, 0, &[d, t]);
    let n2 = b.add_group_norm_g1(c2, dc.eps2, Some(dc.eng), Some(dc.enb), d, t);
    let glu = b.add_glu(n2, 0, &[d, t]).expect("even dim");
    let ls = b.add_layer_scale(glu, dc.ls, &[ch, t]);
    b.add_binary_add(x, ls, &[ch, t])
}

// ---------------------------------------------------------------------------
// Shared binding helpers
// ---------------------------------------------------------------------------

fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

fn add_dconv_bindings(b: &mut Vec<TensorParamBinding>, ch: usize, comp: usize) {
    let d = ch * 2;
    push_weight(b, &[comp, ch, DCONV_KERNEL], WEIGHT_MAG);
    push_weight(b, &[comp], 0.0);
    push_weight(b, &[comp], 1.0);
    push_weight(b, &[comp], 0.0);
    push_weight(b, &[d, comp, 1], WEIGHT_MAG);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[d], 1.0);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[ch], 0.1);
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b.push(TensorParamBinding::ConstantScalar(1e-5));
}

// ===========================================================================
// branch: Encoder → Decoder (no transformer)
// ===========================================================================

mod branch {
    use super::*;

    fn build_temporal_branch() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
        let comp = ENC_CH / DCONV_COMPRESS_RATIO;
        let dbl = ENC_CH * 2;
        let mut b = TensorBlockBuilder::new("demucs_temporal_branch_verify");

        let audio = b.add_input("audio", &[IN_CH, T_IN]);
        let ecw = b.add_input("enc_conv_w", &[ENC_CH, IN_CH, ENC_KERNEL]);
        let ecb = b.add_input("enc_conv_b", &[ENC_CH]);
        let enc_dc: Vec<_> = (0..DCONV_DEPTH)
            .map(|k| DConvInputs::add(&mut b, "enc", k, ENC_CH, comp))
            .collect();
        let erw = b.add_input("enc_rw_w", &[dbl, ENC_CH, 1]);
        let erb = b.add_input("enc_rw_b", &[dbl]);
        let drw = b.add_input("dec_rw_w", &[dbl, ENC_CH, DEC_REWRITE_KERNEL]);
        let drb = b.add_input("dec_rw_b", &[dbl]);
        let dec_dc: Vec<_> = (0..DCONV_DEPTH)
            .map(|k| DConvInputs::add(&mut b, "dec", k, ENC_CH, comp))
            .collect();
        let dctw = b.add_input("dec_ct_w", &[ENC_CH, IN_CH, CT_KERNEL]);
        let dctb = b.add_input("dec_ct_b", &[IN_CH]);

        // === Encoder forward ===
        let t_enc = conv1d_out_len(T_IN, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);
        let x = b.add_conv1d(
            audio,
            ecw,
            Some(ecb),
            ENC_STRIDE,
            ENC_PADDING,
            &[ENC_CH, t_enc],
        );
        let x = b.add_gelu(x, &[ENC_CH, t_enc]);
        let mut x = x;
        for di in &enc_dc {
            x = build_dconv(&mut b, x, di, ENC_CH, comp, t_enc);
        }
        let x = b.add_conv1d(x, erw, Some(erb), 1, 0, &[dbl, t_enc]);
        let enc_out = b.add_glu(x, 0, &[dbl, t_enc]).expect("encoder GLU");

        // === Decoder forward ===
        let x = b.add_binary_add(enc_out, enc_out, &[ENC_CH, t_enc]);
        let rw_t = conv1d_out_len(t_enc, DEC_REWRITE_KERNEL, 1, DEC_REWRITE_PADDING);
        let x = b.add_conv1d(x, drw, Some(drb), 1, DEC_REWRITE_PADDING, &[dbl, rw_t]);
        let x = b.add_glu(x, 0, &[dbl, rw_t]).expect("decoder GLU");
        let mut x = x;
        for di in &dec_dc {
            x = build_dconv(&mut b, x, di, ENC_CH, comp, rw_t);
        }
        let ct_t = conv_transpose_out_len(rw_t, CT_STRIDE, CT_KERNEL, CT_PADDING);
        let x = b.add_conv_transpose_1d(
            x,
            dctw,
            Some(dctb),
            CT_STRIDE,
            CT_PADDING,
            1,
            1,
            0,
            &[IN_CH, ct_t],
        );
        let target_t = T_IN.min(ct_t);
        let x = if ct_t > target_t {
            b.add_narrow(x, 1, 0, target_t, &[IN_CH, target_t])
        } else {
            x
        };
        let out = b.add_gelu(x, &[IN_CH, target_t]);

        (b.build(out).expect("valid temporal branch graph"), target_t)
    }

    fn temporal_branch_bindings() -> Vec<TensorParamBinding> {
        let comp = ENC_CH / DCONV_COMPRESS_RATIO;
        let dbl = ENC_CH * 2;
        let mut b = Vec::new();

        b.push(TensorParamBinding::Variable);
        push_weight(&mut b, &[ENC_CH, IN_CH, ENC_KERNEL], WEIGHT_MAG);
        push_weight(&mut b, &[ENC_CH], 0.0);
        for _ in 0..DCONV_DEPTH {
            add_dconv_bindings(&mut b, ENC_CH, comp);
        }
        push_weight(&mut b, &[dbl, ENC_CH, 1], WEIGHT_MAG);
        push_weight(&mut b, &[dbl], 0.0);
        push_weight(&mut b, &[dbl, ENC_CH, DEC_REWRITE_KERNEL], WEIGHT_MAG);
        push_weight(&mut b, &[dbl], 0.0);
        for _ in 0..DCONV_DEPTH {
            add_dconv_bindings(&mut b, ENC_CH, comp);
        }
        push_weight(&mut b, &[ENC_CH, IN_CH, CT_KERNEL], WEIGHT_MAG);
        push_weight(&mut b, &[IN_CH], 0.0);

        b
    }

    #[test]
    fn test_temporal_branch_def_validates() {
        let (def, _) = build_temporal_branch();
        def.validate().expect("temporal branch def should validate");
    }

    #[test]
    fn test_temporal_branch_graph_builds() {
        let (def, target_t) = build_temporal_branch();

        let t_enc = conv1d_out_len(T_IN, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);
        assert_eq!(t_enc, 4, "encoder Conv1d output T");
        let rw_t = conv1d_out_len(t_enc, DEC_REWRITE_KERNEL, 1, DEC_REWRITE_PADDING);
        assert_eq!(rw_t, t_enc, "decoder rewrite preserves T");
        let ct_t = conv_transpose_out_len(rw_t, CT_STRIDE, CT_KERNEL, CT_PADDING);
        assert_eq!(ct_t, T_IN, "ConvTranspose1d restores original T");
        assert_eq!(target_t, T_IN, "output temporal length matches input");

        let bindings = temporal_branch_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings)
            .expect("temporal branch graph should translate");

        assert!(
            graph.num_nodes() >= 30,
            "graph should have >= 30 nodes, got {}",
            graph.num_nodes()
        );
    }

    #[test]
    fn test_temporal_branch_ibp_propagates() {
        let (def, target_t) = build_temporal_branch();
        let bindings = temporal_branch_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through temporal branch");
        assert_eq!(
            output.lower_upper().0.shape(),
            &[IN_CH, target_t],
            "output shape mismatch"
        );
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("Temporal branch IBP: bounds=[{lo_min}, {hi_max}]");

        assert!(
            lo_min.abs() < 1.0,
            "IBP lower bound magnitude < 1.0, got {lo_min}"
        );
        assert!(
            hi_max.abs() < 1.0,
            "IBP upper bound magnitude < 1.0, got {hi_max}"
        );
    }

    #[test]
    fn test_temporal_branch_crown_propagation() {
        let (def, target_t) = build_temporal_branch();
        let bindings = temporal_branch_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

        let (method, output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);
        let (lo, _) = output.lower_upper();

        assert_eq!(lo.shape(), &[IN_CH, target_t], "output shape mismatch");
        assert_bounds_valid(&output);

        eprintln!("Temporal branch: method={method:?}");
        if let Some(reason) = &fallback_reason {
            eprintln!("CROWN fallback reason: {reason}");
        }
    }

    #[test]
    fn test_temporal_branch_autoencoder_shape() {
        let (def, target_t) = build_temporal_branch();
        let bindings = temporal_branch_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through temporal branch");
        let (lo, _) = output.lower_upper();

        assert_eq!(
            lo.shape(),
            &[IN_CH, T_IN],
            "output shape should match input"
        );
        assert_eq!(target_t, T_IN, "target temporal length matches input");
    }

    #[test]
    fn test_temporal_branch_verify_and_record() {
        let (def, target_t) = build_temporal_branch();
        let bindings = temporal_branch_bindings();
        let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

        let result = verify_and_assert(&def, &bindings, &input, "demucs_temporal_branch");
        assert_eq!(result.num_variables, 1, "single Variable input (audio)");

        let (lo, _) = result.output_bounds.lower_upper();
        assert_eq!(lo.shape(), &[IN_CH, target_t]);
    }
}

// ===========================================================================
// full_pipeline: Encoder → Transformer → Decoder
// ===========================================================================

mod full_pipeline {
    use super::*;
    use nn_dsl::{AttentionMask, TransformerBlockConfig, TransformerBlockWeights};
    use nn_verify::VerificationSoundnessMode;

    const NUM_HEADS: usize = 2;
    const FFN_HIDDEN: usize = ENC_CH * 2;

    struct FullPipelineInputs {
        audio: TensorNodeId,
        ecw: TensorNodeId,
        ecb: TensorNodeId,
        enc_dc: Vec<DConvInputs>,
        erw: TensorNodeId,
        erb: TensorNodeId,
        tw: TransformerBlockWeights,
        drw: TensorNodeId,
        drb: TensorNodeId,
        dec_dc: Vec<DConvInputs>,
        dctw: TensorNodeId,
        dctb: TensorNodeId,
    }

    fn add_full_inputs(b: &mut TensorBlockBuilder) -> FullPipelineInputs {
        let comp = ENC_CH / DCONV_COMPRESS_RATIO;
        let dbl = ENC_CH * 2;
        let d = ENC_CH;
        let audio = b.add_input("audio", &[IN_CH, T_IN]);
        let ecw = b.add_input("enc_conv_w", &[ENC_CH, IN_CH, ENC_KERNEL]);
        let ecb = b.add_input("enc_conv_b", &[ENC_CH]);
        let enc_dc: Vec<_> = (0..DCONV_DEPTH)
            .map(|k| DConvInputs::add(b, "enc", k, ENC_CH, comp))
            .collect();
        let erw = b.add_input("enc_rw_w", &[dbl, ENC_CH, 1]);
        let erb = b.add_input("enc_rw_b", &[dbl]);
        let tw = TransformerBlockWeights {
            ln1_weight: b.add_input("tf_ln1_w", &[d]),
            ln1_bias: b.add_input("tf_ln1_b", &[d]),
            ln2_weight: b.add_input("tf_ln2_w", &[d]),
            ln2_bias: b.add_input("tf_ln2_b", &[d]),
            q_weight: b.add_input("tf_q_w", &[d, d]),
            k_weight: b.add_input("tf_k_w", &[d, d]),
            v_weight: b.add_input("tf_v_w", &[d, d]),
            out_weight: b.add_input("tf_out_w", &[d, d]),
            ffn1_weight: b.add_input("tf_ffn1_w", &[FFN_HIDDEN, d]),
            ffn2_weight: b.add_input("tf_ffn2_w", &[d, FFN_HIDDEN]),
            eps: b.add_input("tf_eps", &[1]),
        };
        let drw = b.add_input("dec_rw_w", &[dbl, ENC_CH, DEC_REWRITE_KERNEL]);
        let drb = b.add_input("dec_rw_b", &[dbl]);
        let dec_dc: Vec<_> = (0..DCONV_DEPTH)
            .map(|k| DConvInputs::add(b, "dec", k, ENC_CH, comp))
            .collect();
        let dctw = b.add_input("dec_ct_w", &[ENC_CH, IN_CH, CT_KERNEL]);
        let dctb = b.add_input("dec_ct_b", &[IN_CH]);
        FullPipelineInputs {
            audio,
            ecw,
            ecb,
            enc_dc,
            erw,
            erb,
            tw,
            drw,
            drb,
            dec_dc,
            dctw,
            dctb,
        }
    }

    fn build_temporal_full() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
        let comp = ENC_CH / DCONV_COMPRESS_RATIO;
        let dbl = ENC_CH * 2;
        let d = ENC_CH;
        let mut b = TensorBlockBuilder::new("demucs_temporal_full_verify");
        let inp = add_full_inputs(&mut b);

        // Encoder forward
        let t_enc = conv1d_out_len(T_IN, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);
        let x = b.add_conv1d(
            inp.audio,
            inp.ecw,
            Some(inp.ecb),
            ENC_STRIDE,
            ENC_PADDING,
            &[ENC_CH, t_enc],
        );
        let x = b.add_gelu(x, &[ENC_CH, t_enc]);
        let mut x = x;
        for di in &inp.enc_dc {
            x = build_dconv(&mut b, x, di, ENC_CH, comp, t_enc);
        }
        let x = b.add_conv1d(x, inp.erw, Some(inp.erb), 1, 0, &[dbl, t_enc]);
        let enc_out = b.add_glu(x, 0, &[dbl, t_enc]).expect("encoder GLU");

        // Transformer bottleneck: [C, T] → [T, C] → transformer → [C, T]
        let x_t = b.add_transpose(enc_out, &[1, 0], &[t_enc, d]);
        let tc = TransformerBlockConfig {
            num_heads: NUM_HEADS,
            mask: AttentionMask::Standard,
            ffn_hidden_dim: FFN_HIDDEN,
        };
        let x_t = b
            .add_transformer_block(x_t, &inp.tw, &tc)
            .expect("transformer block");
        let x = b.add_transpose(x_t, &[1, 0], &[d, t_enc]);

        // Decoder forward: skip + transform → decode
        let x = b.add_binary_add(x, enc_out, &[ENC_CH, t_enc]);
        let rw_t = conv1d_out_len(t_enc, DEC_REWRITE_KERNEL, 1, DEC_REWRITE_PADDING);
        let x = b.add_conv1d(
            x,
            inp.drw,
            Some(inp.drb),
            1,
            DEC_REWRITE_PADDING,
            &[dbl, rw_t],
        );
        let x = b.add_glu(x, 0, &[dbl, rw_t]).expect("decoder GLU");
        let mut x = x;
        for di in &inp.dec_dc {
            x = build_dconv(&mut b, x, di, ENC_CH, comp, rw_t);
        }
        let ct_t = conv_transpose_out_len(rw_t, CT_STRIDE, CT_KERNEL, CT_PADDING);
        let x = b.add_conv_transpose_1d(
            x,
            inp.dctw,
            Some(inp.dctb),
            CT_STRIDE,
            CT_PADDING,
            1,
            1,
            0,
            &[IN_CH, ct_t],
        );
        let target_t = T_IN.min(ct_t);
        let x = if ct_t > target_t {
            b.add_narrow(x, 1, 0, target_t, &[IN_CH, target_t])
        } else {
            x
        };
        let out = b.add_gelu(x, &[IN_CH, target_t]);

        (b.build(out).expect("valid temporal full graph"), target_t)
    }

    fn add_transformer_bindings(b: &mut Vec<TensorParamBinding>) {
        let d = ENC_CH;
        push_weight(b, &[d], 1.0);
        push_weight(b, &[d], 0.0);
        push_weight(b, &[d], 1.0);
        push_weight(b, &[d], 0.0);
        push_weight(b, &[d, d], WEIGHT_MAG);
        push_weight(b, &[d, d], WEIGHT_MAG);
        push_weight(b, &[d, d], WEIGHT_MAG);
        push_weight(b, &[d, d], WEIGHT_MAG);
        push_weight(b, &[FFN_HIDDEN, d], WEIGHT_MAG);
        push_weight(b, &[d, FFN_HIDDEN], WEIGHT_MAG);
        b.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    fn temporal_full_bindings() -> Vec<TensorParamBinding> {
        let comp = ENC_CH / DCONV_COMPRESS_RATIO;
        let dbl = ENC_CH * 2;
        let mut b = Vec::new();

        b.push(TensorParamBinding::Variable);
        push_weight(&mut b, &[ENC_CH, IN_CH, ENC_KERNEL], WEIGHT_MAG);
        push_weight(&mut b, &[ENC_CH], 0.0);
        for _ in 0..DCONV_DEPTH {
            add_dconv_bindings(&mut b, ENC_CH, comp);
        }
        push_weight(&mut b, &[dbl, ENC_CH, 1], WEIGHT_MAG);
        push_weight(&mut b, &[dbl], 0.0);
        add_transformer_bindings(&mut b);
        push_weight(&mut b, &[dbl, ENC_CH, DEC_REWRITE_KERNEL], WEIGHT_MAG);
        push_weight(&mut b, &[dbl], 0.0);
        for _ in 0..DCONV_DEPTH {
            add_dconv_bindings(&mut b, ENC_CH, comp);
        }
        push_weight(&mut b, &[ENC_CH, IN_CH, CT_KERNEL], WEIGHT_MAG);
        push_weight(&mut b, &[IN_CH], 0.0);

        b
    }

    #[test]
    fn test_full_pipeline_def_validates() {
        let (def, _) = build_temporal_full();
        def.validate().expect("full pipeline def should validate");
    }

    #[test]
    fn test_full_pipeline_graph_builds() {
        let (def, target_t) = build_temporal_full();
        assert_eq!(target_t, T_IN, "output temporal length matches input");

        let bindings = temporal_full_bindings();
        let graph =
            tensor_kernel_to_graph(&def, &bindings).expect("full pipeline graph should translate");

        assert!(
            graph.num_nodes() >= 50,
            "full pipeline graph should have >= 50 nodes, got {}",
            graph.num_nodes()
        );
    }

    #[test]
    fn test_full_pipeline_ibp_propagates() {
        let (def, target_t) = build_temporal_full();
        let bindings = temporal_full_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through full pipeline");
        assert_eq!(
            output.lower_upper().0.shape(),
            &[IN_CH, target_t],
            "output shape mismatch"
        );
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("Full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

        assert!(
            lo_min.abs() < 1.0,
            "IBP lower bound magnitude < 1.0, got {lo_min}"
        );
        assert!(
            hi_max.abs() < 1.0,
            "IBP upper bound magnitude < 1.0, got {hi_max}"
        );
    }

    #[test]
    fn test_full_pipeline_crown_propagation() {
        let (def, target_t) = build_temporal_full();
        let bindings = temporal_full_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

        let (method, output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);
        assert_eq!(
            output.lower_upper().0.shape(),
            &[IN_CH, target_t],
            "output shape mismatch"
        );
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);

        eprintln!("Full pipeline: method={method:?}, bounds=[{lo_min}, {hi_max}]");
        if let Some(reason) = &fallback_reason {
            eprintln!("CROWN fallback reason: {reason}");
        }

        assert!(
            lo_min.abs() < 1.0,
            "CROWN: lower bound magnitude < 1.0, got {lo_min}"
        );
        assert!(
            hi_max.abs() < 1.0,
            "CROWN: upper bound magnitude < 1.0, got {hi_max}"
        );
    }

    #[test]
    fn test_full_pipeline_verify_and_record() {
        let (def, target_t) = build_temporal_full();
        let bindings = temporal_full_bindings();
        let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

        let result = verify_and_assert(&def, &bindings, &input, "demucs_temporal_full");
        assert_eq!(result.num_variables, 1, "single Variable input (audio)");

        let (lo, _) = result.output_bounds.lower_upper();
        assert_eq!(lo.shape(), &[IN_CH, target_t]);

        assert_eq!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Heuristic,
            "ForwardMode NormBoundsMode should produce Heuristic, got {:?}",
            result.verification.soundness_mode
        );
    }
}
