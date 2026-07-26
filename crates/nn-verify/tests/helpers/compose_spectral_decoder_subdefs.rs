// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs spectral decoder sub-def composition with NY.
//!
//! Consolidates tests from three former files into one binary:
//! - `compose_spectral_decoder_block.rs` (5 tests) — Rewrite: Conv2d(3×3) → GLU
//! - `compose_spectral_dconv.rs` (4 tests) — DConv residual layers
//! - `compose_spectral_conv_tr.rs` (5 tests) — ConvTranspose1d + Narrow(trim)
//!
//! Each sub-def is built with `TensorBlockBuilder` (manual builder), matching
//! the production topology but at small scale for tractability.
//!
//! Part of #1982: nn-verify test binary consolidation.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_than_ibp, assert_crown_tighter_when_not_fallback,
    bounds_min_max, conv_transpose_out_len, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// =========================================================================
// REWRITE SUB-DEF (from compose_spectral_decoder_block.rs)
// =========================================================================

mod rewrite {
    use super::*;

    // Parameters
    const CHANNELS: usize = 4;
    const FREQ: usize = 4;
    const TIME: usize = 4;
    const REWRITE_KERNEL: usize = 3;
    const REWRITE_PADDING: usize = REWRITE_KERNEL / 2;

    /// Build a spectral decoder rewrite sub-def using TensorBlockBuilder.
    ///
    /// skip_add → Reshape[C,F,T] → Conv2d(3×3, s=1, p=1) → Reshape[2C,F*T] → GLU
    fn build_spectral_rewrite() -> nn_dsl::tensor_ir::TensorKernelDef {
        let doubled = CHANNELS * 2;
        let ft = FREQ * TIME;
        let out_f = FREQ + 2 * REWRITE_PADDING - REWRITE_KERNEL + 1;
        let out_t = TIME + 2 * REWRITE_PADDING - REWRITE_KERNEL + 1;
        let out_ft = out_f * out_t;

        let mut b = TensorBlockBuilder::new("spec_dec_rewrite_verify");

        let data = b.add_input("data", &[CHANNELS, ft]);
        let skip = b.add_input("skip", &[CHANNELS, ft]);
        let rw_weight = b.add_input(
            "rw_weight",
            &[doubled, CHANNELS, REWRITE_KERNEL, REWRITE_KERNEL],
        );
        let rw_bias = b.add_input("rw_bias", &[doubled]);

        let x = b.add_binary_add(data, skip, &[CHANNELS, ft]);
        let x_3d = b.add_reshape(x, &[CHANNELS, FREQ, TIME]);
        let conv_out = b.add_conv2d(
            x_3d,
            rw_weight,
            Some(rw_bias),
            1,
            1,
            REWRITE_PADDING,
            REWRITE_PADDING,
            &[doubled, out_f, out_t],
        );
        let conv_flat = b.add_reshape(conv_out, &[doubled, out_ft]);
        let glu_out = b
            .add_glu(conv_flat, 0, &[doubled, out_ft])
            .expect("even dim");

        b.build(glu_out).expect("valid graph")
    }

    fn bindings() -> Vec<TensorParamBinding> {
        let doubled = CHANNELS * 2;
        let ft = FREQ * TIME;
        let mut bindings = Vec::new();

        bindings.push(TensorParamBinding::Variable);
        let skip = ArrayD::from_elem(IxDyn(&[CHANNELS, ft]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(skip));
        let rw_w = ArrayD::from_elem(
            IxDyn(&[doubled, CHANNELS, REWRITE_KERNEL, REWRITE_KERNEL]),
            0.02f32,
        );
        bindings.push(TensorParamBinding::ConstantTensor(rw_w));
        let rw_b = ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(rw_b));

        bindings
    }

    fn input_bounds() -> BoundedTensor {
        let ft = FREQ * TIME;
        uniform_bounds(&[CHANNELS, ft], 1.0)
    }

    #[test]
    fn test_spectral_rewrite_def_validates() {
        let def = build_spectral_rewrite();
        def.validate()
            .expect("spectral rewrite def should validate");
    }

    #[test]
    fn test_spectral_rewrite_graph_builds() {
        let def = build_spectral_rewrite();
        let b = bindings();
        let graph =
            tensor_kernel_to_graph(&def, &b).expect("spectral rewrite graph should translate");
        assert!(
            graph.num_nodes() > 0,
            "spectral rewrite graph should be non-empty, got {}",
            graph.num_nodes()
        );
    }

    #[test]
    fn test_spectral_rewrite_ibp_propagates() {
        let ft = FREQ * TIME;
        let def = build_spectral_rewrite();
        let b = bindings();
        let graph = tensor_kernel_to_graph(&def, &b).expect("graph translation");
        let input = input_bounds();

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through spectral rewrite");
        let (lo, _) = output.lower_upper();
        assert_eq!(lo.shape(), &[CHANNELS, ft], "output shape mismatch");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("Spectral rewrite IBP bounds: [{lo_min}, {hi_max}] over {CHANNELS}×{ft} output");
    }

    #[test]
    fn test_spectral_rewrite_crown_propagation() {
        let ft = FREQ * TIME;
        let def = build_spectral_rewrite();
        let b = bindings();
        let graph = tensor_kernel_to_graph(&def, &b).expect("graph translation");
        let input = input_bounds();

        let (method, output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);
        let (lo, _) = output.lower_upper();
        assert_eq!(lo.shape(), &[CHANNELS, ft], "output shape mismatch");

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("Spectral rewrite: method={method:?}, bounds=[{lo_min}, {hi_max}]");
        if let Some(reason) = &fallback_reason {
            eprintln!("CROWN fallback reason: {reason}");
        }
    }

    #[test]
    fn test_spectral_rewrite_bounds_finite() {
        let def = build_spectral_rewrite();
        let b = bindings();
        let graph = tensor_kernel_to_graph(&def, &b).expect("graph translation");
        let input = input_bounds();

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through spectral rewrite");
        let (lo_min, hi_max) = bounds_min_max(&output);
        assert!(
            lo_min.is_finite(),
            "lower bound should be finite, got {lo_min}"
        );
        assert!(
            hi_max.is_finite(),
            "upper bound should be finite, got {hi_max}"
        );
    }
}

// =========================================================================
// DCONV SUB-DEF (from compose_spectral_dconv.rs)
// =========================================================================

mod dconv {
    use super::*;

    // Parameters
    const CHANNELS: usize = 8;
    const T_LEN: usize = 4;
    const DCONV_COMPRESS: usize = 4;
    const DCONV_KERNEL: usize = 3;
    const DCONV_DEPTH: usize = 2;

    struct DConvInputs {
        conv_compress_weight: TensorNodeId,
        conv_compress_bias: TensorNodeId,
        norm_compress_gamma: TensorNodeId,
        norm_compress_beta: TensorNodeId,
        conv_expand_weight: TensorNodeId,
        conv_expand_bias: TensorNodeId,
        norm_expand_gamma: TensorNodeId,
        norm_expand_beta: TensorNodeId,
        layer_scale: TensorNodeId,
        eps1: TensorNodeId,
        eps2: TensorNodeId,
        dilation: usize,
    }

    impl DConvInputs {
        fn add_to_builder(
            b: &mut TensorBlockBuilder,
            k: usize,
            channels: usize,
            compressed: usize,
        ) -> Self {
            let doubled = channels * 2;
            Self {
                conv_compress_weight: b
                    .add_input(&format!("dc{k}_cw"), &[compressed, channels, DCONV_KERNEL]),
                conv_compress_bias: b.add_input(&format!("dc{k}_cb"), &[compressed]),
                norm_compress_gamma: b.add_input(&format!("dc{k}_ng"), &[compressed]),
                norm_compress_beta: b.add_input(&format!("dc{k}_nb"), &[compressed]),
                conv_expand_weight: b.add_input(&format!("dc{k}_ew"), &[doubled, compressed, 1]),
                conv_expand_bias: b.add_input(&format!("dc{k}_eb"), &[doubled]),
                norm_expand_gamma: b.add_input(&format!("dc{k}_eng"), &[doubled]),
                norm_expand_beta: b.add_input(&format!("dc{k}_enb"), &[doubled]),
                layer_scale: b.add_input(&format!("dc{k}_ls"), &[channels]),
                eps1: b.add_input(&format!("dc{k}_eps"), &[1]),
                eps2: b.add_input(&format!("dc{k}_eps2"), &[1]),
                dilation: 1 << k,
            }
        }
    }

    fn build_dconv_sublayer(
        b: &mut TensorBlockBuilder,
        input: TensorNodeId,
        dc: &DConvInputs,
        channels: usize,
        compressed: usize,
        t_len: usize,
    ) -> TensorNodeId {
        let doubled = channels * 2;
        let dc_padding = dc.dilation * (DCONV_KERNEL - 1) / 2;

        let c1 = b.add_conv1d_full(
            input,
            dc.conv_compress_weight,
            Some(dc.conv_compress_bias),
            1,
            dc_padding,
            dc.dilation,
            1,
            &[compressed, t_len],
        );
        let n1 = b.add_group_norm_g1(
            c1,
            dc.eps1,
            Some(dc.norm_compress_gamma),
            Some(dc.norm_compress_beta),
            compressed,
            t_len,
        );
        let g1 = b.add_gelu(n1, &[compressed, t_len]);
        let c2 = b.add_conv1d(
            g1,
            dc.conv_expand_weight,
            Some(dc.conv_expand_bias),
            1,
            0,
            &[doubled, t_len],
        );
        let n2 = b.add_group_norm_g1(
            c2,
            dc.eps2,
            Some(dc.norm_expand_gamma),
            Some(dc.norm_expand_beta),
            doubled,
            t_len,
        );
        let glu = b.add_glu(n2, 0, &[doubled, t_len]).expect("even dim");
        let ls = b.add_layer_scale(glu, dc.layer_scale, &[channels, t_len]);
        b.add_binary_add(input, ls, &[channels, t_len])
    }

    fn build_spectral_dconv() -> nn_dsl::tensor_ir::TensorKernelDef {
        let compressed = CHANNELS / DCONV_COMPRESS;
        let mut b = TensorBlockBuilder::new("spec_dconv_verify");
        let data = b.add_input("data", &[CHANNELS, T_LEN]);

        let mut dconv_inputs = Vec::with_capacity(DCONV_DEPTH);
        for k in 0..DCONV_DEPTH {
            let di = DConvInputs::add_to_builder(&mut b, k, CHANNELS, compressed);
            dconv_inputs.push(di);
        }

        let mut x = data;
        for di in &dconv_inputs {
            x = build_dconv_sublayer(&mut b, x, di, CHANNELS, compressed, T_LEN);
        }
        b.build(x).expect("valid graph")
    }

    fn bindings() -> Vec<TensorParamBinding> {
        let compressed = CHANNELS / DCONV_COMPRESS;
        let doubled = CHANNELS * 2;
        let mut bindings = Vec::new();

        bindings.push(TensorParamBinding::Variable);

        for _k in 0..DCONV_DEPTH {
            let cw = ArrayD::from_elem(IxDyn(&[compressed, CHANNELS, DCONV_KERNEL]), 0.01f32);
            bindings.push(TensorParamBinding::ConstantTensor(cw));
            let cb = ArrayD::from_elem(IxDyn(&[compressed]), 0.0f32);
            bindings.push(TensorParamBinding::ConstantTensor(cb));
            let ng = ArrayD::from_elem(IxDyn(&[compressed]), 1.0f32);
            bindings.push(TensorParamBinding::ConstantTensor(ng));
            let nb = ArrayD::from_elem(IxDyn(&[compressed]), 0.0f32);
            bindings.push(TensorParamBinding::ConstantTensor(nb));
            let ew = ArrayD::from_elem(IxDyn(&[doubled, compressed, 1]), 0.01f32);
            bindings.push(TensorParamBinding::ConstantTensor(ew));
            let eb = ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32);
            bindings.push(TensorParamBinding::ConstantTensor(eb));
            let eng = ArrayD::from_elem(IxDyn(&[doubled]), 1.0f32);
            bindings.push(TensorParamBinding::ConstantTensor(eng));
            let enb = ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32);
            bindings.push(TensorParamBinding::ConstantTensor(enb));
            let ls = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.1f32);
            bindings.push(TensorParamBinding::ConstantTensor(ls));
            bindings.push(TensorParamBinding::ConstantScalar(1e-5));
            bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        }
        bindings
    }

    #[test]
    fn test_spectral_dconv_def_validates() {
        let def = build_spectral_dconv();
        def.validate().expect("spectral DConv def should validate");
    }

    #[test]
    fn test_spectral_dconv_graph_builds() {
        let def = build_spectral_dconv();
        let b = bindings();
        let graph =
            tensor_kernel_to_graph(&def, &b).expect("spectral DConv graph should translate");
        assert!(
            graph.num_nodes() > 0,
            "spectral DConv graph should be non-empty, got {}",
            graph.num_nodes()
        );
    }

    #[test]
    fn test_spectral_dconv_ibp_propagates() {
        let def = build_spectral_dconv();
        let b = bindings();
        let graph = tensor_kernel_to_graph(&def, &b).expect("graph translation");
        let input = uniform_bounds(&[CHANNELS, T_LEN], 1.0);

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through spectral DConv");
        assert_eq!(
            output.lower_upper().0.shape(),
            &[CHANNELS, T_LEN],
            "output shape mismatch"
        );
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!(
            "Spectral DConv IBP bounds: [{lo_min}, {hi_max}] over {CHANNELS}×{T_LEN} output"
        );
    }

    #[test]
    fn test_spectral_dconv_crown_propagation() {
        let def = build_spectral_dconv();
        let b = bindings();
        let graph = tensor_kernel_to_graph(&def, &b).expect("graph translation");
        let input = uniform_bounds(&[CHANNELS, T_LEN], 1.0);

        let (method, output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);
        assert_eq!(
            output.lower_upper().0.shape(),
            &[CHANNELS, T_LEN],
            "output shape mismatch"
        );

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("Spectral DConv: method={method:?}, bounds=[{lo_min}, {hi_max}]");
        if let Some(reason) = &fallback_reason {
            eprintln!("CROWN fallback reason: {reason}");
        }
        assert!(
            lo_min.is_finite(),
            "lower bound should be finite, got {lo_min}"
        );
        assert!(
            hi_max.is_finite(),
            "upper bound should be finite, got {hi_max}"
        );
    }
}

// =========================================================================
// CONV_TRANSPOSE SUB-DEF (from compose_spectral_conv_tr.rs)
// =========================================================================

mod conv_tr {
    use super::*;

    // Parameters
    const CHANNELS: usize = 8;
    const F_IN: usize = 4;
    const SPECTRAL_STRIDE: usize = 4;
    const CT_KERNEL: usize = 8;
    const CT_PADDING: usize = CT_KERNEL / 4;
    const OUT_CHANNELS: usize = 4;

    fn build_spectral_conv_tr() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
        let ct_f_out = conv_transpose_out_len(F_IN, SPECTRAL_STRIDE, CT_KERNEL, CT_PADDING);
        let target_f = ct_f_out - 2; // Simulate trim

        let mut b = TensorBlockBuilder::new("spec_conv_tr_verify");
        let data = b.add_input("data", &[CHANNELS, F_IN]);
        let ct_weight = b.add_input("ct_weight", &[CHANNELS, OUT_CHANNELS, CT_KERNEL]);
        let ct_bias = b.add_input("ct_bias", &[OUT_CHANNELS]);

        let ct_out = b.add_conv_transpose_1d(
            data,
            ct_weight,
            Some(ct_bias),
            SPECTRAL_STRIDE,
            CT_PADDING,
            1, // dilation
            1, // groups
            0, // output_padding
            &[OUT_CHANNELS, ct_f_out],
        );

        let trimmed = if ct_f_out > target_f {
            b.add_narrow(ct_out, 1, 0, target_f, &[OUT_CHANNELS, target_f])
        } else {
            ct_out
        };

        (b.build(trimmed).expect("valid graph"), target_f)
    }

    fn bindings() -> Vec<TensorParamBinding> {
        let mut bindings = Vec::new();
        bindings.push(TensorParamBinding::Variable);
        let ct_w = ArrayD::from_elem(IxDyn(&[CHANNELS, OUT_CHANNELS, CT_KERNEL]), 0.02f32);
        bindings.push(TensorParamBinding::ConstantTensor(ct_w));
        let ct_b = ArrayD::from_elem(IxDyn(&[OUT_CHANNELS]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(ct_b));
        bindings
    }

    fn input_bounds() -> BoundedTensor {
        let lower = ArrayD::from_elem(IxDyn(&[CHANNELS, F_IN]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[CHANNELS, F_IN]), 1.0f32);
        BoundedTensor::new(lower, upper).expect("valid ConvTranspose input bounds")
    }

    #[test]
    fn test_spectral_conv_tr_def_validates() {
        let (def, _) = build_spectral_conv_tr();
        def.validate()
            .expect("spectral ConvTranspose def should validate");
    }

    #[test]
    fn test_spectral_conv_tr_graph_builds() {
        let (def, _) = build_spectral_conv_tr();
        let b = bindings();
        let graph = tensor_kernel_to_graph(&def, &b)
            .expect("spectral ConvTranspose graph should translate");
        assert!(
            graph.num_nodes() >= 2,
            "spectral ConvTranspose graph should have >= 2 nodes, got {}",
            graph.num_nodes()
        );
    }

    #[test]
    fn test_spectral_conv_tr_ibp_propagates() {
        let (def, target_f) = build_spectral_conv_tr();
        let b = bindings();
        let graph = tensor_kernel_to_graph(&def, &b).expect("graph translation");
        let input = input_bounds();

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through spectral ConvTranspose");
        let (lo, _) = output.lower_upper();
        assert_eq!(
            lo.shape(),
            &[OUT_CHANNELS, target_f],
            "output shape mismatch"
        );
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!(
            "Spectral ConvTranspose IBP bounds: [{lo_min}, {hi_max}] over {OUT_CHANNELS}×{target_f} output"
        );
    }

    #[test]
    fn test_spectral_conv_tr_crown_propagation() {
        let (def, target_f) = build_spectral_conv_tr();
        let b = bindings();
        let graph = tensor_kernel_to_graph(&def, &b).expect("graph translation");
        let input = input_bounds();

        let (method, output, fallback_reason) =
            propagate_with_crown_fallback(&graph, &input).expect("propagation");
        let (lo, _) = output.lower_upper();
        assert_eq!(
            lo.shape(),
            &[OUT_CHANNELS, target_f],
            "output shape mismatch"
        );
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("Spectral ConvTranspose: method={method:?}, bounds=[{lo_min}, {hi_max}]");
        if let Some(reason) = &fallback_reason {
            eprintln!("CROWN fallback reason: {reason}");
        }
    }

    #[test]
    fn test_spectral_conv_tr_crown_at_least_as_tight() {
        let (def, _) = build_spectral_conv_tr();
        let b = bindings();
        let graph = tensor_kernel_to_graph(&def, &b).expect("graph translation");
        let input = input_bounds();

        let ibp_output = graph
            .propagate_ibp(&input)
            .expect("IBP through spectral ConvTranspose");
        let (_, crown_output, _) =
            propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");

        assert_crown_tighter_than_ibp(&crown_output, &ibp_output);

        let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
        let (crown_lo, crown_hi) = crown_output.lower_upper();
        let ibp_range = ibp_hi
            .iter()
            .zip(ibp_lo.iter())
            .map(|(h, l)| h - l)
            .fold(0.0f32, f32::max);
        let crown_range = crown_hi
            .iter()
            .zip(crown_lo.iter())
            .map(|(h, l)| h - l)
            .fold(0.0f32, f32::max);
        eprintln!("ConvTranspose max range: IBP={ibp_range:.4}, CROWN={crown_range:.4}");
    }
}
