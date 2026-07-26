// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs decoder block and encoder→decoder composition.
//!
//! Two test modules:
//!
//! - `decoder_block`: Standalone decoder block (skip_add → Rewrite → GLU →
//!   DConv(×2) → ConvTranspose1d → center_trim → GELU). 5 tests.
//!
//! - `encoder_decoder`: Full encoder→decoder pipeline via shared helper module
//!   (Conv1d → GELU → DConv → Conv1d(k=1) → GLU → skip_add → Rewrite → GLU →
//!   DConv → ConvTranspose1d → GELU). 6 tests.
//!
//! Part of #779 / #1982 — composition verification consolidation.

use super::common;

#[path = "demucs_enc_dec_helpers_dconv.rs"]
#[allow(dead_code)]
mod demucs_enc_dec_helpers_dconv;

// ===== Decoder block tests (standalone) =====================================
mod decoder_block {
    use super::common::{
        assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max,
        conv1d_out_len, conv_transpose_out_len, uniform_bounds, verify_and_assert,
    };
    use nn_dsl::tensor_block_builder::TensorBlockBuilder;
    use nn_dsl::tensor_ir::TensorNodeId;
    use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
    use ndarray::{ArrayD, IxDyn};

    // -----------------------------------------------------------------------
    // Small-scale decoder block parameters
    // -----------------------------------------------------------------------

    const CHANNELS: usize = 8;
    const T_IN: usize = 4;
    const DCONV_COMPRESS_RATIO: usize = 4;
    const DCONV_KERNEL: usize = 3;
    const REWRITE_KERNEL: usize = 3;
    const REWRITE_PADDING: usize = REWRITE_KERNEL / 2;
    const CT_KERNEL: usize = 4;
    const CT_STRIDE: usize = 2;
    const CT_PADDING: usize = 1;
    const DCONV_DEPTH: usize = 2;
    const OUT_CHANNELS: usize = 4;

    // -----------------------------------------------------------------------
    // Topology builder helpers
    // -----------------------------------------------------------------------

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

    fn build_decoder_block() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
        let compressed = CHANNELS / DCONV_COMPRESS_RATIO;
        let doubled = CHANNELS * 2;

        let mut b = TensorBlockBuilder::new("demucs_dec_block_verify");

        let data = b.add_input("data", &[CHANNELS, T_IN]);
        let skip = b.add_input("skip", &[CHANNELS, T_IN]);
        let rw_weight = b.add_input("rw_weight", &[doubled, CHANNELS, REWRITE_KERNEL]);
        let rw_bias = b.add_input("rw_bias", &[doubled]);

        let mut dconv_inputs = Vec::with_capacity(DCONV_DEPTH);
        for k in 0..DCONV_DEPTH {
            dconv_inputs.push(DConvInputs::add_to_builder(&mut b, k, CHANNELS, compressed));
        }

        let ct_weight = b.add_input("ct_weight", &[CHANNELS, OUT_CHANNELS, CT_KERNEL]);
        let ct_bias = b.add_input("ct_bias", &[OUT_CHANNELS]);

        // Skip add → Rewrite Conv1d → GLU
        let x = b.add_binary_add(data, skip, &[CHANNELS, T_IN]);
        let rw_t_out = conv1d_out_len(T_IN, REWRITE_KERNEL, 1, REWRITE_PADDING);
        let rw_out = b.add_conv1d(
            x,
            rw_weight,
            Some(rw_bias),
            1,
            REWRITE_PADDING,
            &[doubled, rw_t_out],
        );
        let glu_out = b
            .add_glu(rw_out, 0, &[doubled, rw_t_out])
            .expect("even dim");

        // DConv (2 residual sub-layers)
        let mut dconv_out = glu_out;
        for di in &dconv_inputs {
            dconv_out = build_dconv_sublayer(&mut b, dconv_out, di, CHANNELS, compressed, rw_t_out);
        }

        // ConvTranspose1d → Narrow (trim) → GELU
        let ct_t_out = conv_transpose_out_len(rw_t_out, CT_STRIDE, CT_KERNEL, CT_PADDING);
        let ct_out = b.add_conv_transpose_1d(
            dconv_out,
            ct_weight,
            Some(ct_bias),
            CT_STRIDE,
            CT_PADDING,
            1, // dilation
            1, // groups
            0, // output_padding
            &[OUT_CHANNELS, ct_t_out],
        );
        let target_len = ct_t_out.min(T_IN * CT_STRIDE);
        let output = if ct_t_out > target_len {
            b.add_narrow(ct_out, 1, 0, target_len, &[OUT_CHANNELS, target_len])
        } else {
            ct_out
        };
        let final_out = b.add_gelu(output, &[OUT_CHANNELS, target_len]);

        (b.build(final_out).expect("valid graph"), target_len)
    }

    fn decoder_block_bindings() -> Vec<TensorParamBinding> {
        let mut bindings = Vec::new();
        let compressed = CHANNELS / DCONV_COMPRESS_RATIO;
        let doubled = CHANNELS * 2;

        bindings.push(TensorParamBinding::Variable);
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CHANNELS, T_IN]),
            0.0f32,
        )));

        // Rewrite Conv1d
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[doubled, CHANNELS, REWRITE_KERNEL]),
            0.01f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[doubled]),
            0.0f32,
        )));

        // DConv sub-layers
        for _k in 0..DCONV_DEPTH {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[compressed, CHANNELS, DCONV_KERNEL]),
                0.01f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[compressed]),
                0.0f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[compressed]),
                1.0f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[compressed]),
                0.0f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[doubled, compressed, 1]),
                0.01f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[doubled]),
                0.0f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[doubled]),
                1.0f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[doubled]),
                0.0f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[CHANNELS]),
                0.1f32,
            )));
            bindings.push(TensorParamBinding::ConstantScalar(1e-5));
            bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        }

        // ConvTranspose1d
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CHANNELS, OUT_CHANNELS, CT_KERNEL]),
            0.01f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUT_CHANNELS]),
            0.0f32,
        )));

        bindings
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decoder_block_def_validates() {
        let (def, _) = build_decoder_block();
        def.validate().expect("decoder block def should validate");
    }

    #[test]
    fn test_decoder_block_graph_builds() {
        let (def, _) = build_decoder_block();
        let bindings = decoder_block_bindings();
        let graph =
            tensor_kernel_to_graph(&def, &bindings).expect("decoder block graph should translate");
        assert!(
            graph.num_nodes() >= 15,
            "decoder block graph should have >= 15 nodes, got {}",
            graph.num_nodes()
        );
    }

    #[test]
    fn test_decoder_block_ibp_propagates() {
        let (def, target_len) = build_decoder_block();
        let bindings = decoder_block_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[CHANNELS, T_IN], 1.0);

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through decoder block");

        let expected_shape = vec![OUT_CHANNELS, target_len];
        assert_eq!(
            output.lower_upper().0.shape(),
            expected_shape.as_slice(),
            "output shape mismatch"
        );
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!(
            "Decoder block IBP bounds range: [{lo_min}, {hi_max}] over {} elements",
            OUT_CHANNELS * target_len
        );
    }

    #[test]
    fn test_decoder_block_crown_propagation() {
        let (def, target_len) = build_decoder_block();
        let bindings = decoder_block_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[CHANNELS, T_IN], 1.0);

        let (method, output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);
        let (lo, _) = output.lower_upper();

        let expected_shape = vec![OUT_CHANNELS, target_len];
        assert_eq!(
            lo.shape(),
            expected_shape.as_slice(),
            "output shape mismatch"
        );

        eprintln!("Demucs decoder block: method={method:?}");
        if let Some(reason) = &fallback_reason {
            eprintln!("CROWN fallback reason: {reason}");
        }
    }

    #[test]
    fn test_decoder_block_verify_and_record() {
        let (def, target_len) = build_decoder_block();
        let bindings = decoder_block_bindings();
        let input = uniform_bounds(&[CHANNELS, T_IN], 1.0);

        let result = verify_and_assert(&def, &bindings, &input, "demucs_temporal_decoder_block");
        assert_eq!(result.num_variables, 1, "single Variable input (data)");

        let (lo, _) = result.output_bounds.lower_upper();
        assert_eq!(lo.shape(), &[OUT_CHANNELS, target_len]);
    }
}

// ===== Encoder → decoder composition tests ==================================
mod encoder_decoder {
    use super::common::{
        assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max,
    };

    mod helpers {
        #![allow(dead_code)]

        pub(super) use super::super::demucs_enc_dec_helpers_dconv::{
            build_dconv_sublayer, push_dconv_bindings, DConvInputs,
        };
        use nn_dsl::tensor_block_builder::TensorBlockBuilder;
        use nn_dsl::tensor_ir::TensorNodeId;
        use nn_verify::{BoundedTensor, TensorParamBinding};
        use ndarray::{ArrayD, IxDyn};

        // Constants
        pub(super) const ENC_IN_CH: usize = 8;
        pub(super) const BOTTLENECK_CH: usize = 16;
        pub(super) const DEC_OUT_CH: usize = 8;
        pub(super) const T_IN: usize = 16;
        const ENC_CONV_K: usize = 8;
        const ENC_CONV_S: usize = 4;
        const ENC_CONV_P: usize = ENC_CONV_K / 4;
        const DCONV_DEPTH: usize = 1;
        const DEC_RW_K: usize = 3;
        const DEC_RW_P: usize = DEC_RW_K / 2;
        const CT_K: usize = 4;
        const CT_S: usize = 2;
        const CT_P: usize = 1;
        const COMPRESS_RATIO: usize = 4;

        use super::super::common::{conv1d_out_len, conv_transpose_out_len};

        pub(super) fn bottleneck_t() -> usize {
            conv1d_out_len(T_IN, ENC_CONV_K, ENC_CONV_S, ENC_CONV_P)
        }

        pub(super) fn dec_rw_t() -> usize {
            conv1d_out_len(bottleneck_t(), DEC_RW_K, 1, DEC_RW_P)
        }

        pub(super) fn output_t() -> usize {
            conv_transpose_out_len(dec_rw_t(), CT_S, CT_K, CT_P)
        }

        struct EncoderNodes {
            data: TensorNodeId,
            conv_w: TensorNodeId,
            conv_b: TensorNodeId,
            dconv: Vec<DConvInputs>,
            rw_w: TensorNodeId,
            rw_b: TensorNodeId,
        }

        struct DecoderNodes {
            skip: TensorNodeId,
            rw_w: TensorNodeId,
            rw_b: TensorNodeId,
            dconv: Vec<DConvInputs>,
            ct_w: TensorNodeId,
            ct_b: TensorNodeId,
        }

        fn add_encoder_inputs(b: &mut TensorBlockBuilder) -> EncoderNodes {
            let compressed = BOTTLENECK_CH / COMPRESS_RATIO;
            let doubled = BOTTLENECK_CH * 2;
            let data = b.add_input("data", &[ENC_IN_CH, T_IN]);
            let conv_w = b.add_input("enc_conv_w", &[BOTTLENECK_CH, ENC_IN_CH, ENC_CONV_K]);
            let conv_b = b.add_input("enc_conv_b", &[BOTTLENECK_CH]);
            let mut dconv = Vec::with_capacity(DCONV_DEPTH);
            for k in 0..DCONV_DEPTH {
                dconv.push(DConvInputs::add_to_builder(
                    b,
                    "enc",
                    k,
                    BOTTLENECK_CH,
                    compressed,
                ));
            }
            let rw_w = b.add_input("enc_rw_w", &[doubled, BOTTLENECK_CH, 1]);
            let rw_b = b.add_input("enc_rw_b", &[doubled]);
            EncoderNodes {
                data,
                conv_w,
                conv_b,
                dconv,
                rw_w,
                rw_b,
            }
        }

        fn add_decoder_inputs(b: &mut TensorBlockBuilder, t_mid: usize) -> DecoderNodes {
            let compressed = BOTTLENECK_CH / COMPRESS_RATIO;
            let doubled = BOTTLENECK_CH * 2;
            let skip = b.add_input("dec_skip", &[BOTTLENECK_CH, t_mid]);
            let rw_w = b.add_input("dec_rw_w", &[doubled, BOTTLENECK_CH, DEC_RW_K]);
            let rw_b = b.add_input("dec_rw_b", &[doubled]);
            let mut dconv = Vec::with_capacity(DCONV_DEPTH);
            for k in 0..DCONV_DEPTH {
                dconv.push(DConvInputs::add_to_builder(
                    b,
                    "dec",
                    k,
                    BOTTLENECK_CH,
                    compressed,
                ));
            }
            let ct_w = b.add_input("dec_ct_w", &[BOTTLENECK_CH, DEC_OUT_CH, CT_K]);
            let ct_b = b.add_input("dec_ct_b", &[DEC_OUT_CH]);
            DecoderNodes {
                skip,
                rw_w,
                rw_b,
                dconv,
                ct_w,
                ct_b,
            }
        }

        fn wire_encoder(b: &mut TensorBlockBuilder, enc: &EncoderNodes) -> TensorNodeId {
            let compressed = BOTTLENECK_CH / COMPRESS_RATIO;
            let doubled = BOTTLENECK_CH * 2;
            let t_mid = bottleneck_t();
            let conv = b.add_conv1d(
                enc.data,
                enc.conv_w,
                Some(enc.conv_b),
                ENC_CONV_S,
                ENC_CONV_P,
                &[BOTTLENECK_CH, t_mid],
            );
            let gelu = b.add_gelu(conv, &[BOTTLENECK_CH, t_mid]);
            let mut x = gelu;
            for di in &enc.dconv {
                x = build_dconv_sublayer(b, x, di, BOTTLENECK_CH, compressed, t_mid);
            }
            let rw = b.add_conv1d(x, enc.rw_w, Some(enc.rw_b), 1, 0, &[doubled, t_mid]);
            b.add_glu(rw, 0, &[doubled, t_mid])
                .expect("even dim for encoder GLU")
        }

        fn wire_decoder(
            b: &mut TensorBlockBuilder,
            dec: &DecoderNodes,
            enc_out: TensorNodeId,
        ) -> TensorNodeId {
            let compressed = BOTTLENECK_CH / COMPRESS_RATIO;
            let doubled = BOTTLENECK_CH * 2;
            let t_mid = bottleneck_t();
            let rw_t = dec_rw_t();
            let ct_t = output_t();
            let x = b.add_binary_add(enc_out, dec.skip, &[BOTTLENECK_CH, t_mid]);
            let rw = b.add_conv1d(x, dec.rw_w, Some(dec.rw_b), 1, DEC_RW_P, &[doubled, rw_t]);
            let glu = b
                .add_glu(rw, 0, &[doubled, rw_t])
                .expect("even dim for decoder GLU");
            let mut dc = glu;
            for di in &dec.dconv {
                dc = build_dconv_sublayer(b, dc, di, BOTTLENECK_CH, compressed, rw_t);
            }
            let ct = b.add_conv_transpose_1d(
                dc,
                dec.ct_w,
                Some(dec.ct_b),
                CT_S,
                CT_P,
                1, // dilation
                1, // groups
                0, // output_padding
                &[DEC_OUT_CH, ct_t],
            );
            b.add_gelu(ct, &[DEC_OUT_CH, ct_t])
        }

        pub(super) fn build_encoder_decoder() -> (nn_dsl::tensor_ir::TensorKernelDef, usize, usize)
        {
            let t_mid = bottleneck_t();
            let mut b = TensorBlockBuilder::new("demucs_enc_dec_verify");
            let enc = add_encoder_inputs(&mut b);
            let dec = add_decoder_inputs(&mut b, t_mid);
            let enc_out = wire_encoder(&mut b, &enc);
            let output = wire_decoder(&mut b, &dec, enc_out);
            let ct_t = output_t();
            (
                b.build(output).expect("valid encoder-decoder graph"),
                ct_t,
                DEC_OUT_CH,
            )
        }

        pub(super) fn encoder_decoder_bindings() -> Vec<TensorParamBinding> {
            let mut bindings = Vec::new();
            let enc_compressed = BOTTLENECK_CH / COMPRESS_RATIO;
            let enc_doubled = BOTTLENECK_CH * 2;
            let t_mid = bottleneck_t();

            bindings.push(TensorParamBinding::Variable);
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[BOTTLENECK_CH, ENC_IN_CH, ENC_CONV_K]),
                0.01f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[BOTTLENECK_CH]),
                0.0f32,
            )));
            for _k in 0..DCONV_DEPTH {
                push_dconv_bindings(&mut bindings, BOTTLENECK_CH, enc_compressed);
            }
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[enc_doubled, BOTTLENECK_CH, 1]),
                0.01f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[enc_doubled]),
                0.0f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[BOTTLENECK_CH, t_mid]),
                0.0f32,
            )));
            let dec_doubled = BOTTLENECK_CH * 2;
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[dec_doubled, BOTTLENECK_CH, DEC_RW_K]),
                0.01f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[dec_doubled]),
                0.0f32,
            )));
            let dec_compressed = BOTTLENECK_CH / COMPRESS_RATIO;
            for _k in 0..DCONV_DEPTH {
                push_dconv_bindings(&mut bindings, BOTTLENECK_CH, dec_compressed);
            }
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[BOTTLENECK_CH, DEC_OUT_CH, CT_K]),
                0.01f32,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[DEC_OUT_CH]),
                0.0f32,
            )));
            bindings
        }

        pub(super) fn input_bounds() -> BoundedTensor {
            let lower = ArrayD::from_elem(IxDyn(&[ENC_IN_CH, T_IN]), -1.0f32);
            let upper = ArrayD::from_elem(IxDyn(&[ENC_IN_CH, T_IN]), 1.0f32);
            BoundedTensor::new(lower, upper).expect("valid input bounds")
        }
    }

    use helpers::{
        bottleneck_t, build_encoder_decoder, dec_rw_t, encoder_decoder_bindings, input_bounds,
        output_t, DEC_OUT_CH,
    };
    use nn_verify::{tensor_kernel_to_graph, verify_tensor_and_record, PropMethod, VerifyStatus};

    #[test]
    fn test_enc_dec_def_validates() {
        let (def, _, _) = build_encoder_decoder();
        def.validate().expect("enc-dec def should validate");
    }

    #[test]
    fn test_enc_dec_graph_builds() {
        let (def, ct_t, _) = build_encoder_decoder();
        let t_mid = bottleneck_t();

        assert_eq!(t_mid, 4, "encoder Conv1d(k=8, s=4, p=2) on T=16 → T=4");
        let rw_t = dec_rw_t();
        assert_eq!(rw_t, t_mid, "decoder rewrite Conv1d preserves T");
        assert_eq!(ct_t, output_t(), "ConvTranspose1d output T");

        let bindings = encoder_decoder_bindings();
        let graph =
            tensor_kernel_to_graph(&def, &bindings).expect("enc-dec graph should translate");
        assert!(
            graph.num_nodes() >= 30,
            "enc-dec graph should have >= 30 nodes, got {}",
            graph.num_nodes()
        );
    }

    #[test]
    fn test_enc_dec_ibp_propagates() {
        let (def, ct_t, _) = build_encoder_decoder();
        let bindings = encoder_decoder_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = input_bounds();

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through encoder-decoder");
        let (lo, _) = output.lower_upper();

        let expected_shape = [DEC_OUT_CH, ct_t];
        assert_eq!(
            lo.shape(),
            expected_shape.as_slice(),
            "output shape mismatch: expected {expected_shape:?}, got {:?}",
            lo.shape()
        );

        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!(
            "Encoder-decoder IBP bounds: [{lo_min}, {hi_max}] over {} elements",
            DEC_OUT_CH * ct_t
        );
    }

    #[test]
    fn test_enc_dec_crown_propagation() {
        let (def, ct_t, _) = build_encoder_decoder();
        let bindings = encoder_decoder_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = input_bounds();

        let (method, output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);

        let expected_shape = [DEC_OUT_CH, ct_t];
        let (lo, _) = output.lower_upper();
        assert_eq!(
            lo.shape(),
            expected_shape.as_slice(),
            "output shape mismatch"
        );

        assert_bounds_valid(&output);

        eprintln!("Encoder-decoder: method={method:?}");
        if let Some(reason) = &fallback_reason {
            eprintln!("CROWN fallback reason: {reason}");
        }

        // NY cde0ef03 (328-commit bump, #3072) now supports
        // MulConstant per-channel broadcast in CROWN propagation.
        // Previously fell back to IBP; now CROWN propagates through
        // layer_scale MulConstant nodes directly.
        assert_eq!(
            method,
            PropMethod::Crown,
            "CROWN should propagate through MulConstant per-channel broadcast (NY cde0ef03)"
        );
    }

    #[test]
    fn test_enc_dec_bounds_finite() {
        let (def, ct_t, _) = build_encoder_decoder();
        let bindings = encoder_decoder_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = input_bounds();

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through encoder-decoder");

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!(
            "Encoder-decoder bounds: [{lo_min}, {hi_max}] over {} elements",
            DEC_OUT_CH * ct_t
        );

        assert!(lo_min.is_finite(), "lower min must be finite, got {lo_min}");
        assert!(hi_max.is_finite(), "upper max must be finite, got {hi_max}");
    }

    #[test]
    fn test_enc_dec_verify_and_record() {
        let (def, ct_t, _) = build_encoder_decoder();
        let bindings = encoder_decoder_bindings();
        let input = input_bounds();

        let mut status = VerifyStatus::default();
        let result = verify_tensor_and_record(
            &mut status,
            &def,
            &bindings,
            &input,
            Some("demucs_temporal_enc_dec"),
        )
        .expect("verify_tensor_and_record pipeline");

        assert!(
            result.verification.is_finite,
            "encoder-decoder output bounds must be finite"
        );
        assert_eq!(result.num_variables, 1, "single Variable input (data)");

        let (lo, _) = result.output_bounds.lower_upper();
        let expected_shape = [DEC_OUT_CH, ct_t];
        assert_eq!(lo.shape(), expected_shape.as_slice());
        assert_bounds_valid(&result.output_bounds);

        assert!(
            status.kernel("demucs_temporal_enc_dec").is_some(),
            "status should contain 'demucs_temporal_enc_dec' entry"
        );
    }
}
