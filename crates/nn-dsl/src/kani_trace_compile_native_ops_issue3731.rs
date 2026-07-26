// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#[cfg(kani)]
mod proofs {
    use kani::assume;

    use crate::trace_compile::{
        AttentionLayout, FusedNormKind, GemmActivation, NativeOpKind, NormActivConv1dParams,
        NormActivation, StyleBatchOffset, StyleProjectionParams,
    };

    fn sample_phase(activation: NormActivation, channels: usize) -> NormActivConv1dParams {
        NormActivConv1dParams::new(activation, 1e-5, 1, 1, vec![1, channels, 4], channels, 3)
    }

    fn sample_all_native_ops() -> Vec<NativeOpKind> {
        vec![
            NativeOpKind::LstmSequence {
                hidden_size: 8,
                input_shape: vec![2, 1, 8],
                h_shape: vec![1, 8],
                reverse: false,
            },
            NativeOpKind::Cumsum {
                dim: 1,
                input_shape: vec![2, 8],
            },
            NativeOpKind::InstanceNorm {
                eps: 1e-5,
                input_shape: vec![1, 2, 4],
            },
            NativeOpKind::LayerNorm {
                eps: 1e-5,
                input_shape: vec![1, 8],
                hidden_dim: 8,
            },
            NativeOpKind::AddLayerNorm {
                eps: 1e-5,
                input_shape: vec![1, 8],
                hidden_dim: 8,
            },
            NativeOpKind::AdainSnake {
                eps: 1e-5,
                input_shape: vec![1, 4, 8],
                channels: 4,
                residual_gamma: false,
                external_node_ids: Some(vec![0, 1, 2]),
            },
            NativeOpKind::AdainLeakyRelu {
                eps: 1e-5,
                slope: 0.2,
                input_shape: vec![1, 4, 8],
                external_node_ids: Some(vec![3, 4, 5]),
            },
            NativeOpKind::AdaLayerNorm {
                eps: 1e-5,
                input_shape: vec![1, 4, 8],
                hidden_dim: 8,
            },
            NativeOpKind::FlashAttention {
                scale: 0.5,
                causal: false,
                q_shape: vec![1, 2, 4, 8],
                k_shape: vec![1, 2, 4, 8],
                output_shape: vec![1, 2, 4, 8],
                input_layout: AttentionLayout::HeadsFirst,
            },
            NativeOpKind::MaxPool1d {
                kernel_size: 2,
                stride: 2,
                padding: 0,
                input_shape: vec![1, 4, 8],
            },
            NativeOpKind::ConstantWeight {
                name: "const".into(),
                shape: vec![8],
            },
            NativeOpKind::FusedResBlock {
                phase1: sample_phase(NormActivation::Snake, 4),
                phase2: sample_phase(NormActivation::Snake, 4),
                input_steps: vec![0, 1, 2, 3, 4],
                residual_scale: 1.0,
                style_proj: None,
                shortcut_step: None,
                pool_step: None,
                style_batch_offset: None,
            },
            NativeOpKind::BatchedStyleProjection {
                blocks: vec![StyleBatchOffset::new(0, 4, 4)],
                style_dim: 8,
                total_out: 16,
                style_step: 1,
            },
            NativeOpKind::NormActivConv1d {
                activation: NormActivation::LeakyRelu { slope: 0.2 },
                eps: 1e-5,
                conv_dilation: 1,
                conv_padding: 1,
                input_shape: vec![1, 4, 8],
                output_channels: 4,
                kernel_size: 3,
                external_node_ids: Some(vec![6, 7, 8]),
            },
            NativeOpKind::LinearActivation {
                activation: GemmActivation::Gelu,
                in_features: 8,
                out_features: 8,
                has_bias: true,
                input_shape: vec![1, 8],
            },
            NativeOpKind::BatchedLinearProjection {
                in_features: 8,
                total_out_features: 24,
                projection_sizes: vec![8, 8, 8],
                has_bias: true,
                input_shape: vec![1, 4, 8],
            },
            NativeOpKind::ProjectionSlice {
                source_step: 2,
                dim: 2,
                start: 8,
                length: 8,
                output_shape: vec![1, 4, 8],
            },
            NativeOpKind::NormLinear {
                norm_kind: FusedNormKind::LayerNorm,
                eps: 1e-5,
                input_shape: vec![8, 256],
                hidden_dim: 256,
                out_features: 256,
                has_bias: true,
            },
            NativeOpKind::ChannelsFirstLayerNorm {
                eps: 1e-5,
                input_shape: vec![1, 4, 8],
                channels: 4,
                leaky_relu_slope: Some(0.2),
            },
            NativeOpKind::Int8Gemm {
                in_features: 64,
                out_features: 64,
                has_bias: true,
                input_shape: vec![1, 64],
            },
            NativeOpKind::Conv1dGemm {
                input_shape: vec![1, 32, 32],
                out_channels: 32,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
                has_bias: true,
            },
            NativeOpKind::SiluMul {
                input_shape: vec![1, 16],
            },
            NativeOpKind::RotaryEmbedding {
                head_dim: 8,
                input_shape: vec![1, 2, 4, 8],
            },
            NativeOpKind::MoeGating {
                num_experts: 8,
                top_k: 2,
                input_shape: vec![1, 16],
            },
        ]
    }

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(64)]
    fn proof_issue3731_dispatch_surface_is_complete_for_current_native_ops() {
        let ops = sample_all_native_ops();
        assert_eq!(ops.len(), 24, "keep this list in sync with NativeOpKind");

        for i in 0..ops.len() {
            let op = &ops[i];
            let name = op.variant_name();
            assert!(!name.is_empty(), "every native op needs a variant name");

            let dispatches = op.estimated_metal_dispatches();
            let encodings = op.estimated_encoding_events();
            if matches!(op, NativeOpKind::ConstantWeight { .. }) {
                assert_eq!(dispatches, 0, "constant weights must not dispatch");
            } else {
                assert!(
                    dispatches >= 1,
                    "compute native ops need a dispatch estimate"
                );
            }
            assert!(
                encodings <= dispatches + 1,
                "encoding events should stay close to dispatch count"
            );

            for j in (i + 1)..ops.len() {
                assert_ne!(
                    name,
                    ops[j].variant_name(),
                    "each NativeOpKind variant should map to a unique diagnostics name"
                );
            }
        }
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_issue3731_dispatch_boundaries_match_buffer_routing_cases() {
        let style_mode: u8 = kani::any();
        assume(style_mode <= 2);

        let resblock = NativeOpKind::FusedResBlock {
            phase1: sample_phase(NormActivation::Snake, 4),
            phase2: sample_phase(NormActivation::Snake, 4),
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: if style_mode == 1 {
                Some(StyleProjectionParams::new(4, 4, 8))
            } else {
                None
            },
            shortcut_step: Some(6),
            pool_step: Some(7),
            style_batch_offset: if style_mode == 2 {
                Some(StyleBatchOffset::new(0, 4, 4))
            } else {
                None
            },
        };
        assert_eq!(
            resblock.estimated_metal_dispatches(),
            if style_mode == 1 { 7 } else { 3 }
        );
        assert_eq!(
            resblock.estimated_encoding_events(),
            if style_mode == 1 { 6 } else { 2 }
        );

        let axis_large: bool = kani::any();
        let cumsum = NativeOpKind::Cumsum {
            dim: 1,
            input_shape: vec![1, if axis_large { 257 } else { 256 }],
        };
        assert_eq!(
            cumsum.estimated_metal_dispatches(),
            if axis_large { 3 } else { 1 }
        );

        let with_bias: bool = kani::any();
        let conv1d = NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 16, 16],
            out_channels: 16,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: with_bias,
        };
        assert_eq!(
            conv1d.estimated_metal_dispatches(),
            if with_bias { 3 } else { 2 }
        );

        let norm_linear = NativeOpKind::NormLinear {
            norm_kind: FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![8, 256],
            hidden_dim: 256,
            out_features: 256,
            has_bias: true,
        };
        assert_eq!(norm_linear.estimated_metal_dispatches(), 2);
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_issue3731_direct_buffer_counts_stay_bounded() {
        let ids_len: u8 = kani::any();
        assume((1..=3).contains(&ids_len));

        let ids: Vec<u64> = (0..usize::from(ids_len)).map(|idx| idx as u64).collect();
        let adain = NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 4, 8],
            channels: 4,
            residual_gamma: false,
            external_node_ids: Some(ids.clone()),
        };
        assert_eq!(
            adain
                .external_node_ids()
                .expect("ada-in ops carry explicit node ids"),
            ids.as_slice()
        );
        assert!(
            adain.external_node_ids().expect("ids").len() <= 3,
            "AdaIN fusion should never read more than x, gamma, beta"
        );

        let has_shortcut: bool = kani::any();
        let has_pool: bool = kani::any();
        let resblock = NativeOpKind::FusedResBlock {
            phase1: sample_phase(NormActivation::LeakyRelu { slope: 0.2 }, 4),
            phase2: sample_phase(NormActivation::LeakyRelu { slope: 0.2 }, 4),
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: has_shortcut.then_some(5),
            pool_step: has_pool.then_some(6),
            style_batch_offset: None,
        };

        let mut deps = Vec::new();
        resblock.collect_direct_step_deps(&mut deps);
        let expected = 5 + usize::from(has_shortcut) + usize::from(has_pool);
        assert_eq!(deps.len(), expected);
        assert!(
            deps.len() <= 7,
            "FusedResBlock direct buffer fan-in is capped at five inputs plus shortcut/pool"
        );
    }
}
