// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use nn_dsl::trace_compile::{CompiledStep, NativeOpKind};

use super::CompiledModel;

#[test]
fn test_weight_buffer_aliases_excludes_constant_weight_steps() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");

    let mut model = CompiledModel::empty();
    {
        let def = std::sync::Arc::get_mut(&mut model.def)
            .expect("refcount must be 1 for freshly created model");
        def.steps = vec![
            CompiledStep::NativeOp {
                op: NativeOpKind::NormActivConv1d {
                    activation: nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
                    eps: 1e-5,
                    conv_dilation: 1,
                    conv_padding: 0,
                    input_shape: vec![1, 2, 4],
                    output_channels: 2,
                    kernel_size: 2,
                    external_node_ids: None,
                },
                weight_data: HashMap::new(),
            },
            CompiledStep::NativeOp {
                op: NativeOpKind::ConstantWeight {
                    name: "constant_weight".into(),
                    shape: vec![5],
                },
                weight_data: HashMap::new(),
            },
        ];
        def.weight_buffers = vec![
            HashMap::from([(
                "conv_weight".to_string(),
                ctx.create_buffer(&[1.0_f32, 2.0, 3.0, 4.0])
                    .expect("conv_weight buffer"),
            )]),
            HashMap::from([(
                "constant_weight".to_string(),
                ctx.create_buffer(&[9.0_f32; 5])
                    .expect("constant_weight buffer"),
            )]),
        ];
    }

    let aliases = model.weight_buffer_aliases();

    let invariant_key = (0, "conv_weight".to_string());
    let constant_key = (1, "constant_weight".to_string());
    assert!(
        aliases.contains_key(&invariant_key),
        "invariant model weights should still be shared"
    );
    assert!(
        !aliases.contains_key(&constant_key),
        "shape-dependent ConstantWeight buffers must stay out of the shared store"
    );
}
