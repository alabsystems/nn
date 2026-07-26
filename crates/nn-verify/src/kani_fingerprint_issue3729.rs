// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Issue-specific Kani harnesses for `fingerprint.rs` (#3729).

#[cfg(kani)]
mod proofs {
    use kani::assume;
    use nn_core::dyn_tensor::trace::{TraceNode, TraceOp, WeightRef};
    use nn_core::DType;

    use crate::fingerprint::{fingerprint_trace, fingerprint_trace_with_weights};

    fn relu_node(shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            1,
            "relu_0".to_string(),
            TraceOp::Relu,
            vec![0],
            shape,
            DType::F32,
        )
    }

    fn softmax_node(dim: usize, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            1,
            "softmax_0".to_string(),
            TraceOp::Softmax { dim },
            vec![0],
            shape,
            DType::F32,
        )
    }

    fn linear_node(weight: WeightRef) -> TraceNode {
        TraceNode::new(
            1,
            "linear_0".to_string(),
            TraceOp::Linear { weight, bias: None },
            vec![0],
            vec![1, 2],
            DType::F32,
        )
    }

    #[kani::unwind(128)]
    #[kani::proof]
    fn hash_consistency_for_repeated_structural_fingerprint() {
        let dim: u8 = kani::any();
        assume(dim <= 1);

        let node = softmax_node(dim as usize, vec![2, 4]);
        let left = fingerprint_trace(&[node.clone()]);
        let right = fingerprint_trace(&[node]);

        assert_eq!(left.len(), 1);
        assert_eq!(left[0].hash, right[0].hash);
        assert_eq!(left[0].op_summary, right[0].op_summary);
    }

    #[kani::unwind(128)]
    #[kani::proof]
    fn bounded_structural_differences_do_not_collide() {
        let lhs_is_relu: bool = kani::any();
        let rhs_is_relu: bool = kani::any();
        let lhs_wide: bool = kani::any();
        let rhs_wide: bool = kani::any();

        let lhs_shape = if lhs_wide { vec![2, 4] } else { vec![2, 2] };
        let rhs_shape = if rhs_wide { vec![2, 4] } else { vec![2, 2] };
        assume(lhs_is_relu != rhs_is_relu || lhs_shape != rhs_shape);

        let lhs = if lhs_is_relu {
            relu_node(lhs_shape)
        } else {
            softmax_node(1, lhs_shape)
        };
        let rhs = if rhs_is_relu {
            relu_node(rhs_shape)
        } else {
            softmax_node(1, rhs_shape)
        };

        let left = fingerprint_trace(&[lhs]);
        let right = fingerprint_trace(&[rhs]);

        assert_ne!(left[0].hash, right[0].hash);
    }

    #[kani::unwind(128)]
    #[kani::proof]
    fn parametric_fingerprint_detects_weight_updates() {
        let changed_index: u8 = kani::any();
        assume(changed_index < 2);

        let base = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut updated = base.clone();
        updated[changed_index as usize] += 1.0;

        let lhs = linear_node(WeightRef::new(base, vec![2, 2]).expect("valid weight"));
        let rhs = linear_node(WeightRef::new(updated, vec![2, 2]).expect("valid weight"));

        let structural_left = fingerprint_trace(&[lhs.clone()]);
        let structural_right = fingerprint_trace(&[rhs.clone()]);
        let parametric_left = fingerprint_trace_with_weights(&[lhs]);
        let parametric_right = fingerprint_trace_with_weights(&[rhs]);

        assert_eq!(structural_left[0].hash, structural_right[0].hash);
        assert_ne!(parametric_left[0].hash, parametric_right[0].hash);
    }
}
