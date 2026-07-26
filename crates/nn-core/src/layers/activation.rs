// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Activation function enum implementing [`Module`].

use super::Module;
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::DynTensor;
use crate::error::Result;

/// Activation function as a module.
///
/// Wraps element-wise activation functions so they can be used anywhere
/// a `Module` is expected (e.g., in Sequential containers).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Activation {
    Relu,
    Gelu,
    Silu,
    Sigmoid,
    Tanh,
    /// Exponential Linear Unit with alpha parameter.
    Elu(f64),
    /// Leaky ReLU with negative slope parameter.
    LeakyRelu(f64),
}

impl Module for Activation {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        super::traced_forward(
            &[x],
            || {
                Ok(match self {
                    Self::Relu => TraceOp::Relu,
                    Self::Gelu => TraceOp::Gelu,
                    Self::Silu => TraceOp::Silu,
                    Self::Sigmoid => TraceOp::Sigmoid,
                    Self::Tanh => TraceOp::Tanh,
                    Self::Elu(alpha) => TraceOp::Elu { alpha: *alpha },
                    Self::LeakyRelu(slope) => TraceOp::LeakyRelu { slope: *slope },
                })
            },
            || match self {
                Self::Relu => x.relu(),
                Self::Gelu => x.gelu(),
                Self::Silu => x.silu(),
                Self::Sigmoid => x.sigmoid(),
                Self::Tanh => x.tanh(),
                Self::Elu(alpha) => x.elu(*alpha),
                Self::LeakyRelu(slope) => x.leaky_relu(*slope),
            },
        )
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_activation_relu() {
        let input = DynTensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4], &Device::Cpu).unwrap();
        let output = Activation::Relu.forward(&input).unwrap();
        assert_eq!(
            output.to_flat_vec::<f32>().unwrap(),
            vec![0.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn test_activation_sigmoid() {
        let input = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
        let output = Activation::Sigmoid.forward(&input).unwrap();
        let val = output.to_scalar::<f32>().unwrap();
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_activation_tanh() {
        let input = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
        let output = Activation::Tanh.forward(&input).unwrap();
        let val = output.to_scalar::<f32>().unwrap();
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn test_activation_elu() {
        let input = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
        let output = Activation::Elu(1.0).forward(&input).unwrap();
        let flat = output.to_flat_vec::<f32>().unwrap();
        // ELU(-1) = alpha * (exp(-1) - 1) ≈ -0.6321
        assert!((flat[0] - (-0.6321)).abs() < 0.001);
        assert_eq!(flat[1], 0.0);
        assert_eq!(flat[2], 1.0);
    }

    #[test]
    fn test_activation_leaky_relu() {
        let input =
            DynTensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &Device::Cpu).unwrap();
        let output = Activation::LeakyRelu(0.1).forward(&input).unwrap();
        let flat = output.to_flat_vec::<f32>().unwrap();
        // Negative values scaled by 0.1
        assert!((flat[0] - (-0.2)).abs() < 1e-6);
        assert!((flat[1] - (-0.1)).abs() < 1e-6);
        assert_eq!(flat[2], 0.0);
        assert_eq!(flat[3], 1.0);
        assert_eq!(flat[4], 2.0);
    }

    #[test]
    fn test_leaky_relu_method() {
        let input = DynTensor::from_vec(vec![-4.0, 0.0, 4.0], &[3], &Device::Cpu).unwrap();
        let output = input.leaky_relu(0.01).unwrap();
        let flat = output.to_flat_vec::<f32>().unwrap();
        assert!((flat[0] - (-0.04)).abs() < 1e-6);
        assert_eq!(flat[1], 0.0);
        assert_eq!(flat[2], 4.0);
    }

    #[test]
    fn test_activation_gelu() {
        let input = DynTensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4], &Device::Cpu).unwrap();
        let output = Activation::Gelu.forward(&input).unwrap();
        let flat = output.to_flat_vec::<f32>().unwrap();
        // GELU(0) = 0
        assert!(flat[1].abs() < 1e-6);
        // GELU(1) ≈ 0.8412
        assert!((flat[2] - 0.8412).abs() < 0.01);
        // GELU(2) ≈ 1.9545
        assert!((flat[3] - 1.9545).abs() < 0.01);
        // GELU(-1) ≈ -0.1588
        assert!((flat[0] - (-0.1588)).abs() < 0.01);
    }

    #[test]
    fn test_activation_silu() {
        let input = DynTensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4], &Device::Cpu).unwrap();
        let output = Activation::Silu.forward(&input).unwrap();
        let flat = output.to_flat_vec::<f32>().unwrap();
        // SiLU(x) = x * sigmoid(x)
        // SiLU(0) = 0
        assert!(flat[1].abs() < 1e-6);
        // SiLU(1) = sigmoid(1) ≈ 0.7311
        assert!((flat[2] - 0.7311).abs() < 0.01);
        // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.2689
        assert!((flat[0] - (-0.2689)).abs() < 0.01);
        // SiLU(2) = 2 * sigmoid(2) ≈ 1.7616
        assert!((flat[3] - 1.7616).abs() < 0.01);
    }

    /// Verify that Elu trace records the actual alpha parameter (#2246).
    #[test]
    fn test_activation_elu_trace_preserves_alpha() {
        use crate::dyn_tensor::trace::{record_input, trace_graph, TraceOp};
        use crate::DType;

        let input = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
        let alpha = 0.5;
        let (_, graph) = trace_graph(|| {
            let mut x = input.clone();
            let id = record_input(&[3], DType::F32).unwrap();
            x.set_trace_id(id);
            Activation::Elu(alpha).forward(&x)
        })
        .unwrap();

        let nodes = graph.nodes();
        assert_eq!(nodes.len(), 2);
        match nodes[1].op() {
            TraceOp::Elu { alpha: a } => assert!((a - 0.5).abs() < 1e-12),
            other => panic!("expected TraceOp::Elu, got {other:?}"),
        }
    }

    /// Verify that LeakyRelu trace records the actual slope parameter (#2246).
    #[test]
    fn test_activation_leaky_relu_trace_preserves_slope() {
        use crate::dyn_tensor::trace::{record_input, trace_graph, TraceOp};
        use crate::DType;

        let input = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
        let slope = 0.2;
        let (_, graph) = trace_graph(|| {
            let mut x = input.clone();
            let id = record_input(&[3], DType::F32).unwrap();
            x.set_trace_id(id);
            Activation::LeakyRelu(slope).forward(&x)
        })
        .unwrap();

        let nodes = graph.nodes();
        assert_eq!(nodes.len(), 2);
        match nodes[1].op() {
            TraceOp::LeakyRelu { slope: s } => assert!((s - 0.2).abs() < 1e-12),
            other => panic!("expected TraceOp::LeakyRelu, got {other:?}"),
        }
    }
}
