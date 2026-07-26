// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Arithmetic, activation, and scalar operations on [`TrackedTensor`].
//!
//! All methods follow the same pattern: call the corresponding
//! `DynTensor` method, then wrap the result in `Self::from_op` with
//! the appropriate [`Op`] variant. Extracted from `tracked.rs` for
//! 500-line compliance.

use super::TrackedTensor;
use crate::error::Result;
use crate::op::Op;
use std::sync::Arc;

impl TrackedTensor {
    // -- Binary arithmetic --

    /// Element-wise addition.
    pub fn add(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().add(rhs.tensor())?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Add(Arc::clone(self), Arc::clone(rhs)),
        )))
    }

    /// Element-wise subtraction.
    pub fn sub(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().sub(rhs.tensor())?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Sub(Arc::clone(self), Arc::clone(rhs)),
        )))
    }

    /// Element-wise multiplication.
    pub fn mul(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().mul(rhs.tensor())?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Mul(Arc::clone(self), Arc::clone(rhs)),
        )))
    }

    /// Element-wise division.
    pub fn div(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().div(rhs.tensor())?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Div(Arc::clone(self), Arc::clone(rhs)),
        )))
    }

    /// Matrix multiplication.
    pub fn matmul(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().matmul(rhs.tensor())?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::MatMul(Arc::clone(self), Arc::clone(rhs)),
        )))
    }

    // -- Unary activations --

    /// ReLU activation.
    pub fn relu(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().relu()?;
        Ok(Arc::new(Self::from_op(data, Op::Relu(Arc::clone(self)))))
    }

    /// Tanh activation.
    pub fn tanh(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().tanh()?;
        Ok(Arc::new(Self::from_op(data, Op::Tanh(Arc::clone(self)))))
    }

    /// Sigmoid activation.
    pub fn sigmoid(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().sigmoid()?;
        Ok(Arc::new(Self::from_op(data, Op::Sigmoid(Arc::clone(self)))))
    }

    /// Exponential.
    pub fn exp(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().exp()?;
        Ok(Arc::new(Self::from_op(data, Op::Exp(Arc::clone(self)))))
    }

    /// Natural logarithm.
    pub fn log(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().log()?;
        Ok(Arc::new(Self::from_op(data, Op::Log(Arc::clone(self)))))
    }

    /// Square root.
    pub fn sqrt(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().sqrt()?;
        Ok(Arc::new(Self::from_op(data, Op::Sqrt(Arc::clone(self)))))
    }

    /// Square (x^2).
    pub fn sqr(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().sqr()?;
        Ok(Arc::new(Self::from_op(data, Op::Sqr(Arc::clone(self)))))
    }

    /// Negation (-x).
    pub fn neg(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().neg()?;
        Ok(Arc::new(Self::from_op(data, Op::Neg(Arc::clone(self)))))
    }

    /// Absolute value.
    pub fn abs(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().abs()?;
        Ok(Arc::new(Self::from_op(data, Op::Abs(Arc::clone(self)))))
    }

    /// GELU activation (tanh approximation).
    pub fn gelu(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().gelu()?;
        Ok(Arc::new(Self::from_op(data, Op::Gelu(Arc::clone(self)))))
    }

    /// GELU activation (exact erf-based).
    ///
    /// Matches PyTorch's `nn.GELU(approximate='none')`. More accurate than
    /// the tanh approximation in [`gelu()`](Self::gelu).
    pub fn gelu_erf(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().gelu_erf()?;
        Ok(Arc::new(Self::from_op(data, Op::GeluErf(Arc::clone(self)))))
    }

    /// SiLU activation (x * sigmoid(x)).
    pub fn silu(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().silu()?;
        Ok(Arc::new(Self::from_op(data, Op::Silu(Arc::clone(self)))))
    }

    /// Sine.
    pub fn sin(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().sin()?;
        Ok(Arc::new(Self::from_op(data, Op::Sin(Arc::clone(self)))))
    }

    /// Cosine.
    pub fn cos(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().cos()?;
        Ok(Arc::new(Self::from_op(data, Op::Cos(Arc::clone(self)))))
    }

    /// Reciprocal (1/x).
    pub fn recip(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().recip()?;
        Ok(Arc::new(Self::from_op(data, Op::Recip(Arc::clone(self)))))
    }

    // -- Parameterized operations --

    /// Power with scalar exponent.
    pub fn powf(self: &Arc<Self>, exponent: f64) -> Result<Arc<Self>> {
        let data = self.tensor().powf(exponent)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Powf(Arc::clone(self), exponent),
        )))
    }

    /// Clamp to [min, max].
    pub fn clamp(self: &Arc<Self>, min: f64, max: f64) -> Result<Arc<Self>> {
        let data = self.tensor().clamp(min, max)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Clamp(Arc::clone(self), min, max),
        )))
    }

    /// Permute axes (multi-axis reordering).
    pub fn permute(self: &Arc<Self>, dims: &[usize]) -> Result<Arc<Self>> {
        let data = self.tensor().permute(dims)?;
        // Store inverse permutation for backward.
        let mut inv = vec![0; dims.len()];
        for (i, &d) in dims.iter().enumerate() {
            inv[d] = i;
        }
        Ok(Arc::new(Self::from_op(
            data,
            Op::Permute(Arc::clone(self), inv),
        )))
    }

    /// Multiply by scalar.
    pub fn mul_scalar(self: &Arc<Self>, val: f64) -> Result<Arc<Self>> {
        let data = self.tensor().mul_scalar(val)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::MulScalar(Arc::clone(self), val),
        )))
    }

    /// Add scalar.
    pub fn add_scalar(self: &Arc<Self>, val: f64) -> Result<Arc<Self>> {
        let data = self.tensor().add_scalar(val)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::AddScalar(Arc::clone(self), val),
        )))
    }

    /// ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise.
    pub fn elu(self: &Arc<Self>, alpha: f64) -> Result<Arc<Self>> {
        let data = self.tensor().elu(alpha)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Elu(Arc::clone(self), alpha),
        )))
    }

    /// HardSigmoid: max(0, min(1, x/6 + 0.5)).
    pub fn hard_sigmoid(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().hard_sigmoid()?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::HardSigmoid(Arc::clone(self)),
        )))
    }

    /// HardSwish: x * HardSigmoid(x).
    pub fn hard_swish(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().hard_swish()?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::HardSwish(Arc::clone(self)),
        )))
    }

    /// Mish: x * tanh(softplus(x)).
    pub fn mish(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().mish()?;
        Ok(Arc::new(Self::from_op(data, Op::Mish(Arc::clone(self)))))
    }

    /// SELU: lambda * (x if x >= 0, else alpha * (exp(x) - 1)).
    pub fn selu(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().selu()?;
        Ok(Arc::new(Self::from_op(data, Op::Selu(Arc::clone(self)))))
    }

    /// Softplus: log(1 + exp(x)).
    pub fn softplus(self: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().softplus()?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Softplus(Arc::clone(self)),
        )))
    }

    /// CELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1)).
    pub fn celu(self: &Arc<Self>, alpha: f64) -> Result<Arc<Self>> {
        let data = self.tensor().celu(alpha)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Celu(Arc::clone(self), alpha),
        )))
    }

    /// Log-softmax: log(softmax(x, dim)).
    pub fn log_softmax(self: &Arc<Self>, dim: usize) -> Result<Arc<Self>> {
        let data = self.tensor().log_softmax(dim)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::LogSoftmax(Arc::clone(self), dim),
        )))
    }

    /// Element-wise maximum of two tensors.
    pub fn maximum(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().maximum(rhs.tensor())?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Maximum(Arc::clone(self), Arc::clone(rhs)),
        )))
    }

    /// Element-wise minimum of two tensors.
    pub fn minimum(self: &Arc<Self>, rhs: &Arc<Self>) -> Result<Arc<Self>> {
        let data = self.tensor().minimum(rhs.tensor())?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Minimum(Arc::clone(self), Arc::clone(rhs)),
        )))
    }

    /// Stack tensors along a new dimension.
    pub fn stack(tensors: &[Arc<Self>], dim: usize) -> Result<Arc<Self>> {
        let dyn_tensors: Vec<nn_core::dyn_tensor::DynTensor> =
            tensors.iter().map(|t| t.tensor().clone()).collect();
        let refs: Vec<&nn_core::dyn_tensor::DynTensor> = dyn_tensors.iter().collect();
        let data = nn_core::dyn_tensor::DynTensor::stack(&refs, dim)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Stack(tensors.to_vec(), dim),
        )))
    }
}
