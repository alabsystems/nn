// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for nn-optim.

/// Errors from optimizer operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OptimError {
    /// Tensor operation error from nn-core.
    #[error(transparent)]
    Tensor(#[from] nn_core::TensorError),

    /// Autodiff error during backward_step.
    #[error(transparent)]
    Autodiff(#[from] nn_autodiff::AutodiffError),

    /// Invalid optimizer hyperparameter value.
    #[error("invalid {param}: {reason}")]
    InvalidParam { param: &'static str, reason: String },

    /// Missing internal optimizer state (e.g., missing row/col factor).
    #[error("{optimizer}: missing {state}")]
    MissingState {
        optimizer: &'static str,
        state: &'static str,
    },

    /// Non-finite (NaN or Inf) values detected in gradients.
    /// Silent propagation of non-finite gradients corrupts moment estimates
    /// and parameter values irreversibly.
    #[error("non-finite gradient: {count} NaN/Inf values")]
    NonFiniteGradient {
        /// Number of non-finite elements in the gradient tensor.
        count: usize,
    },

    /// Non-finite (NaN or Inf) values detected in updated parameters.
    /// Finite gradients can still produce infinite parameters through
    /// intermediate overflow (e.g., theta near f32::MAX with increasing update).
    /// Once Inf enters parameters, all subsequent gradients become NaN permanently.
    #[error("non-finite parameter update: {count} NaN/Inf values")]
    NonFiniteUpdate {
        /// Number of non-finite elements in the updated parameter tensor.
        count: usize,
    },

    /// I/O error during checkpoint save/load.
    #[error("checkpoint I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Checkpoint tensor shape mismatch during restore.
    #[error("checkpoint shape mismatch for '{key}': expected {expected:?}, got {got:?}")]
    CheckpointShapeMismatch {
        key: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Checkpoint tensor contains non-finite (NaN/Inf) values.
    #[error("checkpoint '{key}' contains {count} non-finite (NaN/Inf) values")]
    NonFiniteCheckpoint { key: String, count: usize },

    /// Checkpoint step value exceeds platform limits.
    #[error("checkpoint step value {step} exceeds usize::MAX")]
    CheckpointStepOverflow { step: i64 },

    /// Checkpoint serialization/deserialization error.
    #[error("checkpoint serialization error: {reason}")]
    CheckpointSerde { reason: String },

    /// Corrupted internal optimizer state (invariant violation).
    /// This should never occur during normal operation — it indicates
    /// a bug in state initialization or checkpoint restoration.
    #[error("corrupted optimizer state: {optimizer}: {reason}")]
    CorruptedState {
        optimizer: &'static str,
        reason: &'static str,
    },
}

/// Convenience result type for optimizer operations.
pub type Result<T> = std::result::Result<T, OptimError>;

/// Count non-finite elements in a tensor, handling GPU tensors by moving to CPU first.
fn count_non_finite(tensor: &nn_core::dyn_tensor::DynTensor) -> Result<usize> {
    let cpu_tensor = if tensor.device().is_gpu() {
        tensor
            .to_device(&nn_core::device::Device::Cpu)
            .map_err(OptimError::Tensor)?
    } else {
        tensor.clone()
    };
    let data = cpu_tensor.to_f32_array().map_err(OptimError::Tensor)?;
    Ok(data.iter().filter(|v| !v.is_finite()).count())
}

/// Validate that a gradient tensor contains no NaN or Inf values.
///
/// Called at the top of each optimizer `step()` to prevent silent corruption
/// of moment estimates and parameter values. Once NaN enters optimizer state,
/// it persists permanently (exponential moving averages never recover).
pub(crate) fn validate_gradient(grad: &nn_core::dyn_tensor::DynTensor) -> Result<()> {
    if grad.any_non_finite().map_err(OptimError::Tensor)? {
        let count = count_non_finite(grad)?;
        return Err(OptimError::NonFiniteGradient { count });
    }
    Ok(())
}

/// Validate a learning rate: must be finite and non-negative.
pub(crate) fn validate_lr(lr: f64) -> Result<()> {
    if !lr.is_finite() || lr < 0.0 {
        return Err(OptimError::InvalidParam {
            param: "lr",
            reason: format!("must be non-negative and finite, got {lr}"),
        });
    }
    Ok(())
}

/// Validate weight decay: must be finite and non-negative.
pub(crate) fn validate_weight_decay(wd: f64) -> Result<()> {
    if !wd.is_finite() || wd < 0.0 {
        return Err(OptimError::InvalidParam {
            param: "weight_decay",
            reason: format!("must be non-negative and finite, got {wd}"),
        });
    }
    Ok(())
}

/// Validate that a restored checkpoint tensor contains no NaN or Inf values.
///
/// Called during `load_checkpoint()` to prevent silently loading corrupted
/// optimizer state. NaN/Inf in moment estimates (Adam m/v, SGD velocity,
/// AdaFactor row/col factors) persists permanently through exponential
/// moving averages — a single corrupted checkpoint poisons all future steps.
pub(crate) fn validate_checkpoint_tensor(
    tensor: &nn_core::dyn_tensor::DynTensor,
    key: &str,
) -> Result<()> {
    if tensor.any_non_finite().map_err(OptimError::Tensor)? {
        let count = count_non_finite(tensor)?;
        return Err(OptimError::NonFiniteCheckpoint {
            key: key.to_string(),
            count,
        });
    }
    Ok(())
}

/// Validate that an updated parameter tensor contains no NaN or Inf values.
///
/// Called after each optimizer parameter update to catch intermediate overflow.
/// Finite gradients can still produce infinite parameters when theta is near
/// f32::MAX and the update direction increases magnitude.
pub(crate) fn validate_update(new_theta: &nn_core::dyn_tensor::DynTensor) -> Result<()> {
    if new_theta.any_non_finite().map_err(OptimError::Tensor)? {
        let count = count_non_finite(new_theta)?;
        return Err(OptimError::NonFiniteUpdate { count });
    }
    Ok(())
}
