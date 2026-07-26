// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Softmax and log-softmax operations with dimension parameter.
//!
//! These free functions match candle's `candle_nn::ops::softmax(x, dim)` API,
//! accepting `impl Dim` so negative indices like `D::Minus1` work.
//! They delegate to the corresponding `DynTensor` methods.

use crate::dyn_tensor::{Dim, DynTensor};
use crate::error::Result;

/// Softmax along a specific dimension (rejects NaN, allows Inf/-Inf).
///
/// Delegates to [`DynTensor::softmax`] which is numerically stable
/// (max-subtraction), handles CPU/GPU tensors, and rejects NaN while
/// allowing `-Inf` for attention masks.
///
/// Accepts `impl Dim` matching candle's `candle_nn::ops::softmax` signature,
/// so `softmax(&x, D::Minus1)` works.
pub fn softmax(x: &DynTensor, dim: impl Dim) -> Result<DynTensor> {
    x.softmax(dim)
}

/// Element-wise sigmoid: `1 / (1 + exp(-x))`.
///
/// Matches candle-nn's `candle_nn::ops::sigmoid()` free function.
/// Delegates to `DynTensor::sigmoid()`.
pub fn sigmoid(x: &DynTensor) -> Result<DynTensor> {
    x.sigmoid()
}

/// Log-softmax along a specific dimension (rejects NaN, allows Inf/-Inf).
///
/// Delegates to [`DynTensor::log_softmax`] which computes
/// `x - max - log(sum(exp(x - max)))` (numerically stable).
///
/// Accepts `impl Dim` matching candle's API signature.
pub fn log_softmax(x: &DynTensor, dim: impl Dim) -> Result<DynTensor> {
    x.log_softmax(dim)
}

#[cfg(kani)]
#[path = "kani_ops_proofs.rs"]
mod kani_ops_proofs;

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_softmax_1d() {
        let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let y = softmax(&x, 0).unwrap();
        let flat = y.to_flat_vec::<f32>().unwrap();
        // Should sum to 1
        let sum: f32 = flat.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Monotonically increasing
        assert!(flat[0] < flat[1]);
        assert!(flat[1] < flat[2]);
    }

    #[test]
    fn test_softmax_2d() {
        let x =
            DynTensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], &Device::Cpu).unwrap();
        let y = softmax(&x, 1).unwrap();
        assert_eq!(y.dims(), &[2, 3]);
        let flat = y.to_flat_vec::<f32>().unwrap();
        // Each row sums to 1
        let row0_sum: f32 = flat[..3].iter().sum();
        let row1_sum: f32 = flat[3..].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
        assert!((row1_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_1d() {
        let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let y = log_softmax(&x, 0).unwrap();
        let flat = y.to_flat_vec::<f32>().unwrap();
        // All values should be negative (log of probability < 1)
        assert!(flat.iter().all(|&v| v < 0.0));
        // exp(log_softmax) should sum to 1
        let exp_sum: f32 = flat.iter().map(|v| v.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_dim_out_of_range() {
        let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
        assert!(softmax(&x, 1).is_err());
    }

    #[test]
    fn test_log_softmax_consistency() {
        // log_softmax should equal softmax().log()
        let x = DynTensor::from_vec(vec![0.5, 1.5, 2.5], &[3], &Device::Cpu).unwrap();
        let ls = log_softmax(&x, 0).unwrap();
        let s = softmax(&x, 0).unwrap().log().unwrap();
        let ls_flat = ls.to_flat_vec::<f32>().unwrap();
        let s_flat = s.to_flat_vec::<f32>().unwrap();
        for (a, b) in ls_flat.iter().zip(s_flat.iter()) {
            assert!((a - b).abs() < 1e-5, "log_softmax mismatch: {a} vs {b}");
        }
    }

    // -- NaN rejection tests --
    // Validation is device-agnostic (no Device::Cpu guard). These tests exercise
    // the unconditional path. GPU-backed tensor NaN tests require nn-metal
    // integration tests (Metal runtime needed for DynTensor::to_device).

    #[test]
    fn test_softmax_nan_input_rejected() {
        let x = DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3], &Device::Cpu).unwrap();
        let result = softmax(&x, 0);
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("non-finite") || msg.contains("Non-finite"),
            "softmax should reject NaN input, got: {msg}"
        );
    }

    #[test]
    fn test_softmax_inf_input_gives_correct_probabilities() {
        // +Inf dominates: gets probability 1.0, finite positions get 0.0.
        let x = DynTensor::from_vec(vec![1.0, f32::INFINITY, 3.0], &[3], &Device::Cpu).unwrap();
        let y = softmax(&x, 0).unwrap();
        let flat = y.to_flat_vec::<f32>().unwrap();
        assert!(flat[0] == 0.0, "finite should be 0, got {}", flat[0]);
        assert!(
            (flat[1] - 1.0).abs() < 1e-6,
            "+inf should get prob 1.0, got {}",
            flat[1]
        );
        assert!(flat[2] == 0.0, "finite should be 0, got {}", flat[2]);
    }

    #[test]
    fn test_softmax_neg_inf_input_allowed() {
        // -Inf produces 0 probability (exp(-Inf) = 0) — correct for attention masks
        let x = DynTensor::from_vec(vec![1.0, f32::NEG_INFINITY, 3.0], &[3], &Device::Cpu).unwrap();
        let y = softmax(&x, 0).unwrap();
        let flat = y.to_flat_vec::<f32>().unwrap();
        // -Inf position (index 1) should have probability 0
        assert!(
            flat[1].abs() < 1e-7,
            "exp(-Inf) should be 0, got {}",
            flat[1]
        );
        // Remaining probabilities should sum to ~1
        let sum: f32 = flat.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_log_softmax_nan_input_rejected() {
        let x = DynTensor::from_vec(vec![f32::NAN, 2.0], &[2], &Device::Cpu).unwrap();
        let result = log_softmax(&x, 0);
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("non-finite") || msg.contains("Non-finite"),
            "log_softmax should reject NaN input, got: {msg}"
        );
    }

    #[test]
    fn test_log_softmax_inf_input_gives_correct_values() {
        // +Inf dominates: log(1) = 0 at +inf, log(0) = -inf at finite.
        let x = DynTensor::from_vec(vec![f32::INFINITY, 2.0], &[2], &Device::Cpu).unwrap();
        let y = log_softmax(&x, 0).unwrap();
        let flat = y.to_flat_vec::<f32>().unwrap();
        assert!(
            (flat[0] - 0.0).abs() < 1e-6,
            "log_softmax at +inf should be 0.0, got {}",
            flat[0]
        );
        assert!(
            flat[1] == f32::NEG_INFINITY,
            "log_softmax at finite should be -inf, got {}",
            flat[1]
        );
    }

    #[test]
    fn test_log_softmax_dim_out_of_range() {
        let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
        assert!(log_softmax(&x, 1).is_err());
    }

    #[test]
    fn test_softmax_with_dim_enum() {
        // Verify impl Dim works with D::Minus1
        use crate::D;
        let x =
            DynTensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], &Device::Cpu).unwrap();
        let y = softmax(&x, D::Minus1).unwrap();
        assert_eq!(y.dims(), &[2, 3]);
        let flat = y.to_flat_vec::<f32>().unwrap();
        // Each row (last dim) should sum to 1
        let row0_sum: f32 = flat[..3].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_with_dim_enum() {
        use crate::D;
        let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let y = log_softmax(&x, D::Minus1).unwrap();
        let flat = y.to_flat_vec::<f32>().unwrap();
        assert!(flat.iter().all(|&v| v < 0.0));
    }
}
