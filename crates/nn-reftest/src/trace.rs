// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Named tensor checkpoints and reference trace capture.

use crate::error::ReftestError;
#[cfg(feature = "nn-core")]
use nn_core::Tensor;

/// A named tensor checkpoint captured during model execution.
///
/// Data is always stored as `f32` for comparison, regardless of the original
/// dtype. Upcasting from f16/bf16 introduces no new error but means comparison
/// happens in f32 space.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NamedTensor {
    /// Checkpoint name (e.g. `"encoder.conv1"`).
    pub name: String,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Flattened f32 element data (row-major).
    pub data: Vec<f32>,
}

impl NamedTensor {
    /// Create a new named tensor from raw f32 data.
    ///
    /// Returns an error if `data.len()` does not match the product of `shape`,
    /// or if the shape product overflows `usize`.
    pub fn new(
        name: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<Self, ReftestError> {
        let name = name.into();
        let expected: usize = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| ReftestError::ShapeProductOverflow(shape.clone()))?;
        if data.len() != expected {
            return Err(ReftestError::ElementCountMismatch {
                name,
                shape,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { name, shape, data })
    }

    /// Number of elements in this tensor.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// An ordered sequence of named tensor checkpoints.
///
/// Used to record intermediate tensors during model execution for comparison
/// against a reference trace (typically exported from PyTorch).
#[derive(Debug, Default, Clone)]
pub struct ReferenceTrace {
    checkpoints: Vec<NamedTensor>,
}

impl ReferenceTrace {
    /// Create an empty trace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Capture checkpoints during a closure-driven forward pass.
    ///
    /// Returns both the captured trace and the closure output so callers can
    /// preserve their model return value.
    #[must_use]
    pub fn capture<F, T>(capture_fn: F) -> (Self, T)
    where
        F: FnOnce(&mut Self) -> T,
    {
        let mut trace = Self::new();
        let output = capture_fn(&mut trace);
        (trace, output)
    }

    /// Record a checkpoint from raw f32 data.
    ///
    /// Returns an error if `data.len()` does not match the product of `shape`.
    pub fn checkpoint(
        &mut self,
        name: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Result<(), ReftestError> {
        self.checkpoints
            .push(NamedTensor::new(name, shape.to_vec(), data.to_vec())?);
        Ok(())
    }

    /// Record a checkpoint from a rank-typed nn-core `Tensor<D, f32>`.
    ///
    /// Generic over rank `D` — works with tensors of any dimensionality.
    /// The tensor must be on CPU (default [`CpuBackend`]).
    #[cfg(feature = "nn-core")]
    pub fn checkpoint_tensor<const D: usize>(
        &mut self,
        name: &str,
        tensor: &Tensor<D, f32>,
    ) -> Result<(), ReftestError> {
        let arr = tensor.as_ndarray();
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<f32> = arr.iter().copied().collect();
        self.checkpoints.push(NamedTensor::new(name, shape, data)?);
        Ok(())
    }

    /// Record a checkpoint from an nn-core f64 tensor, converting to f32.
    ///
    /// Returns an error if any f64 value is non-finite or exceeds `f32::MAX`
    /// in magnitude, which would silently corrupt the reference baseline.
    ///
    /// Generic over rank `D` — works with tensors of any dimensionality.
    #[cfg(feature = "nn-core")]
    pub fn checkpoint_tensor_f64<const D: usize>(
        &mut self,
        name: &str,
        tensor: &Tensor<D, f64>,
    ) -> Result<(), ReftestError> {
        let arr = tensor.as_ndarray();
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<f32> = arr
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if !v.is_finite() || v.abs() > f64::from(f32::MAX) {
                    return Err(ReftestError::F64OutOfF32Range { value: v, index: i });
                }
                Ok(v as f32)
            })
            .collect::<Result<Vec<f32>, _>>()?;
        self.checkpoints.push(NamedTensor::new(name, shape, data)?);
        Ok(())
    }

    /// Number of checkpoints in this trace.
    #[must_use]
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Whether this trace has no checkpoints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Access checkpoint by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&NamedTensor> {
        self.checkpoints.get(index)
    }

    /// Access checkpoint by name (first match).
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<&NamedTensor> {
        self.checkpoints.iter().find(|c| c.name == name)
    }

    /// Iterate over all checkpoints in order.
    pub fn iter(&self) -> impl Iterator<Item = &NamedTensor> {
        self.checkpoints.iter()
    }

    /// Checkpoint names in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.checkpoints.iter().map(|c| c.name.as_str())
    }

    /// Consume the trace and return the underlying checkpoints.
    #[must_use]
    pub fn into_checkpoints(self) -> Vec<NamedTensor> {
        self.checkpoints
    }

    /// Build a trace from a vector of named tensors.
    #[must_use]
    pub fn from_checkpoints(checkpoints: Vec<NamedTensor>) -> Self {
        Self { checkpoints }
    }
}

/// Convert a rank-typed nn-core `Tensor<D, f32>` into a `NamedTensor`.
///
/// The name is set to an empty string; callers should set it afterwards or
/// use [`ReferenceTrace::checkpoint_tensor`].
#[cfg(feature = "nn-core")]
impl<const D: usize> From<&Tensor<D, f32>> for NamedTensor {
    fn from(tensor: &Tensor<D, f32>) -> Self {
        let arr = tensor.as_ndarray();
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<f32> = arr.iter().copied().collect();
        Self {
            name: String::new(),
            shape,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_tensor_new() {
        let t = NamedTensor::new("layer1", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid tensor");
        assert_eq!(t.name, "layer1");
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_named_tensor_shape_mismatch_returns_error() {
        let result = NamedTensor::new("bad", vec![2, 3], vec![1.0, 2.0]);
        assert!(
            matches!(result, Err(ReftestError::ElementCountMismatch { .. })),
            "expected ElementCountMismatch, got {result:?}",
        );
    }

    #[test]
    fn test_reference_trace_checkpoint() {
        let mut trace = ReferenceTrace::new();
        trace.checkpoint("a", &[1.0, 2.0], &[2]).expect("valid");
        trace
            .checkpoint("b", &[3.0, 4.0, 5.0], &[3])
            .expect("valid");

        assert_eq!(trace.len(), 2);
        assert!(!trace.is_empty());
        assert_eq!(trace.get(0).expect("should exist").name, "a");
        assert_eq!(trace.get_by_name("b").expect("should exist").numel(), 3);
    }

    #[test]
    fn test_reference_trace_names() {
        let mut trace = ReferenceTrace::new();
        trace
            .checkpoint("encoder.conv1", &[1.0], &[1])
            .expect("valid");
        trace
            .checkpoint("encoder.relu1", &[2.0], &[1])
            .expect("valid");

        let names: Vec<&str> = trace.names().collect();
        assert_eq!(names, vec!["encoder.conv1", "encoder.relu1"]);
    }

    #[test]
    fn test_scalar_checkpoint() {
        let mut trace = ReferenceTrace::new();
        trace.checkpoint("loss", &[0.5], &[1]).expect("valid");
        assert_eq!(trace.get(0).expect("should exist").data, vec![0.5]);
    }

    #[test]
    fn test_capture_returns_trace_and_output() {
        let (trace, output) = ReferenceTrace::capture(|capture| {
            capture
                .checkpoint("hidden", &[1.0, 2.0], &[2])
                .expect("valid");
            42usize
        });

        assert_eq!(output, 42);
        assert_eq!(trace.len(), 1);
        assert_eq!(trace.get(0).expect("exists").name, "hidden");
    }

    #[test]
    fn test_named_tensor_shape_overflow_returns_error() {
        let result = NamedTensor::new("overflow", vec![usize::MAX, 2], vec![]);
        assert!(
            matches!(result, Err(ReftestError::ShapeProductOverflow(_))),
            "expected ShapeProductOverflow, got {result:?}",
        );
    }

    #[test]
    fn test_checkpoint_shape_mismatch_returns_error() {
        let mut trace = ReferenceTrace::new();
        let result = trace.checkpoint("bad", &[1.0, 2.0], &[3]);
        assert!(
            matches!(result, Err(ReftestError::ElementCountMismatch { .. })),
            "expected ElementCountMismatch, got {result:?}",
        );
        assert!(trace.is_empty(), "no checkpoint should be added on error");
    }

    #[test]
    fn test_get_returns_none_for_out_of_bounds() {
        let mut trace = ReferenceTrace::new();
        trace.checkpoint("only", &[1.0], &[1]).expect("valid");
        assert!(trace.get(0).is_some());
        assert!(trace.get(1).is_none());
        assert!(trace.get(100).is_none());
    }

    #[test]
    fn test_get_by_name_returns_none_for_missing() {
        let mut trace = ReferenceTrace::new();
        trace.checkpoint("exists", &[1.0], &[1]).expect("valid");
        assert!(trace.get_by_name("exists").is_some());
        assert!(trace.get_by_name("missing").is_none());
        assert!(trace.get_by_name("").is_none());
    }

    #[test]
    fn test_into_checkpoints_roundtrip() {
        let mut trace = ReferenceTrace::new();
        trace.checkpoint("a", &[1.0, 2.0], &[2]).expect("valid");
        trace
            .checkpoint("b", &[3.0, 4.0, 5.0], &[3])
            .expect("valid");

        let checkpoints = trace.into_checkpoints();
        assert_eq!(checkpoints.len(), 2);
        assert_eq!(checkpoints[0].name, "a");
        assert_eq!(checkpoints[1].name, "b");

        let rebuilt = ReferenceTrace::from_checkpoints(checkpoints);
        assert_eq!(rebuilt.len(), 2);
        assert_eq!(rebuilt.get(0).expect("exists").name, "a");
        assert_eq!(rebuilt.get(1).expect("exists").name, "b");
    }

    #[test]
    fn test_empty_trace_methods() {
        let trace = ReferenceTrace::new();
        assert!(trace.is_empty());
        assert_eq!(trace.len(), 0);
        assert!(trace.get(0).is_none());
        assert!(trace.get_by_name("anything").is_none());
        let names: Vec<&str> = trace.names().collect();
        assert!(names.is_empty());
        let iter_count = trace.iter().count();
        assert_eq!(iter_count, 0);
    }

    #[test]
    fn test_named_tensor_zero_dim_scalar() {
        // Empty shape = scalar with 1 element.
        let t = NamedTensor::new("scalar", vec![], vec![42.0]).expect("valid scalar");
        assert!(t.shape.is_empty());
        assert_eq!(t.numel(), 1);
        assert_eq!(t.data, vec![42.0]);
    }

    #[test]
    fn test_named_tensor_zero_in_shape_means_zero_elements() {
        // Shape [2, 0, 3] means 0 elements.
        let t =
            NamedTensor::new("zero_dim", vec![2, 0, 3], vec![]).expect("valid zero-element tensor");
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn test_named_tensor_high_rank() {
        // 4-D tensor.
        let data = vec![0.0_f32; 2 * 3 * 4 * 5];
        let t = NamedTensor::new("4d", vec![2, 3, 4, 5], data).expect("valid 4D tensor");
        assert_eq!(t.numel(), 120);
        assert_eq!(t.shape, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_capture_empty_closure() {
        let (trace, result) = ReferenceTrace::capture(|_capture| "done");
        assert!(trace.is_empty());
        assert_eq!(result, "done");
    }

    #[test]
    fn test_checkpoint_with_nan_data_accepted() {
        // NamedTensor::new does not validate element values, only shape.
        let mut trace = ReferenceTrace::new();
        trace
            .checkpoint(
                "nan_layer",
                &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
                &[3],
            )
            .expect("NaN/Inf data should be accepted at checkpoint time");
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_multiple_get_by_name_returns_first_match() {
        let checkpoints = vec![
            NamedTensor::new("dup", vec![1], vec![1.0]).expect("valid"),
            NamedTensor::new("dup", vec![1], vec![2.0]).expect("valid"),
        ];
        let trace = ReferenceTrace::from_checkpoints(checkpoints);
        let found = trace.get_by_name("dup").expect("should find first");
        assert_eq!(found.data, vec![1.0], "should return first match");
    }

    #[test]
    fn test_named_tensor_element_count_mismatch_details() {
        let result = NamedTensor::new("detail_test", vec![2, 3], vec![1.0, 2.0]);
        match result {
            Err(ReftestError::ElementCountMismatch {
                name,
                shape,
                expected,
                actual,
            }) => {
                assert_eq!(name, "detail_test");
                assert_eq!(shape, vec![2, 3]);
                assert_eq!(expected, 6);
                assert_eq!(actual, 2);
            }
            other => panic!("expected ElementCountMismatch, got {other:?}"),
        }
    }
}
