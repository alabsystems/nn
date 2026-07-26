// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `unfold()` operation for [`DynTensor`].
//!
//! Extracts overlapping sliding windows along a dimension in a single operation.
//! Equivalent to PyTorch's `Tensor.unfold(dimension, size, step)`.
//!
//! For a tensor `[B, C, T]` with `unfold(2, window_size, hop_size)`, produces
//! `[B, C, n_windows, window_size]` where `n_windows = (T - window_size) / hop_size + 1`.
//!
//! This is the core primitive for STFT framing — replaces 87K narrow() calls with
//! a single operation (#1945).

use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{
    checked_dim_product, gpu_backend_dispatch, trace, Dim, DynTensor, TensorStorage,
};
use crate::{DType, Device, Result, TensorError};
use ndarray::{ArrayD, IxDyn};

impl DynTensor {
    /// Extract overlapping sliding windows along a dimension.
    ///
    /// Returns a tensor with an additional trailing dimension of size `size`.
    /// For input shape `[d0, ..., d_dim, ..., dN]`, output shape is
    /// `[d0, ..., n_windows, ..., dN, size]` where the `dim` axis is replaced
    /// by `n_windows = (d_dim - size) / step + 1`.
    ///
    /// Matches PyTorch's `Tensor.unfold(dimension, size, step)`.
    ///
    /// # Example
    ///
    /// ```text
    /// input:  [1, 1, 10]  (1 batch, 1 channel, 10 timesteps)
    /// unfold(dim=2, size=4, step=2)
    /// output: [1, 1, 4, 4]  (4 windows of size 4, hop=2)
    ///   window 0: [0, 1, 2, 3]
    ///   window 1: [2, 3, 4, 5]
    ///   window 2: [4, 5, 6, 7]
    ///   window 3: [6, 7, 8, 9]
    /// ```
    ///
    /// # STFT framing
    ///
    /// For STFT framing on `[B, C, T]`: `x.unfold(2, fft_size, hop_size)?`
    /// produces `[B, C, n_frames, fft_size]` in a single GPU dispatch, replacing
    /// O(n_frames) narrow() calls.
    ///
    /// # GPU dispatch
    ///
    /// Float GPU tensors use native Metal kernel dispatch via
    /// [`GpuShapeOps::unfold`]. Non-float GPU tensors fall back to CPU round-trip.
    pub fn unfold(&self, dim: impl Dim, size: usize, step: usize) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        if size == 0 {
            return Err(TensorError::InvalidShape("unfold: size must be > 0".into()));
        }
        if step == 0 {
            return Err(TensorError::InvalidShape("unfold: step must be > 0".into()));
        }
        let mut result = self.unfold_dispatch(dim, size, step)?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Unfold { dim, size, step },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Internal dispatch for unfold — separated so trace recording wraps all paths.
    fn unfold_dispatch(&self, dim: usize, size: usize, step: usize) -> Result<Self> {
        let dim_size = self.dims[dim];
        if size > dim_size {
            return Err(TensorError::InvalidShape(format!(
                "unfold: size ({size}) exceeds dimension size ({dim_size})"
            )));
        }
        let n_windows = (dim_size - size) / step + 1;
        if n_windows == 0 {
            return Err(TensorError::InvalidShape(
                "unfold: no windows fit (size > dimension size)".into(),
            ));
        }

        // Build output shape: replace dim with n_windows, append size at end.
        let mut out_shape: Vec<usize> = self.dims.clone();
        out_shape[dim] = n_windows;
        out_shape.push(size);

        // Try native GPU dispatch first.
        if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.unfold(self, dim, size, step)) {
                return result;
            }
            // CPU fallback for non-float GPU tensors.
            let cpu = self.to_device(&Device::Cpu)?;
            let result = cpu.unfold(dim, size, step)?;
            return result.to_device(&self.device());
        }

        // Auto-dequantize quantized tensors.
        if self.is_quantized() {
            return self.dequantize()?.unfold(dim, size, step);
        }

        // CPU implementation: direct element mapping.
        match &self.storage {
            TensorStorage::Cpu(_) => {
                dispatch_cpu_typed!(
                    self,
                    |arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                        unfold_ndarray(arr, dim, size, step, &out_shape)
                    },
                    "unfold"
                )
            }
            TensorStorage::Gpu { .. } => Err(TensorError::Unsupported(
                "unfold: GPU tensor reached CPU path".into(),
            )),
            TensorStorage::Quantized(_) => unreachable!("handled above"),
        }
    }
}

/// CPU unfold implementation using ndarray.
///
/// For each output element at `[i0, ..., w, ..., iN, k]` (where w is the
/// window index at position `dim` and k is the position within the window),
/// the corresponding input element is at `[i0, ..., w*step + k, ..., iN]`.
fn unfold_ndarray<T: Copy + Default + 'static>(
    arr: &ArrayD<T>,
    dim: usize,
    _size: usize,
    step: usize,
    out_shape: &[usize],
) -> Result<ArrayD<T>> {
    let in_shape = arr.shape();
    let in_rank = in_shape.len();
    let out_rank = out_shape.len(); // in_rank + 1

    // Total output elements.
    let total: usize = checked_dim_product(out_shape)?;
    let mut out_data = vec![T::default(); total];

    // Compute strides for output index decomposition.
    let mut out_strides = vec![1usize; out_rank];
    for i in (0..out_rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Compute strides for input index composition.
    let mut in_strides = vec![1usize; in_rank];
    for i in (0..in_rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    let in_flat = arr.as_standard_layout();
    let in_slice = in_flat.as_slice().ok_or_else(|| {
        TensorError::InvalidShape("unfold: input not contiguous after standard layout".into())
    })?;

    for (flat_idx, out_elem) in out_data.iter_mut().enumerate() {
        // Decompose flat output index into multi-dim index.
        let mut remaining = flat_idx;
        let mut in_flat_idx = 0usize;

        for d in 0..out_rank {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];

            if d == dim {
                // Window index: contributes w*step to input dim.
                in_flat_idx += coord * step * in_strides[d];
            } else if d == out_rank - 1 {
                // Within-window position k: add k to input's unfold dim.
                in_flat_idx += coord * in_strides[dim];
            } else {
                // Regular axis (before or after unfold dim): maps directly.
                in_flat_idx += coord * in_strides[d];
            }
        }

        *out_elem = in_slice[in_flat_idx];
    }

    Ok(ArrayD::from_shape_vec(IxDyn(out_shape), out_data)?)
}

#[cfg(test)]
mod tests {
    use crate::dyn_tensor::test_helpers::{cpu, tnd};

    fn flat_f32(t: &crate::dyn_tensor::DynTensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn test_unfold_1d() {
        // [10] unfold(dim=0, size=4, step=2) -> [4, 4]
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let t = tnd(&data, &[10]);
        let u = t.unfold(0, 4, 2).unwrap();
        assert_eq!(u.dims(), &[4, 4]);
        let vals = flat_f32(&u);
        // window 0: [0,1,2,3], window 1: [2,3,4,5], window 2: [4,5,6,7], window 3: [6,7,8,9]
        assert_eq!(
            vals,
            vec![0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_unfold_2d_dim1() {
        // [2, 6] unfold(dim=1, size=3, step=2) -> [2, 2, 3]
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let t = tnd(&data, &[2, 6]);
        let u = t.unfold(1, 3, 2).unwrap();
        assert_eq!(u.dims(), &[2, 2, 3]);
        let vals = flat_f32(&u);
        // Row 0: [0,1,2,3,4,5] -> window0=[0,1,2], window1=[2,3,4]
        // Row 1: [6,7,8,9,10,11] -> window0=[6,7,8], window1=[8,9,10]
        assert_eq!(
            vals,
            vec![0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0]
        );
    }

    #[test]
    fn test_unfold_3d_stft_pattern() {
        // [1, 1, 8] unfold(dim=2, size=4, step=2) -> [1, 1, 3, 4]
        // This is the STFT framing pattern.
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let t = tnd(&data, &[1, 1, 8]);
        let u = t.unfold(2, 4, 2).unwrap();
        assert_eq!(u.dims(), &[1, 1, 3, 4]);
        let vals = flat_f32(&u);
        // 3 windows of size 4 with hop 2:
        // [0,1,2,3], [2,3,4,5], [4,5,6,7]
        assert_eq!(
            vals,
            vec![0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn test_unfold_step_1() {
        // [5] unfold(dim=0, size=3, step=1) -> [3, 3]
        let data: Vec<f32> = (0..5).map(|x| x as f32).collect();
        let t = tnd(&data, &[5]);
        let u = t.unfold(0, 3, 1).unwrap();
        assert_eq!(u.dims(), &[3, 3]);
        let vals = flat_f32(&u);
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_unfold_non_overlapping() {
        // [8] unfold(dim=0, size=4, step=4) -> [2, 4] (no overlap)
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let t = tnd(&data, &[8]);
        let u = t.unfold(0, 4, 4).unwrap();
        assert_eq!(u.dims(), &[2, 4]);
        let vals = flat_f32(&u);
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_unfold_dim0_on_3d() {
        // [4, 2, 3] unfold(dim=0, size=2, step=1) -> [3, 2, 3, 2]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = tnd(&data, &[4, 2, 3]);
        let u = t.unfold(0, 2, 1).unwrap();
        assert_eq!(u.dims(), &[3, 2, 3, 2]);
        let vals = flat_f32(&u);
        // window 0 at dim 0: slices 0..2, window 1: slices 1..3, window 2: slices 2..4
        // The trailing dim picks which slice within the window:
        // [batch_idx=0, c=0, t=0]: input[0,0,0]=0 and input[1,0,0]=6
        assert_eq!(vals[0], 0.0); // w=0, c=0, t=0, k=0 -> input[0,0,0]
        assert_eq!(vals[1], 6.0); // w=0, c=0, t=0, k=1 -> input[1,0,0]
    }

    #[test]
    fn test_unfold_error_size_zero() {
        let t = crate::dyn_tensor::DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
        assert!(t.unfold(0, 0, 1).is_err());
    }

    #[test]
    fn test_unfold_error_step_zero() {
        let t = crate::dyn_tensor::DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
        assert!(t.unfold(0, 1, 0).is_err());
    }

    #[test]
    fn test_unfold_error_size_exceeds_dim() {
        let t = crate::dyn_tensor::DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
        assert!(t.unfold(0, 3, 1).is_err());
    }

    #[test]
    fn test_unfold_error_dim_oob() {
        let t = crate::dyn_tensor::DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
        assert!(t.unfold(1, 1, 1).is_err());
    }

    #[test]
    fn test_unfold_single_window() {
        // [4] unfold(dim=0, size=4, step=1) -> [1, 4]
        let data: Vec<f32> = (0..4).map(|x| x as f32).collect();
        let t = tnd(&data, &[4]);
        let u = t.unfold(0, 4, 1).unwrap();
        assert_eq!(u.dims(), &[1, 4]);
        let vals = flat_f32(&u);
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_unfold_trace_records_op() {
        use crate::dyn_tensor::trace::{record_input, trace_graph, TraceOp};
        use crate::DType;

        let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 8]);
        let (result, graph) = trace_graph(|| {
            let mut a = a.clone();
            let id = record_input(&[1, 1, 8], DType::F32).unwrap();
            a.set_trace_id(id);
            let b = a.unfold(2, 4, 2)?;
            Ok(b)
        })
        .unwrap();

        assert_eq!(result.dims(), &[1, 1, 3, 4]);
        let nodes = graph.nodes();
        assert_eq!(nodes.len(), 2); // input + unfold
        let output = graph.output_node().unwrap();
        match output.op() {
            TraceOp::Unfold { dim, size, step } => {
                assert_eq!(*dim, 2);
                assert_eq!(*size, 4);
                assert_eq!(*step, 2);
            }
            other => panic!("expected Unfold, got {other:?}"),
        }
        assert_eq!(output.output_shape(), &[1, 1, 3, 4]);
        assert_eq!(output.inputs().len(), 1);
    }
}
