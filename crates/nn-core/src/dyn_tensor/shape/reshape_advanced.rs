// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced reshape operations for [`DynTensor`].
//!
//! - `repeat_interleave_n`: uniform-count repeat-interleave (each element repeated N times)
//! - `tile` enhancement: NumPy-compatible tile with auto-padding for shorter reps

use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

impl DynTensor {
    /// Repeat each element `repeats` times along `dim`.
    ///
    /// This is the uniform-count variant of [`repeat_interleave`](Self::repeat_interleave),
    /// equivalent to PyTorch `torch.repeat_interleave(tensor, repeats, dim)` where
    /// `repeats` is an integer (not a tensor).
    ///
    /// # Example
    ///
    /// ```text
    /// input:  [1, 2, 3]  shape [3]
    /// repeat_interleave_n(2, 0)
    /// output: [1, 1, 2, 2, 3, 3]  shape [6]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `repeats` - Number of times to repeat each element (must be > 0)
    /// * `dim` - Dimension along which to repeat
    pub fn repeat_interleave_n(&self, repeats: usize, dim: usize) -> Result<Self> {
        if dim >= self.rank() {
            return Err(TensorError::DimensionOutOfRange {
                dim,
                rank: self.rank(),
            });
        }
        if repeats == 0 {
            // Zero repeats → empty along dim.
            let mut out_dims = self.dims().to_vec();
            out_dims[dim] = 0;
            return Self::zeros(&out_dims, self.dtype(), &self.device());
        }
        if repeats == 1 {
            return Ok(self.clone());
        }

        // Build a uniform counts tensor and delegate to the tensor-based variant.
        let dim_size = self.dims()[dim];
        let counts_data: Vec<f32> = vec![repeats as f32; dim_size];
        let counts = Self::from_vec(counts_data, &[dim_size], &Device::Cpu)?;
        self.repeat_interleave(dim, &counts)
    }

    /// Tile (repeat) the tensor along each dimension, with NumPy-compatible
    /// auto-padding.
    ///
    /// If `reps` is shorter than the tensor's rank, it is left-padded with 1s
    /// to match. This matches `numpy.tile()` semantics.
    ///
    /// # Example
    ///
    /// ```text
    /// input:  [[1, 2], [3, 4]]  shape [2, 2]
    /// tile_numpy(&[3])     // left-padded to [1, 3]
    /// output: [[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]  shape [2, 6]
    /// ```
    ///
    /// When `reps.len() == self.rank()`, this is identical to [`tile`](Self::tile).
    pub fn tile_numpy(&self, reps: &[usize]) -> Result<Self> {
        if reps.len() > self.rank() {
            return Err(TensorError::InvalidShape(format!(
                "tile_numpy: reps length {} exceeds tensor rank {}",
                reps.len(),
                self.rank()
            )));
        }
        // Left-pad with 1s to match rank.
        let pad_len = self.rank() - reps.len();
        let mut full_reps = vec![1usize; pad_len];
        full_reps.extend_from_slice(reps);
        self.repeat(&full_reps)
    }
}

#[cfg(test)]
mod tests {
    use crate::dyn_tensor::test_helpers::{t1d, t2d, tnd};

    fn flat_f32(t: &crate::dyn_tensor::DynTensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    // ---- repeat_interleave_n ------------------------------------------------

    #[test]
    fn test_repeat_interleave_n_1d() {
        let t = t1d(&[1.0, 2.0, 3.0]);
        let r = t.repeat_interleave_n(2, 0).unwrap();
        assert_eq!(r.dims(), &[6]);
        assert_eq!(flat_f32(&r), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_repeat_interleave_n_2d_dim0() {
        // [[1,2],[3,4]] repeat_interleave_n(2, dim=0) → [[1,2],[1,2],[3,4],[3,4]]
        let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let r = t.repeat_interleave_n(2, 0).unwrap();
        assert_eq!(r.dims(), &[4, 2]);
        assert_eq!(flat_f32(&r), vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_repeat_interleave_n_2d_dim1() {
        // [[1,2],[3,4]] repeat_interleave_n(3, dim=1) → [[1,1,1,2,2,2],[3,3,3,4,4,4]]
        let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let r = t.repeat_interleave_n(3, 1).unwrap();
        assert_eq!(r.dims(), &[2, 6]);
        assert_eq!(
            flat_f32(&r),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
        );
    }

    #[test]
    fn test_repeat_interleave_n_repeats_1_is_identity() {
        let t = t1d(&[5.0, 6.0, 7.0]);
        let r = t.repeat_interleave_n(1, 0).unwrap();
        assert_eq!(r.dims(), t.dims());
        assert_eq!(flat_f32(&r), flat_f32(&t));
    }

    #[test]
    fn test_repeat_interleave_n_repeats_0_empty() {
        let t = t1d(&[1.0, 2.0, 3.0]);
        let r = t.repeat_interleave_n(0, 0).unwrap();
        assert_eq!(r.dims(), &[0]);
        assert_eq!(r.numel(), 0);
    }

    #[test]
    fn test_repeat_interleave_n_dim_oob() {
        let t = t1d(&[1.0, 2.0]);
        assert!(t.repeat_interleave_n(2, 1).is_err());
    }

    #[test]
    fn test_repeat_interleave_n_3d() {
        // [1,2,3] tensor, repeat along dim 1
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let t = tnd(&data, &[1, 2, 3]);
        let r = t.repeat_interleave_n(2, 1).unwrap();
        assert_eq!(r.dims(), &[1, 4, 3]);
        // dim=1 repeats each row: row0, row0, row1, row1
        let vals = flat_f32(&r);
        assert_eq!(
            vals,
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]
        );
    }

    // ---- tile_numpy ---------------------------------------------------------

    #[test]
    fn test_tile_numpy_exact_rank() {
        // Same as tile when reps.len() == rank
        let t = t1d(&[1.0, 2.0]);
        let r = t.tile_numpy(&[3]).unwrap();
        assert_eq!(r.dims(), &[6]);
        assert_eq!(flat_f32(&r), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tile_numpy_shorter_reps_2d() {
        // [2,2] tensor, reps=[3] → padded to [1,3] → tile dim1 only
        let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let r = t.tile_numpy(&[3]).unwrap();
        assert_eq!(r.dims(), &[2, 6]);
        assert_eq!(
            flat_f32(&r),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_tile_numpy_shorter_reps_3d() {
        // [1,2,3] tensor, reps=[2] → padded to [1,1,2] → tile dim2 only
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let t = tnd(&data, &[1, 2, 3]);
        let r = t.tile_numpy(&[2]).unwrap();
        assert_eq!(r.dims(), &[1, 2, 6]);
        let vals = flat_f32(&r);
        assert_eq!(
            vals,
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn test_tile_numpy_empty_reps_is_identity() {
        // reps=[] → padded to [1,1,...,1] → identity
        let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let r = t.tile_numpy(&[]).unwrap();
        assert_eq!(r.dims(), &[2, 2]);
        assert_eq!(flat_f32(&r), flat_f32(&t));
    }

    #[test]
    fn test_tile_numpy_reps_too_long() {
        let t = t1d(&[1.0, 2.0]);
        assert!(t.tile_numpy(&[2, 3]).is_err());
    }

    #[test]
    fn test_tile_numpy_all_ones_is_identity() {
        let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let r = t.tile_numpy(&[1, 1]).unwrap();
        assert_eq!(r.dims(), &[2, 2]);
        assert_eq!(flat_f32(&r), flat_f32(&t));
    }

    #[test]
    fn test_tile_numpy_2d_both_dims() {
        // [2,3] tensor, reps=[2,2]
        let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let r = t.tile_numpy(&[2, 2]).unwrap();
        assert_eq!(r.dims(), &[4, 6]);
        let vals = flat_f32(&r);
        // Row 0: [1,2,3,1,2,3], Row 1: [4,5,6,4,5,6], Row 2: [1,2,3,1,2,3], Row 3: [4,5,6,4,5,6]
        assert_eq!(
            vals,
            vec![
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 1.0,
                2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
            ]
        );
    }
}
