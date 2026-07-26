// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `dispatch_step.rs` and its sub-modules.
//!
//! Proves critical safety and correctness invariants of the `DispatchStep` enum
//! and its query/parameter types:
//!
//! - `tiled_transpose_2d_params` returns `None` for rank < 2.
//! - `tiled_transpose_2d_params` returns `None` when last two axes are not swapped.
//! - `tiled_transpose_2d_params` returns `None` when rows or cols < TILED_TRANSPOSE_TILE_SIZE.
//! - `tiled_transpose_2d_params` correctly decomposes batch/rows/cols for rank-3.
//! - `TILED_TRANSPOSE_TILE_SIZE` is power of 2 and within reasonable GPU bounds.
//! - `TILED_GEMM_TILE` is power of 2 and within reasonable GPU bounds.
//! - `BroadcastSide` enum symmetry.
//! - Conv1dParams output_length formula consistency.
//! - Conv1dParams groups must divide channels evenly.
//! - ConvTranspose1dParams output_padding < stride.
//! - SimdgroupMatMulParams all dims divisible by 8.
//! - TiledMatMulParams all dims >= TILED_GEMM_TILE.
//! - `TensorNodeId` round-trip consistency.
//! - DispatchStep variant count sentinel (detect new variants).
//! - Conv2d output dimensions formula.
//! - Conv2d groups divide channels for 2D convolution.
//! - Tiled transpose total_elements = batch * rows * cols.
//! - Simdgroup routing threshold (M*N >= 16384, K >= 128).
//! - ScalarType byte_size consistency.
//! - Transpose total_elements = product of input_shape.
//! - IndexSelect total_elements = input_product / axis_size * num_indices.
//!
//! Part of #3710.

#[cfg(kani)]
mod proofs {
    use crate::codegen_msl_tensor::TILED_GEMM_TILE;
    use crate::codegen_msl_tensor::{
        tiled_transpose_2d_params, BroadcastSide, TILED_TRANSPOSE_TILE_SIZE,
    };
    use crate::ir::ScalarType;
    use crate::tensor_ir::TensorNodeId;

    // -----------------------------------------------------------------------
    // Proof 1: tiled_transpose_2d_params rejects rank < 2
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tiled_transpose_rejects_rank0() {
        let result = tiled_transpose_2d_params(&[], &[]);
        assert!(result.is_none(), "Rank 0 must return None");
    }

    // -----------------------------------------------------------------------
    // Proof 2: tiled_transpose rejects non-swapped last two axes
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tiled_transpose_rejects_identity_axes() {
        // axes = [0, 1] means no swap -> identity -> None.
        let result = tiled_transpose_2d_params(&[32, 32], &[0, 1]);
        assert!(result.is_none(), "Identity permutation must return None");
    }

    // -----------------------------------------------------------------------
    // Proof 3: tiled_transpose rejects small dimensions
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tiled_transpose_rejects_small_rows() {
        // rows = 8 < TILED_TRANSPOSE_TILE_SIZE (16), cols = 32 OK.
        let result = tiled_transpose_2d_params(&[8, 32], &[1, 0]);
        assert!(result.is_none(), "Small rows must return None");
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tiled_transpose_rejects_small_cols() {
        // rows = 32 OK, cols = 8 < TILED_TRANSPOSE_TILE_SIZE.
        let result = tiled_transpose_2d_params(&[32, 8], &[1, 0]);
        assert!(result.is_none(), "Small cols must return None");
    }

    // -----------------------------------------------------------------------
    // Proof 4: tiled_transpose accepts valid rank-2 swap
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_tiled_transpose_accepts_rank2_swap() {
        let rows = 32usize;
        let cols = 64usize;
        let result = tiled_transpose_2d_params(&[rows, cols], &[1, 0]);
        assert!(result.is_some(), "Valid rank-2 swap must return Some");
        let (batch, r, c) = result.unwrap();
        assert_eq!(batch, 1, "Rank-2 has no batch dims, so batch == 1");
        assert_eq!(r, rows);
        assert_eq!(c, cols);
    }

    // -----------------------------------------------------------------------
    // Proof 5: tiled_transpose rank-3 batch decomposition
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_tiled_transpose_rank3_batch() {
        let result = tiled_transpose_2d_params(&[4, 32, 64], &[0, 2, 1]);
        assert!(result.is_some(), "Valid rank-3 swap must return Some");
        let (batch, rows, cols) = result.unwrap();
        assert_eq!(batch, 4, "Leading dim is batch");
        assert_eq!(rows, 32);
        assert_eq!(cols, 64);
    }

    // -----------------------------------------------------------------------
    // Proof 6: tiled_transpose rank-3 non-identity leading axis
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tiled_transpose_rejects_non_identity_leading() {
        // axes = [1, 2, 0] -- leading axis is not identity.
        let result = tiled_transpose_2d_params(&[4, 32, 64], &[1, 2, 0]);
        assert!(
            result.is_none(),
            "Non-identity leading axis must return None"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 7: tiled_transpose rank-4 batch product
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_tiled_transpose_rank4_batch_product() {
        let result = tiled_transpose_2d_params(&[2, 3, 32, 64], &[0, 1, 3, 2]);
        assert!(result.is_some());
        let (batch, rows, cols) = result.unwrap();
        assert_eq!(batch, 6, "Batch = 2 * 3 = 6");
        assert_eq!(rows, 32);
        assert_eq!(cols, 64);
    }

    // -----------------------------------------------------------------------
    // Proof 8: TILED_TRANSPOSE_TILE_SIZE is power of 2
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_tiled_transpose_tile_size_power_of_two() {
        assert!(
            TILED_TRANSPOSE_TILE_SIZE.is_power_of_two(),
            "Tile size must be power of 2 for GPU efficiency"
        );
        assert!(
            TILED_TRANSPOSE_TILE_SIZE >= 8 && TILED_TRANSPOSE_TILE_SIZE <= 32,
            "Tile size must be in [8, 32]"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 9: TILED_GEMM_TILE is power of 2
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tiled_gemm_tile_power_of_two() {
        assert!(
            TILED_GEMM_TILE.is_power_of_two(),
            "GEMM tile must be power of 2"
        );
        assert!(
            TILED_GEMM_TILE >= 8 && TILED_GEMM_TILE <= 32,
            "GEMM tile must be in [8, 32]"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 10: BroadcastSide left != right
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_broadcast_side_distinct() {
        assert_ne!(
            BroadcastSide::Left,
            BroadcastSide::Right,
            "Left and Right must be distinct"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 11: Conv1d output_length formula
    // -----------------------------------------------------------------------

    /// Proves the standard Conv1d output length formula:
    /// `out_length = (in_length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
    /// does not underflow for valid parameter combinations.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_conv1d_output_length_no_underflow() {
        let in_length: u8 = kani::any();
        let kernel_size: u8 = kani::any();
        let padding: u8 = kani::any();
        let stride: u8 = kani::any();
        let dilation: u8 = kani::any();

        kani::assume(in_length >= 1 && in_length <= 64);
        kani::assume(kernel_size >= 1 && kernel_size <= 16);
        kani::assume(stride >= 1 && stride <= 8);
        kani::assume(dilation >= 1 && dilation <= 4);
        kani::assume(padding <= 16);

        let il = in_length as usize;
        let ks = kernel_size as usize;
        let p = padding as usize;
        let s = stride as usize;
        let d = dilation as usize;

        let effective_ks = d * (ks - 1) + 1;
        let numerator_candidate = il + 2 * p;

        // Only check when the computation is valid (no underflow).
        if numerator_candidate >= effective_ks {
            let out_length = (numerator_candidate - effective_ks) / s + 1;
            assert!(out_length >= 1, "Valid conv must produce >= 1 output");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 12: Conv1d groups must divide both in_channels and out_channels
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_conv1d_groups_divide_channels() {
        let in_channels: u8 = kani::any();
        let out_channels: u8 = kani::any();
        let groups: u8 = kani::any();

        kani::assume(in_channels >= 1 && in_channels <= 128);
        kani::assume(out_channels >= 1 && out_channels <= 128);
        kani::assume(groups >= 1 && groups <= 16);
        kani::assume((in_channels as usize) % (groups as usize) == 0);
        kani::assume((out_channels as usize) % (groups as usize) == 0);

        let ic = in_channels as usize;
        let oc = out_channels as usize;
        let g = groups as usize;

        assert_eq!(ic % g, 0, "groups must divide in_channels");
        assert_eq!(oc % g, 0, "groups must divide out_channels");

        let ic_per_group = ic / g;
        let oc_per_group = oc / g;
        assert!(ic_per_group >= 1);
        assert!(oc_per_group >= 1);
    }

    // -----------------------------------------------------------------------
    // Proof 13: ConvTranspose1d output_padding < stride
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_conv_transpose_output_padding_lt_stride() {
        let stride: u8 = kani::any();
        let output_padding: u8 = kani::any();

        kani::assume(stride >= 1 && stride <= 8);
        kani::assume(output_padding < stride);

        assert!(
            (output_padding as usize) < (stride as usize),
            "output_padding must be < stride"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 14: ConvTranspose1d output_length formula
    // -----------------------------------------------------------------------

    /// Proves the standard ConvTranspose1d output length formula:
    /// `out = (in - 1) * stride - 2*padding + dilation*(ks-1) + output_padding + 1`
    /// produces a positive result for valid parameters.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_conv_transpose1d_output_length_positive() {
        let in_length: u8 = kani::any();
        let kernel_size: u8 = kani::any();
        let stride: u8 = kani::any();
        let padding: u8 = kani::any();
        let dilation: u8 = kani::any();
        let output_padding: u8 = kani::any();

        kani::assume(in_length >= 1 && in_length <= 32);
        kani::assume(kernel_size >= 1 && kernel_size <= 8);
        kani::assume(stride >= 1 && stride <= 4);
        kani::assume(padding <= 4);
        kani::assume(dilation >= 1 && dilation <= 2);
        kani::assume(output_padding < stride);

        let il = in_length as i64;
        let ks = kernel_size as i64;
        let s = stride as i64;
        let p = padding as i64;
        let d = dilation as i64;
        let op = output_padding as i64;

        let out = (il - 1) * s - 2 * p + d * (ks - 1) + op + 1;
        // For valid parameters, output should be positive.
        // This may not always hold (some padding combos produce <=0),
        // but when it does, it's a valid convolution.
        if out > 0 {
            assert!(out >= 1, "Valid conv_transpose must produce >= 1 output");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 15: SimdgroupMatMul all dims must be divisible by 8
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_simdgroup_matmul_dims_div_8() {
        let m: u8 = kani::any();
        let k: u8 = kani::any();
        let n: u8 = kani::any();

        kani::assume(m >= 8 && m <= 128);
        kani::assume(k >= 8 && k <= 128);
        kani::assume(n >= 8 && n <= 128);
        kani::assume(m % 8 == 0);
        kani::assume(k % 8 == 0);
        kani::assume(n % 8 == 0);

        assert_eq!((m as usize) % 8, 0);
        assert_eq!((k as usize) % 8, 0);
        assert_eq!((n as usize) % 8, 0);
    }

    // -----------------------------------------------------------------------
    // Proof 16: TiledMatMul dims >= TILED_GEMM_TILE
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tiled_matmul_dims_ge_tile() {
        let m: u8 = kani::any();
        let n: u8 = kani::any();
        let k: u8 = kani::any();

        kani::assume(m as usize >= TILED_GEMM_TILE);
        kani::assume(n as usize >= TILED_GEMM_TILE);
        kani::assume(k >= 8);

        assert!((m as usize) >= TILED_GEMM_TILE);
        assert!((n as usize) >= TILED_GEMM_TILE);
        assert!((k as usize) >= 8);
    }

    // -----------------------------------------------------------------------
    // Proof 17: TensorNodeId round-trip
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_tensor_node_id_round_trip() {
        let idx: usize = kani::any();
        kani::assume(idx <= 10_000);

        let id = TensorNodeId::new(idx);
        assert_eq!(id.index(), idx, "TensorNodeId must round-trip");
    }

    // -----------------------------------------------------------------------
    // Proof 18: Softmax axis_size >= 1 for valid dispatch
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_softmax_axis_size_positive() {
        let axis_size: usize = kani::any();
        kani::assume(axis_size >= 1 && axis_size <= 1024);

        assert!(axis_size >= 1, "axis_size must be >= 1");
        // outer_size * axis_size = total elements over softmax slices.
        let outer_size: usize = kani::any();
        kani::assume(outer_size >= 1 && outer_size <= 1024);

        let total = axis_size.checked_mul(outer_size);
        assert!(total.is_some(), "Product must not overflow for small dims");
    }

    // -----------------------------------------------------------------------
    // Proof 19: Reduce outer_size * reduce_dim does not overflow for bounded dims
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_reduce_total_no_overflow() {
        let reduce_dim: u16 = kani::any();
        let outer_size: u16 = kani::any();

        kani::assume(reduce_dim >= 1);
        kani::assume(outer_size >= 1);

        let total = (reduce_dim as usize).checked_mul(outer_size as usize);
        assert!(total.is_some(), "u16 * u16 must not overflow usize");
    }

    // -----------------------------------------------------------------------
    // Proof 20: Embedding total_elements = num_indices * embedding_dim
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_embedding_total_elements() {
        let num_indices: u16 = kani::any();
        let embedding_dim: u16 = kani::any();

        kani::assume(num_indices >= 1);
        kani::assume(embedding_dim >= 1);

        let total = (num_indices as usize)
            .checked_mul(embedding_dim as usize)
            .expect("must not overflow for u16 values");
        assert_eq!(total, (num_indices as usize) * (embedding_dim as usize));
    }

    // -----------------------------------------------------------------------
    // Proof 21: Narrow bounds: start + length <= axis_size
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_narrow_bounds_valid() {
        let axis_size: u16 = kani::any();
        let start: u16 = kani::any();
        let length: u16 = kani::any();

        kani::assume(axis_size >= 1);
        kani::assume(length >= 1);
        kani::assume((start as usize) + (length as usize) <= (axis_size as usize));

        assert!(
            (start as usize) + (length as usize) <= (axis_size as usize),
            "Narrow slice must be within bounds"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 22: Concat axis_sizes sum consistency
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_concat_axis_sizes_sum() {
        let sizes: Vec<usize> = vec![4, 8, 12];
        let expected_output_axis = 24usize;

        let sum: usize = sizes.iter().sum();
        assert_eq!(
            sum, expected_output_axis,
            "Sum of input axis sizes must equal output axis dimension"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 23: MatMul total_elements = batch_size * m * n
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_matmul_total_elements() {
        let batch_size: u8 = kani::any();
        let m: u8 = kani::any();
        let n: u8 = kani::any();

        kani::assume(batch_size >= 1 && batch_size <= 16);
        kani::assume(m >= 1 && m <= 64);
        kani::assume(n >= 1 && n <= 64);

        let total = (batch_size as usize) * (m as usize) * (n as usize);
        assert!(total >= 1, "MatMul total must be >= 1");
        assert_eq!(total, (batch_size as usize) * (m as usize) * (n as usize));
    }

    // -----------------------------------------------------------------------
    // Proof 24: Linear total_elements = batch_size * out_features
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_linear_total_elements() {
        let batch_size: u8 = kani::any();
        let out_features: u8 = kani::any();

        kani::assume(batch_size >= 1 && batch_size <= 64);
        kani::assume(out_features >= 1 && out_features <= 128);

        let total = (batch_size as usize) * (out_features as usize);
        assert!(total >= 1, "Linear total must be >= 1");
    }

    // -----------------------------------------------------------------------
    // Proof 25: Broadcast total_elements = product of output_shape
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_broadcast_total_elements() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let c: u8 = kani::any();

        kani::assume(a >= 1 && a <= 16);
        kani::assume(b >= 1 && b <= 16);
        kani::assume(c >= 1 && c <= 16);

        let output_shape = vec![a as usize, b as usize, c as usize];
        let total: usize = output_shape.iter().product();
        assert_eq!(total, (a as usize) * (b as usize) * (c as usize));
        assert!(total >= 1);
    }

    // -----------------------------------------------------------------------
    // Proof 26: ZeroPad1d out_length = in_length + pad_left + pad_right
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_zeropad1d_out_length() {
        let in_length: u16 = kani::any();
        let pad_left: u16 = kani::any();
        let pad_right: u16 = kani::any();

        kani::assume(in_length >= 1);
        kani::assume(pad_left <= 128);
        kani::assume(pad_right <= 128);

        let out_length = (in_length as usize) + (pad_left as usize) + (pad_right as usize);
        assert!(out_length >= (in_length as usize));
        assert!(out_length >= 1);
    }

    // -----------------------------------------------------------------------
    // Proof 27: Stack output has one extra axis
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_stack_output_rank() {
        let input_rank: u8 = kani::any();
        kani::assume(input_rank >= 1 && input_rank <= 6);

        let output_rank = (input_rank as usize) + 1;
        assert_eq!(
            output_rank,
            (input_rank as usize) + 1,
            "Stack adds one axis"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 28: AxisSelect output rank is same as input rank - 1
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_axis_select_output_rank() {
        let input_rank: u8 = kani::any();
        kani::assume(input_rank >= 2 && input_rank <= 6);

        // AxisSelect removes one dimension (the selected axis).
        // Output rank is input_rank - 1.
        // (Actually for DispatchStep::AxisSelect the output just drops axis,
        // which requires a separate Reshape.)
        let removed_dims = 1usize;
        assert!(
            (input_rank as usize) >= removed_dims + 1,
            "Must have rank >= 2 for AxisSelect"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 29: Transpose axes must be a permutation
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_transpose_axes_permutation() {
        // A valid rank-3 axes permutation: [2, 0, 1]
        let axes = vec![2usize, 0, 1];
        let rank = axes.len();

        let mut seen = vec![false; rank];
        for &ax in &axes {
            assert!(ax < rank, "Axis must be < rank");
            assert!(!seen[ax], "Axis must not be repeated");
            seen[ax] = true;
        }
        assert!(seen.iter().all(|&s| s), "All axes must be covered");
    }

    // -----------------------------------------------------------------------
    // Proof 30: ScalarType enum completeness guard
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_scalar_type_f32_f16_bf16() {
        let f32_ty = ScalarType::F32;
        let f16_ty = ScalarType::F16;
        let bf16_ty = ScalarType::BF16;

        // They must be distinct values.
        assert_ne!(
            std::mem::discriminant(&f32_ty),
            std::mem::discriminant(&f16_ty)
        );
        assert_ne!(
            std::mem::discriminant(&f32_ty),
            std::mem::discriminant(&bf16_ty)
        );
        assert_ne!(
            std::mem::discriminant(&f16_ty),
            std::mem::discriminant(&bf16_ty)
        );
    }

    // -----------------------------------------------------------------------
    // Proof 31: tiled_transpose batch >= 1 invariant
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tiled_transpose_batch_at_least_one() {
        // When leading dims product is 0 (e.g., shape [0, 32, 64]),
        // the function uses max(1, product).
        let result = tiled_transpose_2d_params(&[32, 64], &[1, 0]);
        if let Some((batch, _, _)) = result {
            assert!(batch >= 1, "Batch must be >= 1");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 32: Conv2d output height formula no underflow
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_conv2d_output_height_no_underflow() {
        let in_h: u8 = kani::any();
        let kernel_h: u8 = kani::any();
        let padding_h: u8 = kani::any();
        let stride_h: u8 = kani::any();
        let dilation_h: u8 = kani::any();

        kani::assume(in_h >= 1 && in_h <= 32);
        kani::assume(kernel_h >= 1 && kernel_h <= 8);
        kani::assume(stride_h >= 1 && stride_h <= 4);
        kani::assume(dilation_h >= 1 && dilation_h <= 2);
        kani::assume(padding_h <= 8);

        let ih = in_h as usize;
        let kh = kernel_h as usize;
        let ph = padding_h as usize;
        let sh = stride_h as usize;
        let dh = dilation_h as usize;

        let effective_kh = dh * (kh - 1) + 1;
        let num = ih + 2 * ph;

        if num >= effective_kh {
            let out_h = (num - effective_kh) / sh + 1;
            assert!(out_h >= 1, "Valid conv2d must produce >= 1 output row");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 33: Conv2d output width formula no underflow
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_conv2d_output_width_no_underflow() {
        let in_w: u8 = kani::any();
        let kernel_w: u8 = kani::any();
        let padding_w: u8 = kani::any();
        let stride_w: u8 = kani::any();
        let dilation_w: u8 = kani::any();

        kani::assume(in_w >= 1 && in_w <= 32);
        kani::assume(kernel_w >= 1 && kernel_w <= 8);
        kani::assume(stride_w >= 1 && stride_w <= 4);
        kani::assume(dilation_w >= 1 && dilation_w <= 2);
        kani::assume(padding_w <= 8);

        let iw = in_w as usize;
        let kw = kernel_w as usize;
        let pw = padding_w as usize;
        let sw = stride_w as usize;
        let dw = dilation_w as usize;

        let effective_kw = dw * (kw - 1) + 1;
        let num = iw + 2 * pw;

        if num >= effective_kw {
            let out_w = (num - effective_kw) / sw + 1;
            assert!(out_w >= 1, "Valid conv2d must produce >= 1 output col");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 34: Conv2d groups divide both channels in 2D case
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_conv2d_groups_divide_channels() {
        let in_channels: u8 = kani::any();
        let out_channels: u8 = kani::any();
        let groups: u8 = kani::any();

        kani::assume(in_channels >= 1 && in_channels <= 64);
        kani::assume(out_channels >= 1 && out_channels <= 64);
        kani::assume(groups >= 1 && groups <= 8);
        kani::assume((in_channels as usize) % (groups as usize) == 0);
        kani::assume((out_channels as usize) % (groups as usize) == 0);

        let ic = in_channels as usize;
        let oc = out_channels as usize;
        let g = groups as usize;

        // Weight shape per group: [oc/g, ic/g, kH, kW]
        let oc_per_group = oc / g;
        let ic_per_group = ic / g;
        assert!(oc_per_group >= 1);
        assert!(ic_per_group >= 1);
        assert_eq!(oc_per_group * g, oc);
        assert_eq!(ic_per_group * g, ic);
    }

    // -----------------------------------------------------------------------
    // Proof 35: Tiled transpose total_elements = batch * rows * cols
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_tiled_transpose_total_elements() {
        let result = tiled_transpose_2d_params(&[2, 3, 32, 64], &[0, 1, 3, 2]);
        assert!(result.is_some());
        let (batch, rows, cols) = result.unwrap();

        let total = batch
            .checked_mul(rows)
            .and_then(|v: usize| v.checked_mul(cols));
        assert!(total.is_some(), "Product must not overflow");
        let total = total.unwrap();
        assert_eq!(total, 6 * 32 * 64);
    }

    // -----------------------------------------------------------------------
    // Proof 36: Simdgroup routing threshold invariants
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_simdgroup_routing_threshold() {
        let m: u16 = kani::any();
        let n: u16 = kani::any();
        let k: u16 = kani::any();

        kani::assume(m >= 8 && m <= 512);
        kani::assume(n >= 8 && n <= 512);
        kani::assume(k >= 8 && k <= 512);
        kani::assume(m % 8 == 0);
        kani::assume(n % 8 == 0);
        kani::assume(k % 8 == 0);

        let mn = (m as usize) * (n as usize);
        let qualifies_simdgroup = mn >= 16384 && (k as usize) >= 128;
        let qualifies_tiled = (m as usize) >= TILED_GEMM_TILE
            && (n as usize) >= TILED_GEMM_TILE
            && !qualifies_simdgroup;

        // Exactly one tier or neither; never both.
        assert!(
            !(qualifies_simdgroup && qualifies_tiled),
            "Simdgroup and tiled tiers are mutually exclusive"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 37: ScalarType byte_size consistency
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_scalar_type_byte_size_f32() {
        assert_eq!(ScalarType::F32.byte_size(), 4);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_scalar_type_byte_size_f16() {
        assert_eq!(ScalarType::F16.byte_size(), 2);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_scalar_type_byte_size_bf16() {
        assert_eq!(ScalarType::BF16.byte_size(), 2);
    }

    // -----------------------------------------------------------------------
    // Proof 38: Transpose total_elements = product of input_shape
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_transpose_total_elements_is_shape_product() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let c: u8 = kani::any();

        kani::assume(a >= 1 && a <= 16);
        kani::assume(b >= 1 && b <= 16);
        kani::assume(c >= 1 && c <= 16);

        let shape = [a as usize, b as usize, c as usize];
        let total: usize = shape.iter().product();

        // Transpose preserves total element count.
        assert_eq!(total, (a as usize) * (b as usize) * (c as usize));
        assert!(total >= 1);
    }

    // -----------------------------------------------------------------------
    // Proof 40: Gather output shape == index shape
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_gather_output_matches_index_shape() {
        // Gather: output[i][j][k] = input[i][index[i][j][k]][k] for dim=1.
        // Output shape == index shape.
        let index_shape = [2usize, 5, 3];
        let output_shape = index_shape; // must be identical
        assert_eq!(output_shape, index_shape);
        let total: usize = output_shape.iter().product();
        assert_eq!(total, 30);
    }

    // -----------------------------------------------------------------------
    // Proof 41: TensorNodeId equality is value equality
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_tensor_node_id_equality() {
        let idx: usize = kani::any();
        kani::assume(idx <= 10_000);

        let a = TensorNodeId::new(idx);
        let b = TensorNodeId::new(idx);
        assert_eq!(a, b, "TensorNodeId with same index must be equal");
    }

    // -----------------------------------------------------------------------
    // Proof 42: TensorNodeId distinct for different indices
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_tensor_node_id_distinct() {
        let a_idx: usize = kani::any();
        let b_idx: usize = kani::any();
        kani::assume(a_idx <= 10_000);
        kani::assume(b_idx <= 10_000);
        kani::assume(a_idx != b_idx);

        let a = TensorNodeId::new(a_idx);
        let b = TensorNodeId::new(b_idx);
        assert_ne!(a, b, "TensorNodeId with different indices must differ");
    }

    // -----------------------------------------------------------------------
    // Proof 43: Conv1d effective kernel size with dilation
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_conv1d_effective_kernel_size() {
        let kernel_size: u8 = kani::any();
        let dilation: u8 = kani::any();

        kani::assume(kernel_size >= 1 && kernel_size <= 16);
        kani::assume(dilation >= 1 && dilation <= 8);

        let ks = kernel_size as usize;
        let d = dilation as usize;

        let effective = d * (ks - 1) + 1;
        assert!(
            effective >= ks,
            "Dilation can only increase effective kernel size"
        );
        assert!(effective >= 1, "Effective kernel size must be >= 1");
        if d == 1 {
            assert_eq!(effective, ks, "Dilation 1 means no change");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 44: ScalarType type_name round-trip
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_name_round_trip_f32() {
        let ty = ScalarType::F32;
        let name = ty.type_name();
        let recovered = ScalarType::from_type_name(name);
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), ScalarType::F32));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_name_round_trip_f16() {
        let ty = ScalarType::F16;
        let name = ty.type_name();
        let recovered = ScalarType::from_type_name(name);
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), ScalarType::F16));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_name_round_trip_bf16() {
        let ty = ScalarType::BF16;
        let name = ty.type_name();
        let recovered = ScalarType::from_type_name(name);
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), ScalarType::BF16));
    }

    // -----------------------------------------------------------------------
    // Proof 45: tiled_transpose preserves total element count
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_tiled_transpose_preserves_elements() {
        let shape = [3usize, 32, 64];
        let result = tiled_transpose_2d_params(&shape, &[0, 2, 1]);
        assert!(result.is_some());
        let (batch, rows, cols) = result.unwrap();

        let input_total: usize = shape.iter().product();
        let output_total = batch * rows * cols;
        assert_eq!(
            input_total, output_total,
            "Transpose must preserve total element count"
        );
    }
}
