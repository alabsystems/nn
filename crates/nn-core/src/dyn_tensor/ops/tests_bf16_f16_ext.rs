#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended bf16/f16 regression tests: shape ops, constructors, KV cache,
//! conv, cat, cumsum, scatter_add, linear, embedding, to_dtype, clamp.
//!
//! Extracted from `tests_bf16_f16.rs` (#1669) to keep files under 500 lines.
//! Core arithmetic/reduction/matmul/unary tests remain in `tests_bf16_f16.rs`.
//! Normalization + softmax tests extracted to `tests_bf16_f16_ext_softmax.rs`.

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DType, DynTensor};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

// -- BF16/F16 normalization + softmax tests ------------------------------------

#[path = "tests_bf16_f16_ext_softmax.rs"]
mod bf16_f16_ext_softmax;

// -- Helpers (duplicated from parent for test module independence) -------------

fn bf16_tensor(data: &[f32], dims: &[usize]) -> DynTensor {
    let arr = ArrayD::from_shape_vec(
        IxDyn(dims),
        data.iter().map(|&v| bf16::from_f32(v)).collect(),
    )
    .unwrap();
    DynTensor::from_cpu_bf16(arr).unwrap()
}

fn f16_tensor(data: &[f32], dims: &[usize]) -> DynTensor {
    let arr = ArrayD::from_shape_vec(
        IxDyn(dims),
        data.iter().map(|&v| f16::from_f32(v)).collect(),
    )
    .unwrap();
    DynTensor::from_cpu_f16(arr).unwrap()
}

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

// -- BF16 shape ops -----------------------------------------------------------

#[test]
fn test_bf16_narrow_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = a.narrow(1, 1, 2).unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    assert_eq!(c.dims(), &[2, 2]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 2.0, 0.1));
    assert!(approx(vals[1], 3.0, 0.1));
    assert!(approx(vals[2], 5.0, 0.1));
    assert!(approx(vals[3], 6.0, 0.1));
}

#[test]
fn test_bf16_reshape_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = a.reshape([3, 2]).unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    assert_eq!(c.dims(), &[3, 2]);
}

#[test]
fn test_bf16_transpose_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = a.t().unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    assert_eq!(c.dims(), &[3, 2]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    // Transposed: [[1,4],[2,5],[3,6]]
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[1], 4.0, 0.1));
    assert!(approx(vals[2], 2.0, 0.1));
}

// -- Constructors: zeros/ones/full with bf16/f16 ------------------------------

#[test]
fn test_bf16_zeros_dtype() {
    let z = DynTensor::zeros(&[2, 3], DType::BF16, &cpu()).unwrap();
    assert_eq!(z.dtype(), DType::BF16);
    assert_eq!(z.dims(), &[2, 3]);
    let vals = z.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| v == 0.0));
}

#[test]
fn test_f16_ones_dtype() {
    let o = DynTensor::ones(&[3], DType::F16, &cpu()).unwrap();
    assert_eq!(o.dtype(), DType::F16);
    let vals = o.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| approx(v, 1.0, 0.001)));
}

#[test]
fn test_bf16_full_dtype() {
    let f = DynTensor::full(&[2, 2], 2.5, DType::BF16, &cpu()).unwrap();
    assert_eq!(f.dtype(), DType::BF16);
    let vals = f.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| approx(v, 2.5, 0.02)));
}

// -- KV cache round-trip: zeros → slice_set → narrow --------------------------

#[test]
fn test_bf16_kv_cache_slice_set_narrow_round_trip() {
    // Simulates KV cache: create buffer, write a slice, read it back.
    let buf = DynTensor::zeros(&[1, 4, 3], DType::BF16, &cpu()).unwrap();
    let chunk = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    // Write chunk at offset 0 along dim 1.
    let buf = buf.slice_set(1, 0, &chunk).unwrap();
    assert_eq!(buf.dtype(), DType::BF16);
    // Read back via narrow.
    let read = buf.narrow(1, 0, 2).unwrap();
    assert_eq!(read.dtype(), DType::BF16);
    assert_eq!(read.dims(), &[1, 2, 3]);
    let vals = read.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[5], 6.0, 0.1));
}

#[test]
fn test_f16_kv_cache_slice_set_narrow_round_trip() {
    let buf = DynTensor::zeros(&[1, 4, 3], DType::F16, &cpu()).unwrap();
    let chunk = f16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let buf = buf.slice_set(1, 0, &chunk).unwrap();
    assert_eq!(buf.dtype(), DType::F16);
    let read = buf.narrow(1, 0, 2).unwrap();
    assert_eq!(read.dtype(), DType::F16);
    let vals = read.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.0, 0.01));
    assert!(approx(vals[5], 6.0, 0.01));
}

// -- BF16 conv1d dtype preservation (#1657 AC3) --------------------------------

#[test]
fn test_bf16_conv1d_preserves_dtype() {
    // Input: [batch=1, channels=1, length=5], kernel: [out_ch=1, in_ch=1, k=3]
    let input = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
    let kernel = bf16_tensor(&[1.0, 1.0, 1.0], &[1, 1, 3]);
    let result = input.conv1d(&kernel, 0, 1, 1, 1).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "conv1d should preserve BF16");
    assert_eq!(result.dims(), &[1, 1, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Sum of 3 consecutive: [6, 9, 12]
    assert!(approx(vals[0], 6.0, 0.1));
    assert!(approx(vals[1], 9.0, 0.1));
    assert!(approx(vals[2], 12.0, 0.1));
}

#[test]
fn test_bf16_conv_transpose1d_preserves_dtype() {
    let input = bf16_tensor(&[1.0, 2.0, 3.0], &[1, 1, 3]);
    let kernel = bf16_tensor(&[1.0, 1.0], &[1, 1, 2]);
    let result = input.conv_transpose1d(&kernel, 0, 0, 1, 1, 1).unwrap();
    assert_eq!(
        result.dtype(),
        DType::BF16,
        "conv_transpose1d should preserve BF16"
    );
    assert_eq!(result.dims(), &[1, 1, 4]);
}

// -- BF16 cat dtype preservation (#1657 AC-extra) ------------------------------

#[test]
fn test_bf16_cat_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let b = bf16_tensor(&[4.0, 5.0, 6.0], &[1, 3]);
    let result = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "cat should preserve BF16");
    assert_eq!(result.dims(), &[2, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[5], 6.0, 0.1));
}

// -- BF16 conv2d dtype preservation -------------------------------------------

#[test]
fn test_bf16_conv2d_preserves_dtype() {
    // [B=1,C=1,H=4,W=4] all-ones, [O=1,C=1,KH=3,KW=3] all-ones → output=9.0
    let input = bf16_tensor(&[1.0; 16], &[1, 1, 4, 4]);
    let kernel = bf16_tensor(&[1.0; 9], &[1, 1, 3, 3]);
    let result = input.conv2d(&kernel, 0, 1, 1, 1).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "conv2d should preserve BF16");
    assert_eq!(result.dims(), &[1, 1, 2, 2]);
    assert!(approx(result.to_flat_vec::<f32>().unwrap()[0], 9.0, 0.1));
}

// -- BF16 cumsum dtype preservation -------------------------------------------

#[test]
fn test_bf16_cumsum_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let result = a.cumsum(0).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "cumsum should preserve BF16");
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[1], 3.0, 0.1));
    assert!(approx(vals[2], 6.0, 0.1));
    assert!(approx(vals[3], 10.0, 0.1));
}

// -- BF16 scatter_add dtype preservation --------------------------------------

#[test]
fn test_bf16_scatter_add_preserves_dtype() {
    let target = bf16_tensor(&[0.0, 0.0, 0.0], &[3]);
    let src = bf16_tensor(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 2], &[3], &cpu()).unwrap();
    let result = target.scatter_add(0, &index, &src).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "scatter_add preserves BF16");
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[1], 2.0, 0.1));
    assert!(approx(vals[2], 3.0, 0.1));
}

// -- BF16 Linear forward dtype preservation -----------------------------------

#[test]
fn test_bf16_linear_forward_preserves_dtype() {
    use crate::layers::{Linear, Module};
    // weight [2, 3], bias [2], input [1, 3] → output [1, 2]
    let weight = bf16_tensor(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    let bias = bf16_tensor(&[0.5, -0.5], &[2]);
    let linear = Linear::new(weight, Some(bias)).unwrap();
    let x = bf16_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let result = linear.forward(&x).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "Linear should preserve BF16");
    assert_eq!(result.dims(), &[1, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // output[0] = dot([1,0,0],[1,2,3]) + 0.5 = 1.5
    // output[1] = dot([0,1,0],[1,2,3]) + (-0.5) = 1.5
    assert!(approx(vals[0], 1.5, 0.1));
    assert!(approx(vals[1], 1.5, 0.1));
}

// -- BF16 Embedding forward dtype preservation --------------------------------

#[test]
fn test_bf16_embedding_forward_preserves_dtype() {
    use crate::layers::{Embedding, Module};
    // Embedding table: 4 tokens, dim=3
    let weight = bf16_tensor(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
    );
    let emb = Embedding::new(weight).unwrap();
    // Look up tokens [0, 2]
    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &cpu()).unwrap();
    let result = emb.forward(&ids).unwrap();
    assert_eq!(
        result.dtype(),
        DType::BF16,
        "Embedding should preserve BF16"
    );
    assert_eq!(result.dims(), &[2, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Token 0: [1, 2, 3]
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[1], 2.0, 0.1));
    assert!(approx(vals[2], 3.0, 0.1));
    // Token 2: [7, 8, 9]
    assert!(approx(vals[3], 7.0, 0.1));
    assert!(approx(vals[4], 8.0, 0.1));
    assert!(approx(vals[5], 9.0, 0.1));
}

// -- BF16 to_dtype round-trip preservation ------------------------------------

#[test]
fn test_bf16_to_f32_to_bf16_round_trip() {
    let a = bf16_tensor(&[1.5, 2.5, 3.5], &[3]);
    assert_eq!(a.dtype(), DType::BF16);
    // bf16 → f32
    let b = a.to_dtype(DType::F32).unwrap();
    assert_eq!(b.dtype(), DType::F32);
    let vals = b.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.5, 0.1));
    // f32 → bf16
    let c = b.to_dtype(DType::BF16).unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    let vals2 = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals2[0], 1.5, 0.1));
}

// -- BF16 clamp dtype preservation --------------------------------------------

#[test]
fn test_bf16_clamp_preserves_dtype() {
    let a = bf16_tensor(&[-1.0, 0.5, 2.0, 3.5], &[4]);
    let result = a.clamp(0.0, 2.5).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "clamp should preserve BF16");
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 0.0, 0.1));
    assert!(approx(vals[1], 0.5, 0.1));
    assert!(approx(vals[2], 2.0, 0.1));
    assert!(approx(vals[3], 2.5, 0.1));
}

// -- BF16 CPU fallback regression tests (#1793) --------------------------------
// These reproduce the exact scenario where GPU ops return None for BF16,
// triggering CPU fallback. The CPU paths must handle BF16 natively.

#[test]
fn test_bf16_index_select_cpu() {
    // index_select is one of the 12 ops that fall back to CPU for BF16 on Metal.
    // Verify the CPU path handles BF16 storage correctly.
    use crate::Device;
    let data = bf16_tensor(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]);
    let indices = DynTensor::from_vec_u32(vec![0, 2, 4], &[3], &Device::Cpu).unwrap();
    let result = data.index_select(&indices, 0).unwrap();
    assert_eq!(
        result.dtype(),
        DType::BF16,
        "index_select should preserve BF16"
    );
    assert_eq!(result.dims(), &[3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 10.0, 0.5));
    assert!(approx(vals[1], 30.0, 0.5));
    assert!(approx(vals[2], 50.0, 0.5));
}

#[test]
fn test_bf16_slice_set_cpu() {
    // slice_set dispatches to slice_set_half for BF16 — verify it works.
    let target = bf16_tensor(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 3]);
    let src = bf16_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let result = target.slice_set(0, 0, &src).unwrap();
    assert_eq!(
        result.dtype(),
        DType::BF16,
        "slice_set should preserve BF16"
    );
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[1], 2.0, 0.1));
    assert!(approx(vals[2], 3.0, 0.1));
    // Second row should remain zeros
    assert!(approx(vals[3], 0.0, 0.1));
}

#[test]
fn test_bf16_gather_cpu() {
    // gather is another CPU-fallback op for BF16 Metal tensors.
    use crate::Device;
    let data = bf16_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    // Gather along dim=1: select column 1 for row 0, column 0 for row 1
    let indices = DynTensor::from_vec_u32(vec![1, 0], &[2, 1], &Device::Cpu).unwrap();
    let result = data.gather(&indices, 1).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "gather should preserve BF16");
    assert_eq!(result.dims(), &[2, 1]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 2.0, 0.1));
    assert!(approx(vals[1], 3.0, 0.1));
}

#[test]
fn test_bf16_where_cond_cpu() {
    // where_cond is a CPU-fallback op for BF16.
    use crate::Device;
    let cond = DynTensor::from_vec_u8(vec![1, 0, 1, 0], &[4], &Device::Cpu).unwrap();
    let on_true = bf16_tensor(&[10.0, 20.0, 30.0, 40.0], &[4]);
    let on_false = bf16_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let result = cond.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(
        result.dtype(),
        DType::BF16,
        "where_cond should preserve BF16"
    );
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 10.0, 0.5));
    assert!(approx(vals[1], 2.0, 0.1));
    assert!(approx(vals[2], 30.0, 0.5));
    assert!(approx(vals[3], 4.0, 0.1));
}

#[test]
fn test_bf16_cumsum_cpu() {
    // cumsum is another CPU-fallback op for BF16.
    let data = bf16_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let result = data.cumsum(0).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "cumsum should preserve BF16");
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[1], 3.0, 0.1));
    assert!(approx(vals[2], 6.0, 0.1));
    assert!(approx(vals[3], 10.0, 0.5));
}

#[test]
fn test_bf16_repeat_interleave_cpu() {
    let data = bf16_tensor(&[1.0, 2.0, 3.0], &[3]);
    // repeat_interleave expects F32 repeats tensor (to_flat_vec_f32 internally)
    let repeats =
        DynTensor::from_cpu_f32(ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 2.0, 2.0]).unwrap())
            .unwrap();
    let result = data.repeat_interleave(0, &repeats).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    assert_eq!(result.dims(), &[6]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[1], 1.0, 0.1));
    assert!(approx(vals[2], 2.0, 0.1));
    assert!(approx(vals[3], 2.0, 0.1));
    assert!(approx(vals[4], 3.0, 0.1));
    assert!(approx(vals[5], 3.0, 0.1));
}
