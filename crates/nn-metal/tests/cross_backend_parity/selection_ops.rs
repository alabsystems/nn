// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Selection operation parity tests: CPU vs Metal.
//!
//! Tests embedding (index_select), gather, scatter_add, and index_add
//! on both backends.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Embedding, Module};
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-5;

fn init() {
    gpu_init();
}

// -- Embedding (index_select dim=0) ----------------------------------------

#[test]
fn test_parity_embedding() {
    init();
    let vocab_size = 32;
    let embed_dim = 16;
    let seq_len = 8;

    let w_data = rand_f32_vec(900, vocab_size * embed_dim, -1.0, 1.0);
    // Indices: 0..seq_len mapped to valid vocab range
    let ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab_size as u32).collect();

    // CPU
    let w_cpu = DynTensor::new(&w_data, &[vocab_size, embed_dim], &Device::Cpu).unwrap();
    let emb_cpu = Embedding::new(w_cpu).unwrap();
    let ids_cpu = DynTensor::from_vec_u32(ids.clone(), &[seq_len], &Device::Cpu).unwrap();
    let cpu_out = emb_cpu.forward(&ids_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[vocab_size, embed_dim], &Device::metal()).unwrap();
    let emb_gpu = Embedding::new(w_gpu).unwrap();
    let ids_gpu = DynTensor::from_vec_u32(ids, &[seq_len], &Device::metal()).unwrap();
    let gpu_out = emb_gpu.forward(&ids_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[seq_len, embed_dim]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "embedding");
}

// -- Index select (dim=1) --------------------------------------------------

#[test]
fn test_parity_index_select_dim1() {
    init();
    let data = rand_f32_vec(901, 4 * 16, -2.0, 2.0);
    let ids: Vec<u32> = vec![0, 3, 7, 15, 2];

    let x_cpu = DynTensor::new(&data, &[4, 16], &Device::Cpu).unwrap();
    let ids_cpu = DynTensor::from_vec_u32(ids.clone(), &[5], &Device::Cpu).unwrap();
    let cpu_out = x_cpu.index_select(&ids_cpu, 1).unwrap();

    let x_gpu = DynTensor::new(&data, &[4, 16], &Device::metal()).unwrap();
    let ids_gpu = DynTensor::from_vec_u32(ids, &[5], &Device::metal()).unwrap();
    let gpu_out = x_gpu.index_select(&ids_gpu, 1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[4, 5]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "index_select_dim1");
}

// -- Gather ----------------------------------------------------------------

#[test]
fn test_parity_gather() {
    init();
    let data = rand_f32_vec(902, 4 * 8, -3.0, 3.0);
    // Gather indices: same shape as output, values in [0, 8)
    let ids: Vec<u32> = vec![0, 2, 5, 7, 1, 3, 6, 4, 7, 0, 2, 5];

    let x_cpu = DynTensor::new(&data, &[4, 8], &Device::Cpu).unwrap();
    let ids_cpu = DynTensor::from_vec_u32(ids.clone(), &[4, 3], &Device::Cpu).unwrap();
    let cpu_out = x_cpu.gather(&ids_cpu, 1).unwrap();

    let x_gpu = DynTensor::new(&data, &[4, 8], &Device::metal()).unwrap();
    let ids_gpu = DynTensor::from_vec_u32(ids, &[4, 3], &Device::metal()).unwrap();
    let gpu_out = x_gpu.gather(&ids_gpu, 1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[4, 3]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "gather");
}

// -- Scatter add -----------------------------------------------------------

#[test]
fn test_parity_scatter_add() {
    init();
    let dest_data = vec![0.0f32; 4 * 8]; // zero destination
    let src_data = rand_f32_vec(903, 4 * 3, -2.0, 2.0);
    let ids: Vec<u32> = vec![0, 2, 5, 1, 3, 7, 0, 4, 6, 2, 5, 7];

    let dest_cpu = DynTensor::new(&dest_data, &[4, 8], &Device::Cpu).unwrap();
    let src_cpu = DynTensor::new(&src_data, &[4, 3], &Device::Cpu).unwrap();
    let ids_cpu = DynTensor::from_vec_u32(ids.clone(), &[4, 3], &Device::Cpu).unwrap();
    let cpu_out = dest_cpu.scatter_add(1, &ids_cpu, &src_cpu).unwrap();

    let dest_gpu = DynTensor::new(&dest_data, &[4, 8], &Device::metal()).unwrap();
    let src_gpu = DynTensor::new(&src_data, &[4, 3], &Device::metal()).unwrap();
    let ids_gpu = DynTensor::from_vec_u32(ids, &[4, 3], &Device::metal()).unwrap();
    let gpu_out = dest_gpu.scatter_add(1, &ids_gpu, &src_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[4, 8]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "scatter_add");
}

// -- Index add (dim=0) -----------------------------------------------------

#[test]
fn test_parity_index_add() {
    init();
    let dest_data = vec![0.0f32; 8 * 4]; // zero destination
    let src_data = rand_f32_vec(904, 3 * 4, -2.0, 2.0);
    let ids: Vec<u32> = vec![1, 5, 3];

    let dest_cpu = DynTensor::new(&dest_data, &[8, 4], &Device::Cpu).unwrap();
    let src_cpu = DynTensor::new(&src_data, &[3, 4], &Device::Cpu).unwrap();
    let ids_cpu = DynTensor::from_vec_u32(ids.clone(), &[3], &Device::Cpu).unwrap();
    let cpu_out = dest_cpu.index_add(0, &ids_cpu, &src_cpu).unwrap();

    let dest_gpu = DynTensor::new(&dest_data, &[8, 4], &Device::metal()).unwrap();
    let src_gpu = DynTensor::new(&src_data, &[3, 4], &Device::metal()).unwrap();
    let ids_gpu = DynTensor::from_vec_u32(ids, &[3], &Device::metal()).unwrap();
    let gpu_out = dest_gpu.index_add(0, &ids_gpu, &src_gpu).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[8, 4]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "index_add");
}
