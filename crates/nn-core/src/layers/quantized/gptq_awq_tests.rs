// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::awq_loader::{load_awq_linear, unpack_awq_qweight, AwqFormat};
use super::gptq_loader::{
    dequantize_gptq, load_gptq_linear, unpack_gptq_qweight, unpack_gptq_qzeros, GptqFormat,
    GptqLinear,
};
use crate::dyn_tensor::DynTensor;
use crate::layers::Module;
use crate::{DType, Device};

/// Pack 8 INT4 values (each 0..15) into a single u32.
fn pack_int4_to_u32(values: &[u32; 8]) -> u32 {
    let mut packed = 0u32;
    for (i, &v) in values.iter().enumerate() {
        packed |= (v & 0xF) << (i as u32 * 4);
    }
    packed
}

// -- GPTQ unpack tests --------------------------------------------------------

#[test]
fn test_gptq_unpack_known_values() {
    // Pack [1, 2, 3, 4, 5, 6, 7, 8] into a single u32
    let values = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let packed = pack_int4_to_u32(&values);

    // Shape: [1, 1] = one packed row, one output feature
    let packed_tensor = DynTensor::from_vec_u32(vec![packed], &[1, 1], &Device::Cpu).unwrap();

    let unpacked = unpack_gptq_qweight(&packed_tensor).unwrap();
    assert_eq!(unpacked.dims(), &[8, 1]);
    assert_eq!(unpacked.dtype(), DType::F32);

    let data = unpacked.to_flat_vec::<f32>().unwrap();
    for (i, &expected) in values.iter().enumerate() {
        assert_eq!(
            data[i], expected as f32,
            "INT4 value at index {i}: expected {expected}, got {}",
            data[i]
        );
    }
}

#[test]
fn test_gptq_unpack_all_zeros() {
    let packed_tensor = DynTensor::from_vec_u32(vec![0u32], &[1, 1], &Device::Cpu).unwrap();

    let unpacked = unpack_gptq_qweight(&packed_tensor).unwrap();
    assert_eq!(unpacked.dims(), &[8, 1]);

    let data = unpacked.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, 0.0, "expected 0.0 at index {i}, got {v}");
    }
}

#[test]
fn test_gptq_unpack_all_max() {
    // 0xFFFFFFFF = all INT4 values are 15
    let packed_tensor = DynTensor::from_vec_u32(vec![0xFFFFFFFF], &[1, 1], &Device::Cpu).unwrap();

    let unpacked = unpack_gptq_qweight(&packed_tensor).unwrap();
    let data = unpacked.to_flat_vec::<f32>().unwrap();

    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, 15.0, "expected 15.0 at index {i}, got {v}");
    }
}

#[test]
fn test_gptq_unpack_multiple_columns() {
    // Shape: [1, 2] = one packed row, two output features
    let col0_vals = [0u32, 1, 2, 3, 4, 5, 6, 7];
    let col1_vals = [8u32, 9, 10, 11, 12, 13, 14, 15];
    let packed0 = pack_int4_to_u32(&col0_vals);
    let packed1 = pack_int4_to_u32(&col1_vals);

    let packed_tensor =
        DynTensor::from_vec_u32(vec![packed0, packed1], &[1, 2], &Device::Cpu).unwrap();

    let unpacked = unpack_gptq_qweight(&packed_tensor).unwrap();
    assert_eq!(unpacked.dims(), &[8, 2]);

    let data = unpacked.to_flat_vec::<f32>().unwrap();
    // Row-major: [row0_col0, row0_col1, row1_col0, row1_col1, ...]
    for i in 0..8 {
        assert_eq!(data[i * 2], col0_vals[i] as f32, "col0 row {i}");
        assert_eq!(data[i * 2 + 1], col1_vals[i] as f32, "col1 row {i}");
    }
}

// -- GPTQ qzeros unpack tests ------------------------------------------------

#[test]
fn test_gptq_unpack_qzeros_known() {
    let zp_vals = [8u32, 8, 8, 8, 8, 8, 8, 8]; // zero-point = 8 for all channels
    let packed_zp = pack_int4_to_u32(&zp_vals);

    // Shape: [1, 1] = one group, out_features/8 = 1
    let packed_tensor = DynTensor::from_vec_u32(vec![packed_zp], &[1, 1], &Device::Cpu).unwrap();

    let unpacked = unpack_gptq_qzeros(&packed_tensor).unwrap();
    assert_eq!(unpacked.dims(), &[1, 8]);

    let data = unpacked.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, 8.0, "zero-point at index {i}: expected 8.0, got {v}");
    }
}

// -- GPTQ dequantize roundtrip ------------------------------------------------

#[test]
fn test_gptq_dequantize_simple() {
    // 8 input features, 8 output features, group_size=8 (1 group)
    // q_weight: all 8 (mid-range INT4)
    // scales: all 0.1
    // zeros: all 8 (so q - zp = 0, output should be 0)
    let q_data = vec![8.0_f32; 8 * 8];
    let q_weight = DynTensor::from_vec(q_data, &[8, 8], &Device::Cpu).unwrap();

    let scale_data = vec![0.1_f32; 8];
    let scales = DynTensor::from_vec(scale_data, &[1, 8], &Device::Cpu).unwrap();

    let zero_data = vec![8.0_f32; 8];
    let zeros = DynTensor::from_vec(zero_data, &[1, 8], &Device::Cpu).unwrap();

    let result = dequantize_gptq(&q_weight, &scales, &zeros, 8).unwrap();
    assert_eq!(result.dims(), &[8, 8]);

    let data = result.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!(v.abs() < 1e-6, "expected ~0.0 when q==zp, got {v}");
    }
}

#[test]
fn test_gptq_dequantize_nonzero() {
    // q=10, zp=8, scale=0.5 → (10-8)*0.5 = 1.0
    let q_data = vec![10.0_f32; 8 * 8];
    let q_weight = DynTensor::from_vec(q_data, &[8, 8], &Device::Cpu).unwrap();

    let scale_data = vec![0.5_f32; 8];
    let scales = DynTensor::from_vec(scale_data, &[1, 8], &Device::Cpu).unwrap();

    let zero_data = vec![8.0_f32; 8];
    let zeros = DynTensor::from_vec(zero_data, &[1, 8], &Device::Cpu).unwrap();

    let result = dequantize_gptq(&q_weight, &scales, &zeros, 8).unwrap();
    let data = result.to_flat_vec::<f32>().unwrap();

    for &v in &data {
        assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
    }
}

// -- GptqLinear forward tests -------------------------------------------------

#[test]
fn test_gptq_linear_forward_shape() {
    let out_features = 16;
    let in_features = 8;

    // Identity-like weight (scaled down to avoid huge values)
    let mut w_data = vec![0.0_f32; out_features * in_features];
    for i in 0..in_features.min(out_features) {
        w_data[i * in_features + i] = 1.0;
    }
    let weight = DynTensor::from_vec(w_data, &[out_features, in_features], &Device::Cpu).unwrap();
    let format = GptqFormat::default();
    let linear = GptqLinear::new(weight, None, format).unwrap();

    // Batch input [2, 8]
    let input = DynTensor::from_vec(vec![1.0_f32; 2 * 8], &[2, 8], &Device::Cpu).unwrap();
    let output = linear.forward(&input).unwrap();

    assert_eq!(output.dims(), &[2, 16]);
}

#[test]
fn test_gptq_linear_forward_accuracy() {
    let out_features = 4;
    let in_features = 4;

    // Simple weight: identity matrix
    let mut w_data = vec![0.0_f32; out_features * in_features];
    for i in 0..4 {
        w_data[i * in_features + i] = 1.0;
    }
    let weight = DynTensor::from_vec(w_data, &[out_features, in_features], &Device::Cpu).unwrap();
    let format = GptqFormat::default();
    let linear = GptqLinear::new(weight, None, format).unwrap();

    let input_data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let input = DynTensor::from_vec(input_data.clone(), &[1, 4], &Device::Cpu).unwrap();
    let output = linear.forward(&input).unwrap();

    let out_data = output.to_flat_vec::<f32>().unwrap();
    for (i, (&expected, &actual)) in input_data.iter().zip(out_data.iter()).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-5,
            "index {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn test_gptq_linear_with_bias() {
    let out_features = 4;
    let in_features = 4;

    let mut w_data = vec![0.0_f32; out_features * in_features];
    for i in 0..4 {
        w_data[i * in_features + i] = 1.0;
    }
    let weight = DynTensor::from_vec(w_data, &[out_features, in_features], &Device::Cpu).unwrap();
    let bias = DynTensor::from_vec(vec![10.0_f32; 4], &[4], &Device::Cpu).unwrap();
    let format = GptqFormat::default();
    let linear = GptqLinear::new(weight, Some(bias), format).unwrap();

    let input = DynTensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let output = linear.forward(&input).unwrap();

    let out_data = output.to_flat_vec::<f32>().unwrap();
    assert!((out_data[0] - 11.0).abs() < 1e-5);
    assert!((out_data[1] - 12.0).abs() < 1e-5);
    assert!((out_data[2] - 13.0).abs() < 1e-5);
    assert!((out_data[3] - 14.0).abs() < 1e-5);
}

#[test]
fn test_gptq_linear_batched_input() {
    let out_features = 4;
    let in_features = 8;

    let w_data = vec![0.1_f32; out_features * in_features];
    let weight = DynTensor::from_vec(w_data, &[out_features, in_features], &Device::Cpu).unwrap();
    let format = GptqFormat::default();
    let linear = GptqLinear::new(weight, None, format).unwrap();

    // 3D input: [batch=2, seq=3, in_features=8]
    let input = DynTensor::from_vec(vec![1.0_f32; 2 * 3 * 8], &[2, 3, 8], &Device::Cpu).unwrap();
    let output = linear.forward(&input).unwrap();

    assert_eq!(output.dims(), &[2, 3, 4]);
}

// -- End-to-end load_gptq_linear tests ----------------------------------------

#[test]
fn test_load_gptq_linear_end_to_end() {
    // 8 in_features, 8 out_features, group_size=8
    // Pack q=10 for all weights, zp=8, scale=0.5 → dequant = (10-8)*0.5 = 1.0
    let q_vals = [10u32, 10, 10, 10, 10, 10, 10, 10];
    let packed_q = pack_int4_to_u32(&q_vals);

    // qweight: [1, 8] (in_features/8=1 packed rows, 8 out_features)
    let qweight = DynTensor::from_vec_u32(vec![packed_q; 8], &[1, 8], &Device::Cpu).unwrap();

    // scales: [1, 8] (1 group, 8 out_features)
    let scales = DynTensor::from_vec(vec![0.5_f32; 8], &[1, 8], &Device::Cpu).unwrap();

    // qzeros: [1, 1] (1 group, 8 out_features / 8 = 1 packed column)
    let zp_vals = [8u32, 8, 8, 8, 8, 8, 8, 8];
    let packed_zp = pack_int4_to_u32(&zp_vals);
    let qzeros = DynTensor::from_vec_u32(vec![packed_zp], &[1, 1], &Device::Cpu).unwrap();

    let linear = load_gptq_linear(&qweight, &scales, &qzeros, None, 8).unwrap();

    assert_eq!(linear.out_features(), 8);
    assert_eq!(linear.in_features(), 8);

    // Forward: all-ones input → each output = sum of 8 weights = 8 * 1.0 = 8.0
    let input = DynTensor::from_vec(vec![1.0_f32; 8], &[1, 8], &Device::Cpu).unwrap();
    let output = linear.forward(&input).unwrap();

    let out_data = output.to_flat_vec::<f32>().unwrap();
    for (i, &v) in out_data.iter().enumerate() {
        assert!((v - 8.0).abs() < 1e-4, "output[{i}]: expected 8.0, got {v}");
    }
}

// -- Group size tests ---------------------------------------------------------

#[test]
fn test_gptq_group_size_32() {
    // 32 in_features, 8 out_features, group_size=32 (1 group)
    let q_vals = [10u32, 10, 10, 10, 10, 10, 10, 10];
    let packed_q = pack_int4_to_u32(&q_vals);

    // qweight: [4, 8] (32/8=4 packed rows, 8 out_features)
    let qweight = DynTensor::from_vec_u32(vec![packed_q; 4 * 8], &[4, 8], &Device::Cpu).unwrap();

    let scales = DynTensor::from_vec(vec![0.5_f32; 8], &[1, 8], &Device::Cpu).unwrap();

    let zp_vals = [8u32, 8, 8, 8, 8, 8, 8, 8];
    let packed_zp = pack_int4_to_u32(&zp_vals);
    let qzeros = DynTensor::from_vec_u32(vec![packed_zp], &[1, 1], &Device::Cpu).unwrap();

    let linear = load_gptq_linear(&qweight, &scales, &qzeros, None, 32).unwrap();
    assert_eq!(linear.out_features(), 8);
    assert_eq!(linear.in_features(), 32);
}

#[test]
fn test_gptq_group_size_128() {
    // 128 in_features, 8 out_features, group_size=128 (1 group)
    let q_vals = [5u32, 5, 5, 5, 5, 5, 5, 5];
    let packed_q = pack_int4_to_u32(&q_vals);

    // qweight: [16, 8] (128/8=16 packed rows, 8 out_features)
    let qweight = DynTensor::from_vec_u32(vec![packed_q; 16 * 8], &[16, 8], &Device::Cpu).unwrap();

    let scales = DynTensor::from_vec(vec![1.0_f32; 8], &[1, 8], &Device::Cpu).unwrap();

    let zp_vals = [0u32; 8];
    let packed_zp = pack_int4_to_u32(&zp_vals);
    let qzeros = DynTensor::from_vec_u32(vec![packed_zp], &[1, 1], &Device::Cpu).unwrap();

    let linear = load_gptq_linear(&qweight, &scales, &qzeros, None, 128).unwrap();
    assert_eq!(linear.in_features(), 128);

    // Forward: all-ones → each output = 128 * (5-0)*1.0 = 640.0
    let input = DynTensor::from_vec(vec![1.0_f32; 128], &[1, 128], &Device::Cpu).unwrap();
    let output = linear.forward(&input).unwrap();

    let out_data = output.to_flat_vec::<f32>().unwrap();
    for (i, &v) in out_data.iter().enumerate() {
        assert!(
            (v - 640.0).abs() < 1e-2,
            "output[{i}]: expected 640.0, got {v}"
        );
    }
}

// -- AWQ tests ----------------------------------------------------------------

#[test]
fn test_awq_unpack_known_values() {
    // AWQ uses the same packing as GPTQ
    let values = [3u32, 7, 11, 15, 0, 4, 8, 12];
    let packed = pack_int4_to_u32(&values);

    let packed_tensor = DynTensor::from_vec_u32(vec![packed], &[1, 1], &Device::Cpu).unwrap();
    let unpacked = unpack_awq_qweight(&packed_tensor).unwrap();

    assert_eq!(unpacked.dims(), &[8, 1]);
    let data = unpacked.to_flat_vec::<f32>().unwrap();
    for (i, &expected) in values.iter().enumerate() {
        assert_eq!(data[i], expected as f32, "AWQ unpack index {i}");
    }
}

#[test]
fn test_awq_linear_forward() {
    // Same setup as GPTQ end-to-end test
    let q_vals = [10u32, 10, 10, 10, 10, 10, 10, 10];
    let packed_q = pack_int4_to_u32(&q_vals);

    let qweight = DynTensor::from_vec_u32(vec![packed_q; 8], &[1, 8], &Device::Cpu).unwrap();
    let scales = DynTensor::from_vec(vec![0.5_f32; 8], &[1, 8], &Device::Cpu).unwrap();

    let zp_vals = [8u32, 8, 8, 8, 8, 8, 8, 8];
    let packed_zp = pack_int4_to_u32(&zp_vals);
    let qzeros = DynTensor::from_vec_u32(vec![packed_zp], &[1, 1], &Device::Cpu).unwrap();

    let linear = load_awq_linear(&qweight, &scales, &qzeros, None, 8).unwrap();

    let input = DynTensor::from_vec(vec![1.0_f32; 8], &[1, 8], &Device::Cpu).unwrap();
    let output = linear.forward(&input).unwrap();

    let out_data = output.to_flat_vec::<f32>().unwrap();
    for (i, &v) in out_data.iter().enumerate() {
        assert!(
            (v - 8.0).abs() < 1e-4,
            "AWQ output[{i}]: expected 8.0, got {v}"
        );
    }
}

#[test]
fn test_awq_format_defaults() {
    let fmt = AwqFormat::default();
    assert_eq!(fmt.group_size, 128);
    assert_eq!(fmt.bits, 4);
}

#[test]
fn test_gptq_format_defaults() {
    let fmt = GptqFormat::default();
    assert_eq!(fmt.group_size, 128);
    assert_eq!(fmt.bits, 4);
    assert!(!fmt.act_order);
}
