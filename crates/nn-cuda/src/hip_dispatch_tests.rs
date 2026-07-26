// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `hip_dispatch::launch_config_for_step`.

use super::*;
use nn_dsl::{SimdgroupLinearParams, SimdgroupMatMulParams, TensorNodeId};

#[test]
fn test_dispatcher_not_available_on_macos() {
    if crate::hip_runtime::is_hip_available() {
        return;
    }
    let result = HipDispatcher::new(0, "gfx90a");
    assert!(result.is_err());
}

#[test]
fn test_launch_config_for_reshape_is_none() {
    let step = DispatchStep::Reshape {
        input: TensorNodeId::new(0),
        output: TensorNodeId::new(1),
    };
    assert!(launch_config_for_step(&step).unwrap().is_none());
}

#[test]
fn test_launch_config_for_binary_add() {
    let step = DispatchStep::BinaryAdd {
        kernel_name: "add".into(),
        dtype: ScalarType::F32,
        left: TensorNodeId::new(0),
        right: TensorNodeId::new(1),
        output: TensorNodeId::new(2),
        total_elements: 1024,
        broadcast: None,
    };
    let cfg = launch_config_for_step(&step).unwrap().unwrap();
    assert_eq!(cfg.grid.x, 4); // ceil(1024/256)
    assert_eq!(cfg.block.x, 256);
}

#[test]
fn test_launch_config_for_matmul() {
    let step = DispatchStep::MatMul {
        kernel_name: "matmul".into(),
        dtype: ScalarType::F32,
        left: TensorNodeId::new(0),
        right: TensorNodeId::new(1),
        output: TensorNodeId::new(2),
        m: 128,
        k: 64,
        n: 256,
        batch_size: 1,
        transpose_right: false,
        broadcast_right: false,
        scale: None,
        total_elements: 128 * 256,
    };
    let cfg = launch_config_for_step(&step).unwrap().unwrap();
    assert_eq!(cfg.grid.x, 16); // ceil(256/16)
    assert_eq!(cfg.grid.y, 8); // ceil(128/16)
}

#[test]
fn test_launch_config_for_softmax() {
    let step = DispatchStep::Softmax {
        kernel_name: "softmax".into(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(0),
        output: TensorNodeId::new(1),
        axis: 1,
        axis_size: 64,
        outer_size: 8,
    };
    let cfg = launch_config_for_step(&step).unwrap().unwrap();
    assert!(cfg.shared_mem_bytes > 0);
}

#[test]
fn test_launch_config_for_embedding() {
    let step = DispatchStep::Embedding {
        kernel_name: "embed".into(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(0),
        weight: TensorNodeId::new(1),
        output: TensorNodeId::new(2),
        embedding_dim: 256,
        num_indices: 32,
        total_elements: 32 * 256,
    };
    let cfg = launch_config_for_step(&step).unwrap().unwrap();
    assert_eq!(cfg.grid.x, 32); // ceil(8192/256)
}

#[test]
fn test_launch_config_for_narrow() {
    let step = DispatchStep::Narrow {
        kernel_name: "narrow".into(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(0),
        output: TensorNodeId::new(1),
        input_shape: vec![4, 8, 16],
        axis: 1,
        start: 2,
        length: 3,
    };
    let cfg = launch_config_for_step(&step).unwrap().unwrap();
    // Total = 4 * 3 * 16 = 192
    let expected_grid: u32 = 192_u32.div_ceil(256);
    assert_eq!(cfg.grid.x, expected_grid);
}

#[test]
fn test_launch_config_simdgroup_linear_rocwmma() {
    // M=128, K=128, N=256 — all %16==0, M*N=32768>=16384, K=128>=128.
    let step = DispatchStep::SimdgroupLinear(SimdgroupLinearParams {
        kernel_name: "sg_linear".into(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(0),
        weight: TensorNodeId::new(1),
        bias: None,
        output: TensorNodeId::new(2),
        in_features: 128,
        out_features: 256,
        batch_size: 128,
    });
    let cfg = launch_config_for_step(&step).unwrap().unwrap();
    // rocWMMA: grid = (ceil(256/32), ceil(128/32), 1) = (8, 4, 1)
    assert_eq!(cfg.grid.x, 8);
    assert_eq!(cfg.grid.y, 4);
    assert_eq!(cfg.grid.z, 1);
    assert_eq!(cfg.block.x, 256);
    assert_eq!(cfg.block.y, 1);
}

#[test]
fn test_launch_config_simdgroup_linear_fallback_naive() {
    // K=64 < 128 — rocWMMA not used, falls back to naive matmul grid.
    let step = DispatchStep::SimdgroupLinear(SimdgroupLinearParams {
        kernel_name: "sg_linear_small".into(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(0),
        weight: TensorNodeId::new(1),
        bias: None,
        output: TensorNodeId::new(2),
        in_features: 64,
        out_features: 32,
        batch_size: 32,
    });
    let cfg = launch_config_for_step(&step).unwrap().unwrap();
    // M=32>=16, N=32>=16 => for_matmul(32, 32, 16, 16)
    assert_eq!(cfg.grid.x, 2); // ceil(32/16)
    assert_eq!(cfg.grid.y, 2); // ceil(32/16)
}

#[test]
fn test_launch_config_simdgroup_matmul_rocwmma_batched() {
    // M=256, K=256, N=128, batch=4 — uses rocWMMA with 3D grid.
    let step = DispatchStep::SimdgroupMatMul(SimdgroupMatMulParams {
        kernel_name: "sg_mm".into(),
        dtype: ScalarType::F32,
        left: TensorNodeId::new(0),
        right: TensorNodeId::new(1),
        output: TensorNodeId::new(2),
        m: 256,
        k: 256,
        n: 128,
        batch_size: 4,
        transpose_right: false,
        broadcast_right: false,
        scale: None,
    });
    let cfg = launch_config_for_step(&step).unwrap().unwrap();
    // rocWMMA: grid = (ceil(128/32), ceil(256/32), 4) = (4, 8, 4)
    assert_eq!(cfg.grid.x, 4);
    assert_eq!(cfg.grid.y, 8);
    assert_eq!(cfg.grid.z, 4);
    assert_eq!(cfg.block.x, 256);
}

#[test]
fn test_launch_config_simdgroup_matmul_small_fallback() {
    // M=8 < 16, N=8 < 16 — falls back to elementwise.
    let step = DispatchStep::SimdgroupMatMul(SimdgroupMatMulParams {
        kernel_name: "sg_mm_small".into(),
        dtype: ScalarType::F32,
        left: TensorNodeId::new(0),
        right: TensorNodeId::new(1),
        output: TensorNodeId::new(2),
        m: 8,
        k: 32,
        n: 8,
        batch_size: 1,
        transpose_right: false,
        broadcast_right: false,
        scale: None,
    });
    let cfg = launch_config_for_step(&step).unwrap().unwrap();
    // 8*8*1 = 64, elementwise: grid = ceil(64/256) = 1
    assert_eq!(cfg.grid.x, 1);
    assert_eq!(cfg.block.x, 256);
}
