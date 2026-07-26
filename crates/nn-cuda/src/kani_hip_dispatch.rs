// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for HIP dispatch launch selection.
//!
//! Proves that representative `DispatchStep` variants map to the expected
//! HIP launch shapes, including reduction shared memory sizing and rocWMMA
//! dispatch selection.
//!
//! Part of #3727.

#[cfg(kani)]
mod proofs {
    use crate::codegen_hip::HIP_BLOCK_SIZE;
    use crate::hip_dispatch::launch_config_for_step;
    use nn_dsl::{DispatchStep, ScalarType, SimdgroupMatMulParams, TensorNodeId};

    fn node(id: usize) -> TensorNodeId {
        TensorNodeId::new(id)
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_sigmoid_dispatch_uses_elementwise_launch() {
        let total_elements: u16 = kani::any();
        kani::assume(total_elements > 0);

        let step = DispatchStep::Sigmoid {
            kernel_name: "sigmoid".into(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            total_elements: usize::from(total_elements),
        };

        let cfg = launch_config_for_step(&step).unwrap().unwrap();
        assert_eq!(cfg.block.x, HIP_BLOCK_SIZE as u32);
        assert_eq!(cfg.block.y, 1);
        assert_eq!(cfg.block.z, 1);
        assert_eq!(cfg.shared_mem_bytes, 0);
        assert!(u64::from(cfg.grid.x) * u64::from(cfg.block.x) >= u64::from(total_elements));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_softmax_dispatch_uses_reduction_shared_memory() {
        let outer_size: u16 = kani::any();
        kani::assume(outer_size > 0);

        let step = DispatchStep::Softmax {
            kernel_name: "softmax".into(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            axis: 0,
            axis_size: 1,
            outer_size: usize::from(outer_size),
        };

        let cfg = launch_config_for_step(&step).unwrap().unwrap();
        assert_eq!(cfg.block.x, HIP_BLOCK_SIZE as u32);
        assert_eq!(cfg.block.y, 1);
        assert_eq!(cfg.shared_mem_bytes, HIP_BLOCK_SIZE as u32 * 4);
        assert!(u64::from(cfg.grid.x) * u64::from(cfg.block.x) >= u64::from(outer_size));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_large_matmul_dispatch_uses_16x16_tiles() {
        let m_tiles: u8 = kani::any();
        let n_tiles: u8 = kani::any();
        kani::assume(m_tiles > 0);
        kani::assume(n_tiles > 0);

        let m = usize::from(m_tiles) * 16;
        let n = usize::from(n_tiles) * 16;
        let step = DispatchStep::MatMul {
            kernel_name: "matmul".into(),
            dtype: ScalarType::F16,
            left: node(0),
            right: node(1),
            output: node(2),
            m,
            k: 16,
            n,
            batch_size: 1,
            transpose_right: false,
            broadcast_right: false,
            scale: None,
            total_elements: m * n,
        };

        let cfg = launch_config_for_step(&step).unwrap().unwrap();
        assert_eq!(cfg.block.x, 16);
        assert_eq!(cfg.block.y, 16);
        assert_eq!(cfg.block.z, 1);
        assert_eq!(cfg.shared_mem_bytes, 0);
        assert!(u64::from(cfg.grid.x) * 16 >= n as u64);
        assert!(u64::from(cfg.grid.y) * 16 >= m as u64);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_small_matmul_dispatch_falls_back_to_elementwise() {
        let m: u8 = kani::any();
        let n: u8 = kani::any();
        kani::assume(m > 0);
        kani::assume(n > 0);
        kani::assume(m < 16 || n < 16);

        let total_elements = usize::from(m) * usize::from(n);
        let step = DispatchStep::MatMul {
            kernel_name: "matmul_small".into(),
            dtype: ScalarType::F32,
            left: node(0),
            right: node(1),
            output: node(2),
            m: usize::from(m),
            k: 8,
            n: usize::from(n),
            batch_size: 1,
            transpose_right: false,
            broadcast_right: false,
            scale: None,
            total_elements,
        };

        let cfg = launch_config_for_step(&step).unwrap().unwrap();
        assert_eq!(cfg.block.x, HIP_BLOCK_SIZE as u32);
        assert_eq!(cfg.block.y, 1);
        assert_eq!(cfg.shared_mem_bytes, 0);
        assert!(u64::from(cfg.grid.x) * u64::from(cfg.block.x) >= total_elements as u64);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_simdgroup_matmul_rocwmma_branch_uses_batch_grid() {
        let m_tiles: u8 = kani::any();
        let n_tiles: u8 = kani::any();
        let k_tiles: u8 = kani::any();
        let batch_size: u8 = kani::any();
        kani::assume(m_tiles >= 8);
        kani::assume(n_tiles >= 8);
        kani::assume(k_tiles >= 8);
        kani::assume(batch_size > 0);

        let m = usize::from(m_tiles) * 16;
        let k = usize::from(k_tiles) * 16;
        let n = usize::from(n_tiles) * 16;
        let step = DispatchStep::SimdgroupMatMul(SimdgroupMatMulParams {
            kernel_name: "simdgroup_matmul".into(),
            dtype: ScalarType::F16,
            left: node(0),
            right: node(1),
            output: node(2),
            m,
            k,
            n,
            batch_size: usize::from(batch_size),
            transpose_right: false,
            broadcast_right: false,
            scale: None,
        });

        let cfg = launch_config_for_step(&step).unwrap().unwrap();
        assert_eq!(cfg.block.x, 256);
        assert_eq!(cfg.block.y, 1);
        assert_eq!(cfg.block.z, 1);
        assert_eq!(cfg.grid.z, u32::from(batch_size));
        assert_eq!(cfg.shared_mem_bytes, 0);
        assert!(u64::from(cfg.grid.x) * 32 >= n as u64);
        assert!(u64::from(cfg.grid.y) * 32 >= m as u64);
    }
}
