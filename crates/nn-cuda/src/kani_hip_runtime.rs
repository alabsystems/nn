// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for HIP runtime validation.
//!
//! Proves launch configuration limits used by the HIP runtime wrapper.
//!
//! Part of #3727.

#[cfg(kani)]
mod proofs {
    use crate::hip_ffi::{Dim3, LaunchConfig};
    use crate::hip_runtime::{validate_launch_config, HipRuntimeError};

    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_launch_validation_accepts_valid_config() {
        let grid_x: u16 = kani::any();
        let block_x: u16 = kani::any();
        kani::assume(grid_x > 0);
        kani::assume(block_x > 0);
        kani::assume(block_x <= 1024);

        let cfg = LaunchConfig {
            grid: Dim3::d1(u32::from(grid_x)),
            block: Dim3::d1(u32::from(block_x)),
            shared_mem_bytes: 0,
        };

        assert!(validate_launch_config(&cfg).is_ok());
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_launch_validation_rejects_zero_dimensions() {
        let zero_grid = LaunchConfig {
            grid: Dim3::d1(0),
            block: Dim3::d1(1),
            shared_mem_bytes: 0,
        };
        assert!(matches!(
            validate_launch_config(&zero_grid),
            Err(HipRuntimeError::InvalidLaunchConfig { .. })
        ));

        let zero_block = LaunchConfig {
            grid: Dim3::d1(1),
            block: Dim3::d1(0),
            shared_mem_bytes: 0,
        };
        assert!(matches!(
            validate_launch_config(&zero_block),
            Err(HipRuntimeError::InvalidLaunchConfig { .. })
        ));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_launch_validation_rejects_thread_counts_above_limit() {
        let cfg = LaunchConfig {
            grid: Dim3::d1(1),
            block: Dim3::new(33, 32, 1),
            shared_mem_bytes: 0,
        };

        let err = validate_launch_config(&cfg).unwrap_err();
        assert!(matches!(err, HipRuntimeError::InvalidLaunchConfig { .. }));
    }
}
