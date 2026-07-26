// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch planning types for multi-mode Metal compute dispatch.
//!
//! [`DispatchMode`] describes the logical shape of a kernel launch (element-wise,
//! 2D/3D grid, or per-slice reduction). [`DispatchPlan`] is the concrete grid
//! configuration derived from a mode, ready for the Metal encoder.
//!
//! All conversions are pure functions with no Metal side effects, making them
//! unit-testable without GPU access.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::error::MetalError;

// ---------------------------------------------------------------------------
// Thread-local dispatch plan cache
// ---------------------------------------------------------------------------

/// Maximum entries before the cache is halved (simple size cap).
///
/// Dispatch plans are small (~80 bytes each). 512 entries ≈ 40KB — negligible
/// on any Metal-capable device. Compiled models with fixed shapes produce a
/// bounded set of modes that never triggers eviction in practice.
const DISPATCH_PLAN_CACHE_MAX: usize = 512;

thread_local! {
    static PLAN_CACHE: RefCell<HashMap<DispatchMode, DispatchPlan>> =
        RefCell::new(HashMap::with_capacity(64));
}

/// Clear the thread-local dispatch plan cache.
///
/// Useful for testing and for resetting state between inference sessions.
pub(crate) fn clear_dispatch_plan_cache() {
    PLAN_CACHE.with(|cell| cell.borrow_mut().clear());
}

/// Number of entries in the thread-local dispatch plan cache (diagnostics/testing).
pub(crate) fn dispatch_plan_cache_len() -> usize {
    PLAN_CACHE.with(|cell| cell.borrow().len())
}

/// Logical dispatch shape for a kernel launch.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DispatchMode {
    /// One thread per scalar element, 1D grid.
    Elementwise { total: u32 },

    /// Explicit 2D grid (e.g. RoPE: `[seq_len, head_dim/2]`).
    Grid2D { grid: [u32; 2], threads: [u32; 2] },

    /// Explicit 3D grid (e.g. batched RoPE: `[seq, head_dim/2, heads*batch]`).
    Grid3D { grid: [u32; 3], threads: [u32; 3] },

    /// Per-slice reduction: one threadgroup per outer slice, threads cooperate
    /// to reduce over the inner axis.
    ///
    /// Used by InstanceNorm (K2), RMSNorm (K5), LayerNorm (K7).
    PerSliceReduction {
        /// Number of output slices (rows/channels).
        outer: u32,
        /// Elements per slice to reduce over.
        reduce: u32,
        /// Threads per threadgroup for the reduction.
        threads: u32,
        /// Threadgroup shared memory in bytes for partial sums.
        shared_bytes: u32,
    },
}

/// Concrete Metal dispatch configuration derived from a [`DispatchMode`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct DispatchPlan {
    /// Metal grid size `[width, height, depth]`.
    grid: [u32; 3],
    /// Threads per threadgroup `[width, height, depth]`.
    threads: [u32; 3],
    /// Number of elements in the output buffer.
    output_elems: usize,
    /// Constants to bind after buffers; starting slot depends on dispatch API.
    constants: Vec<u32>,
    /// Threadgroup shared memory in bytes, if needed (e.g. reduction scratch).
    threadgroup_memory_bytes: Option<u64>,
    /// Whether to use `dispatch_threadgroups` (true) or `dispatch_threads` (false).
    use_threadgroups: bool,
}

impl DispatchPlan {
    /// Metal grid size `[width, height, depth]`.
    #[must_use]
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    /// Threads per threadgroup `[width, height, depth]`.
    #[must_use]
    pub fn threads(&self) -> [u32; 3] {
        self.threads
    }

    /// Number of elements in the output buffer.
    #[must_use]
    pub fn output_elems(&self) -> usize {
        self.output_elems
    }

    /// Constant values bound after buffers.
    #[must_use]
    pub fn constants(&self) -> &[u32] {
        &self.constants
    }

    /// Threadgroup shared memory in bytes, if the kernel uses it.
    #[must_use]
    pub fn threadgroup_memory_bytes(&self) -> Option<u64> {
        self.threadgroup_memory_bytes
    }

    /// Whether to use `dispatch_threadgroups` instead of `dispatch_threads`.
    #[must_use]
    pub fn use_threadgroups(&self) -> bool {
        self.use_threadgroups
    }

    /// Override output element count (builder pattern).
    pub fn with_output_elems(mut self, output_elems: usize) -> Self {
        self.output_elems = output_elems;
        self
    }

    /// Override constant values (builder pattern).
    pub fn with_constants(mut self, constants: Vec<u32>) -> Self {
        self.constants = constants;
        self
    }

    /// Override threadgroup shared memory (builder pattern).
    pub fn with_threadgroup_memory_bytes(mut self, threadgroup_memory_bytes: Option<u64>) -> Self {
        self.threadgroup_memory_bytes = threadgroup_memory_bytes;
        self
    }

    /// Override dispatch mode (threadgroups vs threads) (builder pattern).
    pub fn with_use_threadgroups(mut self, use_threadgroups: bool) -> Self {
        self.use_threadgroups = use_threadgroups;
        self
    }
}

impl DispatchMode {
    /// Convert to a concrete [`DispatchPlan`].
    ///
    /// Returns an error if any grid dimension is zero or threadgroup size is
    /// zero (would cause a Metal validation error).
    #[must_use = "returns a Result that may contain an error"]
    pub fn plan(&self) -> Result<DispatchPlan, MetalError> {
        match self {
            Self::Elementwise { total } => plan_elementwise(*total),
            Self::Grid2D { grid, threads } => plan_grid_2d(*grid, *threads),
            Self::Grid3D { grid, threads } => plan_grid_3d(*grid, *threads),
            Self::PerSliceReduction {
                outer,
                reduce,
                threads,
                shared_bytes,
            } => plan_reduction(*outer, *reduce, *threads, *shared_bytes),
        }
    }

    /// Convert to a concrete [`DispatchPlan`], returning a cached clone when
    /// the same `DispatchMode` has been seen before on this thread.
    ///
    /// Compiled models execute the same set of dispatch modes on every forward
    /// pass. Caching eliminates redundant plan computation and `Vec<u32>`
    /// allocation for constants on the hot path.
    ///
    /// The cache is thread-local (`RefCell<HashMap>`) so there is no
    /// synchronization overhead. Clear with [`clear_dispatch_plan_cache()`].
    #[must_use = "returns a Result that may contain an error"]
    pub fn plan_cached(&self) -> Result<DispatchPlan, MetalError> {
        PLAN_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(cached) = cache.get(self) {
                return Ok(cached.clone());
            }
            let plan = self.plan()?;
            // Simple eviction: if we hit the cap, clear half the entries.
            // In practice, compiled models produce a bounded mode set that
            // never triggers this.
            if cache.len() >= DISPATCH_PLAN_CACHE_MAX {
                cache.clear();
            }
            cache.insert(self.clone(), plan.clone());
            Ok(plan)
        })
    }
}

pub(crate) fn plan_elementwise(total: u32) -> Result<DispatchPlan, MetalError> {
    if total == 0 {
        return Ok(DispatchPlan {
            grid: [0, 1, 1],
            threads: [1, 1, 1],
            output_elems: 0,
            constants: vec![total],
            threadgroup_memory_bytes: None,
            use_threadgroups: false,
        });
    }
    let tg = threadgroup_width_1d(total);
    Ok(DispatchPlan {
        grid: [total, 1, 1],
        threads: [tg, 1, 1],
        output_elems: total as usize,
        constants: vec![total],
        threadgroup_memory_bytes: None,
        use_threadgroups: false,
    })
}

pub(crate) fn plan_grid_2d(grid: [u32; 2], threads: [u32; 2]) -> Result<DispatchPlan, MetalError> {
    validate_nonzero(&grid, "grid")?;
    validate_nonzero(&threads, "threadgroup")?;
    let output_elems = (grid[0] as usize)
        .checked_mul(grid[1] as usize)
        .ok_or(MetalError::DispatchSizeOverflow(usize::MAX))?;
    Ok(DispatchPlan {
        grid: [grid[0], grid[1], 1],
        threads: [threads[0], threads[1], 1],
        output_elems,
        constants: vec![grid[0], grid[1]],
        threadgroup_memory_bytes: None,
        use_threadgroups: false,
    })
}

pub(crate) fn plan_grid_3d(grid: [u32; 3], threads: [u32; 3]) -> Result<DispatchPlan, MetalError> {
    validate_nonzero(&grid, "grid")?;
    validate_nonzero(&threads, "threadgroup")?;
    let output_elems = (grid[0] as usize)
        .checked_mul(grid[1] as usize)
        .and_then(|p| p.checked_mul(grid[2] as usize))
        .ok_or(MetalError::DispatchSizeOverflow(usize::MAX))?;
    Ok(DispatchPlan {
        grid: [grid[0], grid[1], grid[2]],
        threads: [threads[0], threads[1], threads[2]],
        output_elems,
        constants: vec![grid[0], grid[1], grid[2]],
        threadgroup_memory_bytes: None,
        use_threadgroups: false,
    })
}

pub(crate) fn plan_reduction(
    outer: u32,
    reduce: u32,
    threads: u32,
    shared_bytes: u32,
) -> Result<DispatchPlan, MetalError> {
    if outer == 0 {
        return Err(MetalError::InvalidGridDimension {
            dimension: "outer",
            value: 0,
        });
    }
    if reduce == 0 {
        return Err(MetalError::InvalidGridDimension {
            dimension: "reduce",
            value: 0,
        });
    }
    if threads == 0 {
        return Err(MetalError::InvalidGridDimension {
            dimension: "threads_per_group",
            value: 0,
        });
    }
    // One threadgroup per outer slice. Within each threadgroup, `threads`
    // threads cooperate to reduce `reduce` elements.
    Ok(DispatchPlan {
        grid: [outer, 1, 1],
        threads: [threads, 1, 1],
        output_elems: outer as usize,
        constants: vec![outer, reduce],
        threadgroup_memory_bytes: Some(u64::from(shared_bytes)),
        use_threadgroups: true,
    })
}

fn validate_nonzero(dims: &[u32], label: &'static str) -> Result<(), MetalError> {
    for &d in dims {
        if d == 0 {
            return Err(MetalError::InvalidGridDimension {
                dimension: label,
                value: 0,
            });
        }
    }
    Ok(())
}

/// Compute threadgroup width for 1D dispatch: `min(64, total)`.
#[inline]
pub(crate) const fn threadgroup_width_1d(total: u32) -> u32 {
    if total < 64 {
        total
    } else {
        64
    }
}

#[cfg(test)]
#[path = "dispatch_plan_tests.rs"]
mod tests;
