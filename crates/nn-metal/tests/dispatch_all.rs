// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Consolidated dispatch tests: cache behavior, LRU eviction, coverage,
//! and MSL dispatch mode resolution.

mod test_utils;

#[path = "dispatch/cache.rs"]
mod dispatch_cache;
#[path = "dispatch/cache_lru.rs"]
mod dispatch_cache_lru;
#[path = "dispatch/coverage.rs"]
mod dispatch_coverage;
#[path = "dispatch/memory_safety.rs"]
mod dispatch_memory_safety;
#[path = "dispatch/modes_from_msl.rs"]
mod dispatch_modes_from_msl;
#[path = "dispatch/modes_from_msl_kernels.rs"]
mod dispatch_modes_from_msl_kernels;
#[path = "dispatch/registry_dispatch_infra.rs"]
mod dispatch_registry_dispatch_infra;
