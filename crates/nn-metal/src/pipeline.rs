// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compiled Metal compute pipeline state.
//!
//! [`ComputePipeline`] holds the Metal pipeline state object alongside its
//! entry point name and fast-math flag. Created by [`MetalContext::compile_pipeline`]
//! or via the [`PipelineCache`](crate::PipelineCache).

/// Compiled Metal compute pipeline metadata.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ComputePipeline {
    inner: metal::ComputePipelineState,
    entry_point: String,
    fast_math: bool,
}

impl ComputePipeline {
    pub(crate) fn from_raw(
        inner: metal::ComputePipelineState,
        entry_point: impl Into<String>,
        fast_math: bool,
    ) -> Self {
        Self {
            inner,
            entry_point: entry_point.into(),
            fast_math,
        }
    }

    /// MSL function name used as the compute pipeline entry point.
    #[must_use]
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Whether this pipeline was compiled with Metal fast-math optimizations.
    #[must_use]
    pub fn fast_math(&self) -> bool {
        self.fast_math
    }

    #[must_use]
    pub(crate) fn inner(&self) -> &metal::ComputePipelineState {
        &self.inner
    }
}
