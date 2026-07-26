// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source text paired with compilation options.
//!
//! [`KernelSource`] bundles the MSL source, entry point name, and fast-math
//! flag needed to compile one Metal compute pipeline. It serves as the
//! cache key for [`PipelineCache`](crate::PipelineCache).

/// Input source used to compile one Metal compute pipeline.
///
/// When `function_constants` is non-empty, the pipeline is specialized
/// at creation time via `MTLFunctionConstantValues`. This enables the
/// Metal compiler to unroll loops and eliminate dead code based on
/// compile-time-known values (#3449).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct KernelSource {
    msl_source: String,
    entry_point: String,
    fast_math: bool,
    /// Metal function constant values: `(index, uint32_value)`.
    /// Each entry maps to a `[[function_constant(index)]]` declaration
    /// in the MSL source. Part of #3449.
    function_constants: Vec<(u32, u32)>,
}

impl KernelSource {
    /// Create a new kernel source with fast-math disabled by default.
    #[must_use]
    pub fn new(msl_source: impl Into<String>, entry_point: impl Into<String>) -> Self {
        Self {
            msl_source: msl_source.into(),
            entry_point: entry_point.into(),
            fast_math: false,
            function_constants: Vec::new(),
        }
    }

    /// Enable or disable Metal fast-math compilation (builder pattern).
    #[must_use]
    pub fn with_fast_math(mut self, fast_math: bool) -> Self {
        self.fast_math = fast_math;
        self
    }

    /// Add a uint32 function constant at the given index (builder pattern).
    ///
    /// Corresponds to `constant uint NAME [[function_constant(index)]]`
    /// in the MSL source. The Metal compiler uses these values to specialize
    /// the pipeline at creation time (loop unrolling, dead code elimination).
    #[must_use]
    pub fn with_function_constant(mut self, index: u32, value: u32) -> Self {
        self.function_constants.push((index, value));
        self
    }

    /// The raw MSL source text.
    #[must_use]
    pub fn msl_source(&self) -> &str {
        &self.msl_source
    }

    /// Kernel entry point function name in the MSL source.
    #[must_use]
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Whether fast-math is enabled for compilation.
    #[must_use]
    pub fn fast_math(&self) -> bool {
        self.fast_math
    }

    /// Function constant `(index, value)` pairs for pipeline specialization.
    #[must_use]
    pub fn function_constants(&self) -> &[(u32, u32)] {
        &self.function_constants
    }
}

#[cfg(test)]
mod tests {
    use super::KernelSource;

    #[test]
    fn test_new_defaults_fast_math_false() {
        let ks = KernelSource::new("code", "entry");
        assert_eq!(ks.msl_source(), "code");
        assert_eq!(ks.entry_point(), "entry");
        assert!(!ks.fast_math());
    }

    #[test]
    fn test_with_fast_math_true() {
        let ks = KernelSource::new("code", "entry").with_fast_math(true);
        assert!(ks.fast_math());
    }

    #[test]
    fn test_with_fast_math_false_explicit() {
        let ks = KernelSource::new("code", "entry")
            .with_fast_math(true)
            .with_fast_math(false);
        assert!(!ks.fast_math());
    }

    #[test]
    fn test_equality_same_source() {
        let a = KernelSource::new("x", "y").with_fast_math(true);
        let b = KernelSource::new("x", "y").with_fast_math(true);
        assert_eq!(a, b);
    }

    #[test]
    fn test_inequality_different_fast_math() {
        let a = KernelSource::new("x", "y").with_fast_math(false);
        let b = KernelSource::new("x", "y").with_fast_math(true);
        assert_ne!(a, b);
    }

    #[test]
    fn test_inequality_different_entry_point() {
        let a = KernelSource::new("x", "y");
        let b = KernelSource::new("x", "z");
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::HashSet;
        let a = KernelSource::new("code", "entry").with_fast_math(true);
        let b = KernelSource::new("code", "entry").with_fast_math(true);
        let mut set = HashSet::new();
        set.insert(a);
        // b should be found in the set since it's equal to a.
        assert!(set.contains(&b));
    }

    #[test]
    fn test_accepts_string_and_str() {
        let from_str = KernelSource::new("src", "fn");
        let from_string = KernelSource::new(String::from("src"), String::from("fn"));
        assert_eq!(from_str, from_string);
    }

    #[test]
    fn test_function_constants_default_empty() {
        let ks = KernelSource::new("code", "entry");
        assert!(ks.function_constants().is_empty());
    }

    #[test]
    fn test_function_constants_builder() {
        let ks = KernelSource::new("code", "entry")
            .with_function_constant(0, 3)
            .with_function_constant(1, 1);
        assert_eq!(ks.function_constants(), &[(0, 3), (1, 1)]);
    }

    #[test]
    fn test_function_constants_affect_equality() {
        let a = KernelSource::new("code", "entry").with_function_constant(0, 3);
        let b = KernelSource::new("code", "entry").with_function_constant(0, 7);
        assert_ne!(a, b);
    }

    #[test]
    fn test_function_constants_affect_hash() {
        use std::collections::HashSet;
        let a = KernelSource::new("code", "entry").with_function_constant(0, 3);
        let b = KernelSource::new("code", "entry").with_function_constant(0, 7);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(!set.contains(&b));
    }

    #[test]
    fn test_no_constants_equals_empty_constants() {
        let a = KernelSource::new("code", "entry");
        let b = KernelSource::new("code", "entry");
        assert_eq!(a, b);
    }
}
