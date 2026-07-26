// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani harnesses for CompiledPlanSerdeError variant reachability.

#[cfg(kani)]
mod proofs {
    use crate::trace_compile::CompiledPlanSerdeError;

    /// Prove CompiledPlanSerdeError::Io variant is constructible from std::io::Error.
    #[kani::unwind(1)]
    #[kani::proof]
    fn compiled_plan_serde_error_io_variant() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let err: CompiledPlanSerdeError = io_err.into();
        assert!(matches!(err, CompiledPlanSerdeError::Io(_)));
    }

    /// Prove CompiledPlanSerdeError::Json variant is constructible.
    /// We construct a serde_json::Error by parsing invalid JSON.
    #[kani::unwind(1)]
    #[kani::proof]
    fn compiled_plan_serde_error_json_variant() {
        let json_result: Result<serde_json::Value, _> = serde_json::from_str("{invalid");
        let json_err = json_result.unwrap_err();
        let err: CompiledPlanSerdeError = json_err.into();
        assert!(matches!(err, CompiledPlanSerdeError::Json(_)));
    }

    /// Prove both variants are distinct (Io != Json).
    #[kani::unwind(1)]
    #[kani::proof]
    fn compiled_plan_serde_error_variants_distinct() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "io");
        let err_io: CompiledPlanSerdeError = io_err.into();

        // Verify the variant discriminant
        let is_io = matches!(err_io, CompiledPlanSerdeError::Io(_));
        let is_json = matches!(err_io, CompiledPlanSerdeError::Json(_));
        assert!(is_io);
        assert!(!is_json);
    }
}
