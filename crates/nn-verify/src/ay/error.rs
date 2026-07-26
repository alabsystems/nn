// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Re-export of SmtError from the always-available `smt_error` module.
//!
//! SmtError was extracted to `crate::smt_error` (#859) so analytical bounds
//! tests can run without the `ay-smt` feature flag. This module preserves
//! the `crate::ay::SmtError` path for existing ay code.

pub(crate) use crate::smt_error::SmtError;

// Tests for SmtError now live in smt_error_tests.rs (always-available).
