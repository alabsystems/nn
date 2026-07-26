// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Display implementations for moonshot CROWN types.

use std::fmt;

use super::{MoonshotCrownBundle, MoonshotPropertyResult};

impl fmt::Display for MoonshotPropertyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "P{}: [{}] {} — {}",
            self.property_index + 1,
            self.level,
            self.property_name,
            self.explanation
        )
    }
}

impl fmt::Display for MoonshotCrownBundle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== Moonshot CROWN Bundle (D={}) ===",
            self.verification_dim
        )?;
        for result in &self.results {
            writeln!(f, "  {result}")?;
        }
        writeln!(
            f,
            "  Overall: {} ({}/{} proven)",
            if self.all_proven { "PASS" } else { "FAIL" },
            self.results.iter().filter(|r| r.proven).count(),
            self.results.len()
        )
    }
}
