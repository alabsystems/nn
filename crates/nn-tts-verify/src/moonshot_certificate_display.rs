// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Display and JSON serialization for [`MoonshotCertificate`].

use std::fmt;

use super::{MoonshotCertificate, VerificationLevel};

impl MoonshotCertificate {
    /// Serialize to JSON string for archival.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("{\n");
        out.push_str(&format!("  \"schema_version\": {},\n", self.schema_version));
        out.push_str(&format!("  \"model_name\": {:?},\n", self.model_name));
        out.push_str(&format!(
            "  \"input_specification\": {:?},\n",
            self.input_specification
        ));
        out.push_str(&format!("  \"source_hash\": {:?},\n", self.source_hash));
        out.push_str(&format!(
            "  \"verification_date\": {:?},\n",
            self.verification_date
        ));
        if let Some(dim) = self.verification_dim {
            out.push_str(&format!("  \"verification_dim\": {dim},\n"));
        }
        out.push_str(&format!(
            "  \"all_at_least_partial\": {},\n",
            self.all_at_least_partial
        ));
        out.push_str(&format!("  \"all_proven\": {},\n", self.all_proven));
        out.push_str(&format!(
            "  \"constructive_proof_count\": {},\n",
            self.constructive_proof_count
        ));
        out.push_str("  \"properties\": [\n");

        for (i, prop) in self.properties.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"index\": {},\n", prop.property_index));
            out.push_str(&format!("      \"name\": {:?},\n", prop.property_name));
            out.push_str(&format!(
                "      \"level\": {:?},\n",
                format!("{}", prop.level)
            ));
            if let Some(v) = prop.bound_value {
                out.push_str(&format!("      \"bound_value\": {v},\n"));
            }
            if let Some(t) = prop.threshold {
                out.push_str(&format!("      \"threshold\": {t},\n"));
            }
            out.push_str("      \"proof_artifacts\": [");
            for (j, a) in prop.proof_artifacts.iter().enumerate() {
                out.push_str(&format!("{a:?}"));
                if j + 1 < prop.proof_artifacts.len() {
                    out.push_str(", ");
                }
            }
            out.push_str("],\n");
            out.push_str("      \"assumptions\": [");
            for (j, a) in prop.assumptions.iter().enumerate() {
                out.push_str(&format!("{a:?}"));
                if j + 1 < prop.assumptions.len() {
                    out.push_str(", ");
                }
            }
            out.push_str("]\n");
            out.push_str("    }");
            if i + 1 < self.properties.len() {
                out.push(',');
            }
            out.push('\n');
        }

        out.push_str("  ]\n");
        out.push_str("}\n");
        out
    }
}

impl fmt::Display for MoonshotCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Moonshot Certificate: {} ===", self.model_name)?;
        writeln!(f, "Schema: v{}", self.schema_version)?;
        writeln!(f, "Input: {}", self.input_specification)?;
        writeln!(f, "Date: {}", self.verification_date)?;
        writeln!(f, "Source: {}", self.source_hash)?;
        if let Some(dim) = self.verification_dim {
            writeln!(f, "CROWN dimension: D={dim}")?;
        }
        writeln!(f)?;

        for prop in &self.properties {
            write!(
                f,
                "  P{}: [{}] {}",
                prop.property_index + 1,
                prop.level,
                prop.property_name
            )?;
            if let (Some(bound), Some(thresh)) = (prop.bound_value, prop.threshold) {
                write!(f, " (bound={bound:.4}, threshold={thresh:.4})")?;
            }
            writeln!(f)?;
        }

        let proven_count = self
            .properties
            .iter()
            .filter(|p| {
                matches!(
                    p.level,
                    VerificationLevel::CrownProven
                        | VerificationLevel::KaniProven
                        | VerificationLevel::SmtProven
                )
            })
            .count();

        writeln!(f)?;
        writeln!(
            f,
            "Status: {proven_count}/{} properties proven, all_partial={}, all_proven={}, constructive_proofs={}",
            self.properties.len(),
            self.all_at_least_partial,
            self.all_proven,
            self.constructive_proof_count,
        )
    }
}

/// Get current date as ISO 8601 string (YYYY-MM-DD).
///
/// Uses `std::time::SystemTime` to avoid external dependencies.
pub(super) fn current_date() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Convert unix timestamp to civil date (days since epoch → y/m/d).
    let days = (secs / 86_400) as i64;
    let (year, month, day) = days_to_civil(days);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Convert days since Unix epoch to (year, month, day).
///
/// Algorithm from Howard Hinnant's `chrono`-compatible civil_from_days:
/// <http://howardhinnant.github.io/date_algorithms.html#civil_from_days>
fn days_to_civil(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = (z - era * 146_097) as u32; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = (i64::from(yoe) + era * 400) as i32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify days_to_civil against known dates (Howard Hinnant algorithm).
    #[test]
    fn test_days_to_civil_known_dates() {
        // Unix epoch: 1970-01-01 = day 0
        assert_eq!(days_to_civil(0), (1970, 1, 1));

        // 2000-01-01 = day 10957 (30 years: 365*30 + 7 leap days)
        assert_eq!(days_to_civil(10957), (2000, 1, 1));

        // 2024-02-29 (leap day): 2024 is a leap year
        // 2024-01-01 = day 19723, Jan(31) + Feb 1-29(29) = 60 days, day 19782
        assert_eq!(days_to_civil(19782), (2024, 2, 29));

        // 2026-03-22 (today per project): 2026-01-01 = day 20454
        // Jan=31, Feb=28, Mar 1-22 = 22 → offset = 31+28+22-1 = 80
        assert_eq!(days_to_civil(20534), (2026, 3, 22));
    }

    /// Verify days_to_civil handles negative days (before Unix epoch).
    #[test]
    fn test_days_to_civil_before_epoch() {
        // 1969-12-31 = day -1
        assert_eq!(days_to_civil(-1), (1969, 12, 31));

        // 1900-01-01: not a leap year (divisible by 100 but not 400)
        // 1900-01-01 = day -25567
        assert_eq!(days_to_civil(-25567), (1900, 1, 1));
    }
}
