// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `nn-optimize` CLI -- self-optimizing ML compiler pipeline.
//!
//! Provides a command-line interface for running the tiered optimization
//! pipeline on compiled execution plans:
//!
//! - **Tier 1 (Gap Analysis):** Read a `FusionGapAnalysis` from stdin and
//!   produce a human-readable or JSON summary.
//! - **Tier 2 (Config Search):** Exhaustive `PeepholeConfig` search over a
//!   computation graph (not yet implemented -- requires graph input).
//! - **Tier 3 (AI-in-Harness):** AI agent optimization loop (not yet
//!   implemented).
//! - **Report:** Generate an optimization dashboard report from a JSON file.
//!
//! # Usage
//!
//! ```bash
//! # Print the FusionGapAnalysis JSON schema:
//! nn-optimize --schema
//!
//! # Tier 1: analyze a gap analysis from stdin:
//! cat gap_analysis.json | nn-optimize --tier 1
//! cat gap_analysis.json | nn-optimize --tier 1 --json
//!
//! # Tier 2 (future): config search on a computation graph:
//! nn-optimize --tier 2 --budget 30s
//!
//! # Report: generate optimization dashboard from a file:
//! nn-optimize report --input gap_analysis.json
//! nn-optimize report --input gap_analysis.json --format markdown
//! nn-optimize report --input gap_analysis.json --format json
//! ```

use std::io::Read;
use std::path::PathBuf;
use std::process;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use nn_dsl::{fusion_gap_analysis_schema, FusionGapAnalysis};

/// Self-optimizing ML compiler pipeline.
///
/// Analyzes compiled execution plans, identifies fusion gaps, and searches
/// for optimal PeepholeConfig settings to minimize GPU dispatch count.
#[derive(Parser)]
#[command(
    name = "nn-optimize",
    about = "Self-optimizing ML compiler pipeline",
    version
)]
struct Cli {
    /// Analysis tier: 1 (gap analysis), 2 (config search), 3 (AI-in-harness).
    #[arg(short, long, default_value = "1")]
    tier: u8,

    /// Time budget for optimization search (e.g., "5m", "30s").
    /// Only used with tier 2.
    #[arg(short, long)]
    budget: Option<String>,

    /// Output in JSON format instead of human-readable.
    #[arg(long)]
    json: bool,

    /// Print the FusionGapAnalysis JSON schema and exit.
    #[arg(long)]
    schema: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Generate an optimization dashboard report from a JSON file.
    Report {
        /// Path to JSON file containing FusionGapAnalysis (single or array).
        #[arg(short, long)]
        input: PathBuf,

        /// Output format for the report.
        #[arg(short, long, default_value = "text", value_enum)]
        format: ReportFormat,
    },
}

/// Output format for the `report` subcommand.
#[derive(Clone, Debug, ValueEnum)]
enum ReportFormat {
    /// Plain text output to stdout.
    Text,
    /// Markdown with headers, tables, and bullet points.
    Markdown,
    /// Structured JSON for programmatic consumption.
    Json,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    // If a subcommand is present, dispatch to it.
    if let Some(command) = cli.command {
        return match command {
            Command::Report { input, format } => cmd_report(&input, &format),
        };
    }

    // --schema: print schema and exit regardless of tier.
    if cli.schema {
        let schema = fusion_gap_analysis_schema();
        let pretty = serde_json::to_string_pretty(&schema).context("failed to serialize schema")?;
        println!("{pretty}");
        return Ok(());
    }

    match cli.tier {
        1 => cmd_tier1(cli.json),
        2 => cmd_tier2(cli.json, cli.budget.as_deref()),
        3 => cmd_tier3(),
        _ => bail!("invalid tier: {}. Valid tiers are 1, 2, 3.", cli.tier),
    }
}

/// Tier 1: Read a `FusionGapAnalysis` from stdin and produce a summary.
///
/// Reads JSON from stdin, deserializes to `FusionGapAnalysis`, and prints
/// the human-readable summary (or JSON output with `--json`).
fn cmd_tier1(json_output: bool) -> Result<()> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .context("failed to read from stdin")?;

    if input.trim().is_empty() {
        bail!("stdin is empty. Pipe a FusionGapAnalysis JSON document to stdin.");
    }

    let value: serde_json::Value =
        serde_json::from_str(&input).context("failed to parse stdin as JSON")?;

    let analysis =
        FusionGapAnalysis::from_json(&value).context("failed to deserialize FusionGapAnalysis")?;

    if json_output {
        // Enrich with computed fields.
        let mut output = analysis.to_json();
        if let Some(obj) = output.as_object_mut() {
            obj.insert(
                "optimization_opportunity_pct".into(),
                serde_json::json!(analysis.optimization_opportunity_pct()),
            );
            obj.insert(
                "blocker_counts".into(),
                serde_json::json!(analysis.blocker_counts()),
            );
        }
        let pretty = serde_json::to_string_pretty(&output).context("failed to serialize output")?;
        println!("{pretty}");
    } else {
        println!("{}", analysis.summarize());

        // Also print cost model estimate hint.
        let pct = analysis.optimization_opportunity_pct();
        if pct > 0.0 {
            println!(
                "\n{} of {} gaps have non-zero savings potential.",
                analysis.gaps.iter().filter(|g| g.savings > 0).count(),
                analysis.gaps.len(),
            );
        }
    }

    Ok(())
}

/// Tier 2: Exhaustive PeepholeConfig search (requires computation graph input).
///
/// Not yet implemented -- requires a graph serialization/loading mechanism.
fn cmd_tier2(json_output: bool, budget: Option<&str>) -> Result<()> {
    let budget_display = budget.unwrap_or("unlimited");

    if json_output {
        let output = serde_json::json!({
            "tier": 2,
            "status": "not_implemented",
            "budget": budget_display,
            "message": "Tier 2 (PeepholeConfig search) requires a computation graph input. \
                        Use `nn compile` to generate a graph, then pipe it here.",
        });
        let pretty = serde_json::to_string_pretty(&output).context("failed to serialize output")?;
        println!("{pretty}");
    } else {
        eprintln!(
            "Tier 2 (PeepholeConfig search) is not yet implemented.\n\
             Budget: {budget_display}\n\
             \n\
             This tier will exhaustively search 2048 PeepholeConfig combinations\n\
             to find the configuration that minimizes dispatch count.\n\
             Requires a computation graph input mechanism (future work)."
        );
    }

    Ok(())
}

/// Tier 3: AI-agent optimization loop.
///
/// Not yet implemented.
fn cmd_tier3() -> Result<()> {
    eprintln!(
        "Tier 3 (AI-in-harness optimization) is not yet implemented.\n\
         \n\
         This tier will use AI agent suggestions to propose new peephole\n\
         passes and native ops based on FusionGapAnalysis results."
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Report subcommand
// ---------------------------------------------------------------------------

/// Read a JSON file and deserialize as a single `FusionGapAnalysis` or an
/// array of them. Returns one or more analyses.
fn load_analyses(path: &std::path::Path) -> Result<Vec<FusionGapAnalysis>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let value: serde_json::Value =
        serde_json::from_str(&data).context("failed to parse input as JSON")?;

    if value.is_array() {
        let analyses: Vec<FusionGapAnalysis> = serde_json::from_value(value)
            .context("failed to deserialize JSON array as Vec<FusionGapAnalysis>")?;
        if analyses.is_empty() {
            bail!("input JSON array is empty -- expected at least one FusionGapAnalysis");
        }
        Ok(analyses)
    } else {
        let analysis: FusionGapAnalysis = serde_json::from_value(value)
            .context("failed to deserialize JSON as FusionGapAnalysis")?;
        Ok(vec![analysis])
    }
}

/// Generate actionable recommendations based on blocker distribution.
fn recommendations_for(counts: &std::collections::BTreeMap<String, usize>) -> Vec<String> {
    let mut recs = Vec::new();

    if let Some(&n) = counts.get("NonFusibleOp") {
        if n > 0 {
            recs.push(format!(
                "NonFusibleOp ({n} gaps): Implement NativeOp variants or extend \
                 is_fusible_elementwise() for the unfused op types."
            ));
        }
    }
    if let Some(&n) = counts.get("FanOut") {
        if n > 0 {
            recs.push(format!(
                "FanOut ({n} gaps): Consider kernel duplication or multi-output \
                 fused kernels for high fan-out nodes."
            ));
        }
    }
    if let Some(&n) = counts.get("ShapeMismatch") {
        if n > 0 {
            recs.push(format!(
                "ShapeMismatch ({n} gaps): Add reshape-absorbing peephole passes \
                 to eliminate intermediate shape changes."
            ));
        }
    }
    if let Some(&n) = counts.get("NoPeepholePattern") {
        if n > 0 {
            recs.push(format!(
                "NoPeepholePattern ({n} gaps): Write new peephole passes that \
                 match these adjacent-kernel patterns."
            ));
        }
    }
    if let Some(&n) = counts.get("NoDependency") {
        if n > 0 {
            recs.push(format!(
                "NoDependency ({n} gaps): These are independent dispatches -- no \
                 fusion possible, but consider concurrent launch."
            ));
        }
    }
    if let Some(&n) = counts.get("NotDispatch") {
        if n > 0 {
            recs.push(format!(
                "NotDispatch ({n} gaps): Non-dispatch steps (NativeOp, Passthrough) -- \
                 not fusible by definition."
            ));
        }
    }
    if let Some(&n) = counts.get("AlreadyOptimal") {
        if n > 0 {
            recs.push(format!(
                "AlreadyOptimal ({n} gaps): Already fused or NativeOp -- no action needed."
            ));
        }
    }

    if recs.is_empty() {
        recs.push("No optimization gaps detected.".to_string());
    }

    recs
}

/// Format a single analysis in plain text.
fn format_text(analysis: &FusionGapAnalysis, segment_label: Option<&str>) -> String {
    let mut out = String::new();

    if let Some(label) = segment_label {
        out.push_str(&format!("=== Segment: {label} ===\n"));
    }

    let pct = analysis.optimization_opportunity_pct();
    out.push_str(&format!(
        "Total dispatches:      {}\n\
         Theoretical minimum:   {}\n\
         Optimization potential: {:.1}%\n",
        analysis.total_dispatches, analysis.theoretical_minimum, pct,
    ));

    let counts = analysis.blocker_counts();
    if !counts.is_empty() {
        out.push_str("\nBlocker distribution:\n");
        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in &sorted {
            out.push_str(&format!("  {name:<22} {count}\n"));
        }
    }

    let recs = recommendations_for(&counts);
    if !recs.is_empty() {
        out.push_str("\nRecommendations:\n");
        for rec in &recs {
            out.push_str(&format!("  - {rec}\n"));
        }
    }

    out
}

/// Format a single analysis in Markdown.
fn format_markdown(analysis: &FusionGapAnalysis, segment_label: Option<&str>) -> String {
    let mut out = String::new();

    if let Some(label) = segment_label {
        out.push_str(&format!("## Segment: {label}\n\n"));
    } else {
        out.push_str("## Optimization Report\n\n");
    }

    let pct = analysis.optimization_opportunity_pct();
    out.push_str(&format!(
        "| Metric | Value |\n\
         |--------|-------|\n\
         | Total dispatches | {} |\n\
         | Theoretical minimum | {} |\n\
         | Optimization potential | {:.1}% |\n\n",
        analysis.total_dispatches, analysis.theoretical_minimum, pct,
    ));

    let counts = analysis.blocker_counts();
    if !counts.is_empty() {
        out.push_str("### Blocker Distribution\n\n");
        out.push_str("| Blocker | Count |\n|---------|-------|\n");
        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in &sorted {
            out.push_str(&format!("| {name} | {count} |\n"));
        }
        out.push('\n');
    }

    let recs = recommendations_for(&counts);
    if !recs.is_empty() {
        out.push_str("### Recommendations\n\n");
        for rec in &recs {
            out.push_str(&format!("- {rec}\n"));
        }
        out.push('\n');
    }

    out
}

/// Format a single analysis as a JSON value with enriched fields.
fn format_json_value(analysis: &FusionGapAnalysis) -> serde_json::Value {
    let mut obj = analysis.to_json();
    if let Some(map) = obj.as_object_mut() {
        map.insert(
            "optimization_opportunity_pct".into(),
            serde_json::json!(analysis.optimization_opportunity_pct()),
        );
        map.insert(
            "blocker_counts".into(),
            serde_json::json!(analysis.blocker_counts()),
        );
        let counts = analysis.blocker_counts();
        map.insert(
            "recommendations".into(),
            serde_json::json!(recommendations_for(&counts)),
        );
    }
    obj
}

/// The `report` subcommand entry point.
fn cmd_report(input: &std::path::Path, format: &ReportFormat) -> Result<()> {
    let analyses = load_analyses(input)?;

    match format {
        ReportFormat::Text => {
            if analyses.len() == 1 {
                print!("{}", format_text(&analyses[0], None));
            } else {
                for (i, a) in analyses.iter().enumerate() {
                    let label = format!("{}", i + 1);
                    print!("{}", format_text(a, Some(&label)));
                    if i + 1 < analyses.len() {
                        println!();
                    }
                }
            }
        }
        ReportFormat::Markdown => {
            println!("# nn Optimization Dashboard\n");
            if analyses.len() == 1 {
                print!("{}", format_markdown(&analyses[0], None));
            } else {
                for (i, a) in analyses.iter().enumerate() {
                    let label = format!("{}", i + 1);
                    print!("{}", format_markdown(a, Some(&label)));
                }
            }
        }
        ReportFormat::Json => {
            let output = if analyses.len() == 1 {
                format_json_value(&analyses[0])
            } else {
                let arr: Vec<_> = analyses.iter().map(format_json_value).collect();
                serde_json::json!(arr)
            };
            let pretty =
                serde_json::to_string_pretty(&output).context("failed to serialize report")?;
            println!("{pretty}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // ---- Argument Parsing ----

    #[test]
    fn test_default_tier_is_1() {
        let cli = Cli::try_parse_from(["nn-optimize"]).expect("default args should parse");
        assert_eq!(cli.tier, 1);
        assert!(!cli.json);
        assert!(!cli.schema);
        assert!(cli.budget.is_none());
    }

    #[test]
    fn test_schema_flag() {
        let cli = Cli::try_parse_from(["nn-optimize", "--schema"]).expect("--schema should parse");
        assert!(cli.schema);
    }

    #[test]
    fn test_json_flag() {
        let cli = Cli::try_parse_from(["nn-optimize", "--json"]).expect("--json should parse");
        assert!(cli.json);
    }

    #[test]
    fn test_tier_2_with_budget() {
        let cli = Cli::try_parse_from(["nn-optimize", "--tier", "2", "--budget", "30s"])
            .expect("tier 2 with budget should parse");
        assert_eq!(cli.tier, 2);
        assert_eq!(cli.budget.as_deref(), Some("30s"));
    }

    #[test]
    fn test_tier_3() {
        let cli =
            Cli::try_parse_from(["nn-optimize", "--tier", "3"]).expect("tier 3 should parse");
        assert_eq!(cli.tier, 3);
    }

    #[test]
    fn test_combined_flags() {
        let cli = Cli::try_parse_from(["nn-optimize", "--tier", "1", "--json", "--schema"])
            .expect("combined flags should parse");
        assert_eq!(cli.tier, 1);
        assert!(cli.json);
        assert!(cli.schema);
    }

    #[test]
    fn test_short_tier_flag() {
        let cli =
            Cli::try_parse_from(["nn-optimize", "-t", "2"]).expect("short -t flag should parse");
        assert_eq!(cli.tier, 2);
    }

    #[test]
    fn test_short_budget_flag() {
        let cli =
            Cli::try_parse_from(["nn-optimize", "-b", "5m"]).expect("short -b flag should parse");
        assert_eq!(cli.budget.as_deref(), Some("5m"));
    }

    // ---- Schema Output ----

    #[test]
    fn test_schema_output_is_valid_json() {
        let schema = fusion_gap_analysis_schema();
        let pretty = serde_json::to_string_pretty(&schema).expect("schema should serialize");
        let reparsed: serde_json::Value =
            serde_json::from_str(&pretty).expect("serialized schema should reparse");
        assert!(reparsed.is_object());
        assert!(reparsed.get("$schema").is_some());
    }

    // ---- Invalid Tier ----

    #[test]
    fn test_invalid_tier_returns_error() {
        let result = run(Cli {
            tier: 99,
            budget: None,
            json: false,
            schema: false,
            command: None,
        });
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("invalid tier"), "error: {err}");
    }

    // ---- Tier 2 Not Implemented ----

    #[test]
    fn test_tier2_json_output_returns_status() {
        // cmd_tier2 with json=true should not error.
        let result = cmd_tier2(true, Some("10s"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_tier2_human_output_returns_ok() {
        let result = cmd_tier2(false, None);
        assert!(result.is_ok());
    }

    // ---- Tier 3 Not Implemented ----

    #[test]
    fn test_tier3_returns_ok() {
        let result = cmd_tier3();
        assert!(result.is_ok());
    }

    // ---- Report Subcommand Parsing ----

    #[test]
    fn test_report_subcommand_parses() {
        let cli = Cli::try_parse_from(["nn-optimize", "report", "--input", "/tmp/test.json"])
            .expect("report subcommand should parse");
        assert!(cli.command.is_some());
        match cli.command.unwrap() {
            Command::Report { input, format } => {
                assert_eq!(input, PathBuf::from("/tmp/test.json"));
                assert!(matches!(format, ReportFormat::Text));
            }
        }
    }

    #[test]
    fn test_report_subcommand_markdown_format() {
        let cli = Cli::try_parse_from([
            "nn-optimize",
            "report",
            "--input",
            "/tmp/test.json",
            "--format",
            "markdown",
        ])
        .expect("report with markdown format should parse");
        match cli.command.unwrap() {
            Command::Report { format, .. } => {
                assert!(matches!(format, ReportFormat::Markdown));
            }
        }
    }

    #[test]
    fn test_report_subcommand_json_format() {
        let cli = Cli::try_parse_from([
            "nn-optimize",
            "report",
            "--input",
            "/tmp/test.json",
            "--format",
            "json",
        ])
        .expect("report with json format should parse");
        match cli.command.unwrap() {
            Command::Report { format, .. } => {
                assert!(matches!(format, ReportFormat::Json));
            }
        }
    }

    // ---- Report Formatting ----

    fn sample_analysis() -> FusionGapAnalysis {
        use nn_dsl::FusionBlocker;
        FusionGapAnalysis {
            gaps: vec![
                nn_dsl::FusionGap {
                    step_a: 0,
                    step_b: 1,
                    kernel_a: "snake".into(),
                    kernel_b: "add".into(),
                    reason: FusionBlocker::NonFusibleOp,
                    savings: 1,
                },
                nn_dsl::FusionGap {
                    step_a: 2,
                    step_b: 3,
                    kernel_a: "matmul".into(),
                    kernel_b: "relu".into(),
                    reason: FusionBlocker::FanOut,
                    savings: 1,
                },
            ],
            total_dispatches: 10,
            theoretical_minimum: 8,
        }
    }

    #[test]
    fn test_format_text_contains_dispatches() {
        let a = sample_analysis();
        let text = format_text(&a, None);
        assert!(text.contains("Total dispatches:      10"), "text: {text}");
        assert!(text.contains("Theoretical minimum:   8"), "text: {text}");
        assert!(text.contains("20.0%"), "text: {text}");
        assert!(text.contains("NonFusibleOp"), "text: {text}");
        assert!(text.contains("FanOut"), "text: {text}");
        assert!(text.contains("Recommendations:"), "text: {text}");
    }

    #[test]
    fn test_format_text_with_segment_label() {
        let a = sample_analysis();
        let text = format_text(&a, Some("encoder"));
        assert!(text.contains("=== Segment: encoder ==="), "text: {text}");
    }

    #[test]
    fn test_format_markdown_contains_table() {
        let a = sample_analysis();
        let md = format_markdown(&a, None);
        assert!(md.contains("## Optimization Report"), "md: {md}");
        assert!(md.contains("| Total dispatches | 10 |"), "md: {md}");
        assert!(md.contains("### Blocker Distribution"), "md: {md}");
        assert!(md.contains("### Recommendations"), "md: {md}");
    }

    #[test]
    fn test_format_json_value_has_enriched_fields() {
        let a = sample_analysis();
        let val = format_json_value(&a);
        assert!(val.get("optimization_opportunity_pct").is_some());
        assert!(val.get("blocker_counts").is_some());
        assert!(val.get("recommendations").is_some());
        let pct = val["optimization_opportunity_pct"].as_f64().unwrap();
        assert!((pct - 20.0).abs() < 0.01, "pct: {pct}");
    }

    #[test]
    fn test_recommendations_empty_analysis() {
        let counts = std::collections::BTreeMap::new();
        let recs = recommendations_for(&counts);
        assert_eq!(recs.len(), 1);
        assert!(recs[0].contains("No optimization gaps"), "recs: {recs:?}");
    }

    #[test]
    fn test_recommendations_covers_all_blockers() {
        let mut counts = std::collections::BTreeMap::new();
        counts.insert("NonFusibleOp".to_string(), 5);
        counts.insert("FanOut".to_string(), 3);
        counts.insert("ShapeMismatch".to_string(), 2);
        counts.insert("NoPeepholePattern".to_string(), 1);
        let recs = recommendations_for(&counts);
        assert_eq!(recs.len(), 4, "recs: {recs:?}");
    }

    // ---- Report File Loading ----

    #[test]
    fn test_load_analyses_single() {
        let dir = std::env::temp_dir().join("nn_optimize_test_single");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("single.json");
        let a = sample_analysis();
        let json = serde_json::to_string_pretty(&a).unwrap();
        std::fs::write(&path, &json).unwrap();

        let loaded = load_analyses(&path).expect("should load single analysis");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].total_dispatches, 10);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_analyses_array() {
        let dir = std::env::temp_dir().join("nn_optimize_test_array");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("array.json");
        let a = sample_analysis();
        let json = serde_json::to_string_pretty(&vec![&a, &a]).unwrap();
        std::fs::write(&path, &json).unwrap();

        let loaded = load_analyses(&path).expect("should load array of analyses");
        assert_eq!(loaded.len(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_analyses_missing_file() {
        let result = load_analyses(std::path::Path::new("/tmp/nonexistent_nn_test.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_report_text_format() {
        let dir = std::env::temp_dir().join("nn_optimize_test_report_text");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.json");
        let a = sample_analysis();
        std::fs::write(&path, serde_json::to_string(&a).unwrap()).unwrap();

        let result = cmd_report(&path, &ReportFormat::Text);
        assert!(result.is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_cmd_report_markdown_format() {
        let dir = std::env::temp_dir().join("nn_optimize_test_report_md");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.json");
        let a = sample_analysis();
        std::fs::write(&path, serde_json::to_string(&a).unwrap()).unwrap();

        let result = cmd_report(&path, &ReportFormat::Markdown);
        assert!(result.is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_cmd_report_json_format() {
        let dir = std::env::temp_dir().join("nn_optimize_test_report_json");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.json");
        let a = sample_analysis();
        std::fs::write(&path, serde_json::to_string(&a).unwrap()).unwrap();

        let result = cmd_report(&path, &ReportFormat::Json);
        assert!(result.is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }
}
