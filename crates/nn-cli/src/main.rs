// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `nn` CLI — the hero command-line interface for the nn ML framework.
//!
//! # Usage
//!
//! ```bash
//! # Compile pre-exported torch.export artifacts to Metal-ready inference:
//! nn convert graph.json weights.safetensors
//!
//! # Export from a PyTorch checkpoint via the helper script first:
//! nn convert model.pt --from-pytorch --model-spec mymodule:NnModel \
//!     --input-shape 1 3 224 224
//!
//! # With all options:
//! nn convert graph.json weights.safetensors \
//!     --target metal \
//!     --optimize aggressive \
//!     --verify full \
//!     --reference trace.safetensors \
//!     --output model.nnc \
//!     --report-output model.convert.json
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

/// nn — verified ML framework CLI.
///
/// Compile exported `torch.export` artifacts to optimized Metal inference
/// pipelines, or shell out to the helper export script for PyTorch checkpoints.
/// Start with `nn device`, the one subcommand that needs no input files.
#[derive(Debug, Parser)]
#[command(name = "nn", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Report the Metal device and pipeline-cache status. Takes no input files.
    ///
    /// This is the zero-input first run: every other subcommand needs a
    /// `graph.json` + `weights.safetensors` pair you must already have, so
    /// `nn device` is the one command that shows whether this build can talk to
    /// the GPU at all. It initializes the same global Metal backend the other
    /// subcommands use and prints what that initialization found; it compiles
    /// nothing and runs no model.
    Device {
        /// Print the device report as JSON instead of the human-readable format.
        #[arg(long)]
        json: bool,
    },

    /// Compile exported `torch.export` artifacts (`graph.json` +
    /// `weights.safetensors`) to a Metal-ready model plus a structured report.
    ///
    /// Without --from-pytorch: expects pre-exported graph.json +
    /// weights.safetensors.
    /// With --from-pytorch: shells out to `nn_export.py` first, then compiles
    /// the exported artifacts. This is not raw ONNX input.
    Convert {
        /// Path to the torch.export graph JSON file.
        /// When --from-pytorch is set, this becomes the checkpoint path for
        /// the helper export step (pass a dummy value if the model needs no
        /// checkpoint).
        graph: PathBuf,

        /// Path to the safetensors weights file.
        /// Optional when --from-pytorch is set (weights are exported automatically).
        weights: Option<PathBuf>,

        /// Export from a PyTorch model via nn_export.py subprocess.
        /// When set, --model-spec is required and nn_export.py converts
        /// the model to graph.json + weights.safetensors in a temp directory.
        #[arg(long)]
        from_pytorch: bool,

        /// Python model specification as module:class (e.g. mymodule:NnModel).
        /// Required when --from-pytorch is set.
        #[arg(long)]
        model_spec: Option<String>,

        /// Example input shape for torch.export (e.g. --input-shape 1 3 224 224).
        /// Used with --from-pytorch to construct the example tensor.
        #[arg(long, num_args = 1..)]
        input_shape: Option<Vec<usize>>,

        /// Also capture reference activations when exporting from PyTorch.
        #[arg(long)]
        capture_reference: bool,

        /// Compilation target backend.
        #[arg(long, default_value = "metal")]
        target: Target,

        /// Optimization level.
        #[arg(long, default_value = "normal")]
        optimize: Optimize,

        /// Verification/report level request.
        ///
        /// The populated report fields depend on which verification features
        /// this CLI build enables. `full` requests the fullest report path
        /// available in this build today: bounds when supported, plus optional
        /// reference parity when `--reference` is provided. It does not run
        /// inline Kani kernel-safety reporting.
        #[arg(long, default_value = "bounds")]
        verify: Verify,

        /// Path to PyTorch reference activations for L3 parity checking.
        #[arg(long)]
        reference: Option<PathBuf>,

        /// Output path for the compiled plan (.nnc file).
        ///
        /// When set, saves the compiled plan to disk so it can be loaded
        /// later with `nn run --compiled` to skip recompilation.
        #[arg(long)]
        output: Option<PathBuf>,

        /// Output path for the structured conversion report JSON artifact.
        ///
        /// When set, writes the same `ConvertReport` JSON emitted by `--json`
        /// to disk.
        #[arg(long)]
        report_output: Option<PathBuf>,

        /// Print the structured conversion report as JSON instead of the
        /// human-readable format.
        ///
        /// Useful for piping to `jq`, saving to files, or ingesting into dashboards.
        #[arg(long)]
        json: bool,
    },

    /// Compile exported `torch.export` artifacts (`graph.json` +
    /// `weights.safetensors`) into a `.nnc` plan plus a report JSON artifact.
    ///
    /// Reuses the same default exported-artifact compile/report pipeline as
    /// `nn convert`, including the builder's default verification/report
    /// request, then persists two artifacts:
    /// a `.nnc` (JSON-serialized `CompiledPlan`) file and a structured
    /// `ConvertReport` JSON file describing that compiled plan.
    /// `nn compile` does not currently expose a separate `--verify` flag.
    /// This does not accept raw PyTorch modules or ONNX files.
    Compile {
        /// Path to the torch.export graph JSON file.
        graph: PathBuf,

        /// Path to the safetensors weights file.
        weights: PathBuf,

        /// Output `.nnc` plan path written by `nn compile`.
        #[arg(short, long)]
        output: PathBuf,

        /// Output path for the structured compile report JSON artifact.
        ///
        /// When set, writes the same `ConvertReport` JSON schema used by
        /// `nn convert --report-output`. When omitted, `nn compile`
        /// still persists the report, using a sibling
        /// `<output-stem>.compile.json` path next to the `.nnc` output.
        #[arg(long)]
        report_output: Option<PathBuf>,

        /// Print the structured compile report as JSON instead of the
        /// human-readable format.
        ///
        /// JSON stdout does not replace report persistence; `nn compile`
        /// still writes the structured report to `--report-output` or the
        /// default sibling report path.
        #[arg(long)]
        json: bool,
    },

    /// Run a compiled model from a graph JSON + safetensors weights.
    ///
    /// Loads input tensors from a safetensors file, executes the model on
    /// Metal GPU, and prints output tensor shapes and statistics.
    ///
    /// With --compiled: consumes a saved `.nnc` plan path produced by
    /// `nn compile` and skips the trace compilation step. Graph + weights
    /// are still required for edge map construction and weight upload.
    Run {
        /// Path to the torch.export graph JSON file.
        graph: PathBuf,

        /// Path to the safetensors weights file.
        weights: PathBuf,

        /// Path to a saved `.nnc` plan file from `nn compile`.
        /// Skips trace compilation and loads that persisted plan directly.
        #[arg(long)]
        compiled: Option<PathBuf>,

        /// Path to input tensors as a safetensors file.
        #[arg(long)]
        input: PathBuf,

        /// Optional output file path to save results as safetensors.
        #[arg(long)]
        output: Option<PathBuf>,

        /// Compilation target backend.
        #[arg(long, default_value = "metal")]
        target: Target,

        /// Optimization level.
        #[arg(long, default_value = "normal")]
        optimize: Optimize,
    },

    /// Run the self-optimizing compiler on a .nnc compiled plan.
    ///
    /// Loads a .nnc compiled plan file (from `nn compile`) as the baseline,
    /// imports the model from graph JSON + safetensors weights, then
    /// exhaustively searches the PeepholeConfig space to find the
    /// configuration that minimizes dispatch count and estimated cost.
    ///
    /// Prints a summary comparing baseline vs optimized dispatches and cost,
    /// then saves the optimized plan back to a .nnc file.
    #[command(name = "optimize")]
    Optimize {
        /// Path to the .nnc compiled plan file (baseline).
        plan: PathBuf,

        /// Path to the torch.export graph JSON file.
        #[arg(long)]
        graph: PathBuf,

        /// Path to the safetensors weights file.
        #[arg(long)]
        weights: PathBuf,

        /// Time budget in seconds for the PeepholeConfig search.
        #[arg(long, default_value = "30")]
        budget: u64,

        /// Hardware cost model for tie-breaking.
        #[arg(long, default_value = "apple-m4", value_name = "MODEL")]
        cost_model: CostModelChoice,

        /// Output path for the optimized .nnc file.
        /// Defaults to overwriting the input plan file.
        #[arg(long)]
        output: Option<PathBuf>,

        /// Also write the optimal PeepholeConfig to a JSON file.
        /// The config is saved next to the output .nnc with a
        /// `.peephole.json` suffix.
        #[arg(long)]
        persist: bool,
    },
}

/// Compilation target backend.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Target {
    /// Apple Metal GPU (default).
    Metal,
}

/// Optimization level for the convert pipeline.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Optimize {
    /// No fusion or peephole optimization.
    None,
    /// Full optimization: constant folding + fusion + peephole (default).
    Normal,
    /// Aggressive optimization: full + profile-guided (future).
    Aggressive,
}

/// Verification level for the convert pipeline.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Verify {
    /// No verification.
    None,
    /// Request IBP composition-bounds reporting when available (default).
    Bounds,
    /// Request the fullest report path this CLI build supports today.
    Full,
}

/// Hardware cost model for the optimizer.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CostModelChoice {
    /// Apple M4 (base).
    #[value(name = "apple-m4")]
    AppleM4,
    /// Apple M4 Max (higher memory bandwidth, more GPU cores).
    #[value(name = "apple-m4-max")]
    AppleM4Max,
}

impl From<Optimize> for nn_import::OptLevel {
    fn from(opt: Optimize) -> Self {
        match opt {
            Optimize::None => Self::None,
            Optimize::Normal => Self::Full,
            Optimize::Aggressive => Self::Aggressive,
        }
    }
}

impl From<Verify> for nn_import::VerifyLevel {
    fn from(v: Verify) -> Self {
        match v {
            Verify::None => Self::None,
            Verify::Bounds => Self::Bounds,
            Verify::Full => Self::Full,
        }
    }
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Device { json } => cmd_device(json),
        Commands::Convert {
            graph,
            weights,
            from_pytorch,
            model_spec,
            input_shape,
            capture_reference,
            target: _target,
            optimize,
            verify,
            reference,
            output,
            report_output,
            json,
        } => {
            if from_pytorch {
                cmd_convert_from_pytorch(
                    graph,
                    model_spec,
                    input_shape,
                    capture_reference,
                    optimize,
                    verify,
                    reference,
                    output,
                    report_output,
                    json,
                )
            } else {
                let weights = weights.ok_or_else(|| {
                    anyhow::anyhow!("weights argument is required when --from-pytorch is not set")
                })?;
                cmd_convert(
                    graph,
                    weights,
                    None,
                    optimize,
                    verify,
                    reference,
                    output,
                    report_output,
                    json,
                )
            }
        }
        Commands::Compile {
            graph,
            weights,
            output,
            report_output,
            json,
        } => cmd_compile(graph, weights, output, report_output, json),
        Commands::Run {
            graph,
            weights,
            compiled,
            input,
            output,
            target: _target,
            optimize,
        } => cmd_run(graph, weights, compiled, input, output, optimize),
        Commands::Optimize {
            plan,
            graph,
            weights,
            budget,
            cost_model,
            output,
            persist,
        } => cmd_optimize(plan, graph, weights, budget, cost_model, output, persist),
    }
}

/// `nn device` — report the Metal device and pipeline-cache status.
///
/// A thin wrapper, and deliberately nothing more. Every number it prints is
/// read straight back out of an existing nn-metal entry point:
///
/// - [`nn_metal::MetalBackend::init`] brings up the global context (and loads
///   the embedded metallib, which is what populates the precompiled store),
/// - `MetalContext::device()` supplies the device name,
/// - [`nn_metal::precompiled_pipeline_count`] reports the embedded metallib's
///   preloaded pipelines,
/// - [`nn_metal::PipelineCache::new_global`] plus `len` / `max_entries` /
///   `shared_cache_len` report the L1 and L2 JIT cache state,
/// - [`nn_metal::metal_budget_bytes`] and [`nn_metal::metal_allocated_bytes`]
///   report the GPU memory budget and current allocation.
///
/// It compiles no kernels, imports no graph and dispatches no work.
fn cmd_device(json: bool) -> Result<()> {
    let backend = nn_metal::MetalBackend::init().map_err(|e| {
        anyhow::anyhow!(
            "Metal backend initialization failed: {e}\n\
             \n\
             `nn` runs inference on the Apple Metal GPU, so it needs macOS on \
             Apple silicon with a working Metal device.\n\
             - On non-Apple hardware there is no supported path today; the \
             other backends (nn-cuda, nn-vulkan) are not wired into this CLI.\n\
             - Over SSH or in a sandbox without GPU access, run `nn device` \
             from a local login session instead.\n\
             - To confirm the host sees a GPU at all, run: \
             system_profiler SPDisplaysDataType"
        )
    })?;

    let device_name = backend.context().device().name().to_string();
    let precompiled = nn_metal::precompiled_pipeline_count();

    // The global cache handle is available only because init() succeeded above.
    let cache = nn_metal::PipelineCache::new_global()
        .context("Metal backend initialized but the global pipeline cache was unavailable")?;
    let cache_len = cache.len();
    let cache_capacity = cache.max_entries();
    let shared_cache_len = nn_metal::PipelineCache::shared_cache_len();

    let budget_bytes = nn_metal::metal_budget_bytes();
    let allocated_bytes = nn_metal::metal_allocated_bytes();

    if json {
        let report = serde_json::json!({
            "device": device_name,
            "backend": "metal",
            "precompiled_pipelines": precompiled,
            "pipeline_cache": {
                "l1_entries": cache_len,
                "l1_capacity": cache_capacity,
                "l2_entries": shared_cache_len,
            },
            "memory": {
                "budget_bytes": budget_bytes,
                "allocated_bytes": allocated_bytes,
            },
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("Metal device:          {device_name}");
    println!("Backend:               metal (initialized)");
    println!("Precompiled pipelines: {precompiled} (from the embedded metallib)");
    println!("Pipeline cache:        {cache_len}/{cache_capacity} L1, {shared_cache_len} L2");
    match budget_bytes {
        Some(b) => println!("GPU memory budget:     {:.1} GiB", gib(b)),
        None => println!("GPU memory budget:     unavailable"),
    }
    match allocated_bytes {
        Some(b) => println!("GPU memory allocated:  {:.1} MiB", mib(b as u64)),
        None => println!("GPU memory allocated:  unavailable"),
    }
    println!();
    println!("Next: `nn convert graph.json weights.safetensors` compiles exported");
    println!("torch.export artifacts; `nn convert --help` lists the export options.");

    Ok(())
}

/// Bytes as gibibytes, for the human-readable `nn device` report.
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Bytes as mebibytes, for the human-readable `nn device` report.
fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Locate the `nn_export.py` script.
///
/// Resolution order:
/// 1. `NN_EXPORT_SCRIPT` environment variable
/// 2. `../nn-import/python/nn_export.py` relative to the binary
/// 3. `crates/nn-import/python/nn_export.py` relative to cwd (workspace dev layout)
fn find_export_script() -> Result<PathBuf> {
    // 1. Env var override.
    if let Ok(path) = std::env::var("NN_EXPORT_SCRIPT") {
        let p = PathBuf::from(&path);
        if p.is_file() {
            return Ok(p);
        }
        bail!(
            "NN_EXPORT_SCRIPT is set to '{path}' but the file does not exist"
        );
    }

    // 2. Relative to binary location (install layout).
    if let Ok(exe) = std::env::current_exe() {
        if let Some(bin_dir) = exe.parent() {
            let candidate = bin_dir.join("../nn-import/python/nn_export.py");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }

    // 3. Relative to cwd (workspace development layout).
    let workspace_candidate = Path::new("crates/nn-import/python/nn_export.py");
    if workspace_candidate.is_file() {
        return Ok(workspace_candidate.to_path_buf());
    }

    bail!(
        "could not find nn_export.py. Set NN_EXPORT_SCRIPT env var \
         or run from the nn workspace root"
    )
}

/// Run `nn_export.py` as a subprocess and return the temp directory
/// containing `graph.json` and `weights.safetensors`.
fn run_pytorch_export(
    checkpoint: &Path,
    model_spec: &str,
    input_shape: Option<&[usize]>,
    capture_reference: bool,
) -> Result<tempfile::TempDir> {
    let script = find_export_script()?;

    // Verify python3 is available.
    let python_check = process::Command::new("python3").arg("--version").output();
    match python_check {
        Ok(ref out) if out.status.success() => {}
        Ok(ref out) => bail!(
            "python3 found but returned error: {}",
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(e) => bail!("python3 not found on PATH. Install Python 3 to use --from-pytorch: {e}"),
    }

    // Create temp directory for exported artifacts.
    let tmp_dir =
        tempfile::tempdir().context("failed to create temporary directory for PyTorch export")?;

    eprintln!("nn: exporting PyTorch model via nn_export.py");
    eprintln!("  script:     {}", script.display());
    eprintln!("  model_spec: {model_spec}");
    if checkpoint.exists() {
        eprintln!("  checkpoint: {}", checkpoint.display());
    }
    eprintln!("  output_dir: {}", tmp_dir.path().display());

    let mut cmd = process::Command::new("python3");
    cmd.arg(&script)
        .arg("--model")
        .arg(model_spec)
        .arg("--output-dir")
        .arg(tmp_dir.path());

    if checkpoint.exists() {
        cmd.arg("--checkpoint").arg(checkpoint);
    }

    if let Some(shape) = input_shape {
        cmd.arg("--input-shape");
        for &dim in shape {
            cmd.arg(dim.to_string());
        }
    }

    if capture_reference {
        cmd.arg("--reference");
    }

    let output = cmd
        .stdout(process::Stdio::inherit())
        .stderr(process::Stdio::inherit())
        .output()
        .context("failed to run nn_export.py subprocess")?;

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        bail!(
            "nn_export.py failed with exit code {code}. \
             Ensure torch and safetensors are installed: pip install torch safetensors"
        );
    }

    // Verify expected outputs exist.
    let graph_out = tmp_dir.path().join("graph.json");
    let weights_out = tmp_dir.path().join("weights.safetensors");
    if !graph_out.is_file() {
        bail!(
            "nn_export.py succeeded but graph.json not found in {}",
            tmp_dir.path().display()
        );
    }
    if !weights_out.is_file() {
        bail!(
            "nn_export.py succeeded but weights.safetensors not found in {}",
            tmp_dir.path().display()
        );
    }

    eprintln!("nn: PyTorch export complete");
    Ok(tmp_dir)
}

/// Convert from a PyTorch model: run the exporter, then the standard pipeline.
fn cmd_convert_from_pytorch(
    checkpoint: PathBuf,
    model_spec: Option<String>,
    input_shape: Option<Vec<usize>>,
    capture_reference: bool,
    optimize: Optimize,
    verify: Verify,
    reference: Option<PathBuf>,
    output: Option<PathBuf>,
    report_output: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let model_spec = model_spec.ok_or_else(|| {
        anyhow::anyhow!(
            "--model-spec is required when --from-pytorch is set \
             (e.g. --model-spec mymodule:NnModel)"
        )
    })?;

    // Run the Python exporter.
    let tmp_dir = run_pytorch_export(
        &checkpoint,
        &model_spec,
        input_shape.as_deref(),
        capture_reference,
    )?;

    let graph = tmp_dir.path().join("graph.json");
    let weights = tmp_dir.path().join("weights.safetensors");

    // If reference was captured and none was explicitly provided, use the exported one.
    let reference = reference.or_else(|| {
        if capture_reference {
            let r = tmp_dir.path().join("reference.safetensors");
            if r.is_file() {
                return Some(r);
            }
        }
        None
    });

    // Run the standard convert pipeline. tmp_dir stays alive until this returns,
    // ensuring the temp files are not cleaned up prematurely.
    cmd_convert(
        graph,
        weights,
        Some(nn_import::ConvertIntakePath::CliExportedPytorch),
        optimize,
        verify,
        reference,
        output,
        report_output,
        json,
    )
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create parent directory {}", parent.display()))
}

fn init_metal_cache() -> Result<nn_metal::PipelineCache> {
    let _backend =
        nn_metal::MetalBackend::init().context("failed to initialize Metal GPU backend")?;
    nn_metal::register_metal_dyn_backend();
    nn_metal::PipelineCache::new_global().context("failed to create Metal pipeline cache")
}

fn write_report_json(report: &nn_import::ConvertReport, report_path: &Path) -> Result<()> {
    ensure_parent_dir(report_path)?;
    std::fs::write(report_path, report.to_json())
        .with_context(|| format!("failed to write report JSON to {}", report_path.display()))?;
    let file_size = std::fs::metadata(report_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Saved structured report {} ({} bytes)",
        report_path.display(),
        file_size
    );
    Ok(())
}

fn default_compile_report_path(output: &Path) -> PathBuf {
    let file_name = output
        .file_stem()
        .or_else(|| output.file_name())
        .map(|stem| {
            let mut name = std::ffi::OsString::from(stem);
            name.push(".compile.json");
            name
        })
        .unwrap_or_else(|| std::ffi::OsString::from("compile.compile.json"));
    let mut report_path = output.to_path_buf();
    report_path.set_file_name(file_name);
    report_path
}

fn save_compiled_plan(
    imported: &nn_import::ImportedGraph,
    output: &Path,
    context_label: &str,
) -> Result<()> {
    ensure_parent_dir(output)?;
    let plan = nn_dsl::trace_compile::compile_trace_to_plan_with_fusion(&imported.graph)
        .with_context(|| {
            format!("failed to compile trace to plan for {context_label} serialization")
        })?;
    nn_dsl::save_plan(&plan, output).context("failed to save compiled plan")?;
    let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    eprintln!("Saved {} ({} bytes)", output.display(), file_size);
    Ok(())
}

fn cmd_convert(
    graph: PathBuf,
    weights: PathBuf,
    intake_path: Option<nn_import::ConvertIntakePath>,
    optimize: Optimize,
    verify: Verify,
    reference: Option<PathBuf>,
    output: Option<PathBuf>,
    report_output: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    // Validate input files exist.
    if !graph.exists() {
        bail!("graph file not found: {}", graph.display());
    }
    if !weights.exists() {
        bail!("weights file not found: {}", weights.display());
    }
    if let Some(ref r) = reference {
        if !r.exists() {
            bail!("reference trace not found: {}", r.display());
        }
    }

    eprintln!("nn convert");
    eprintln!("  graph:    {}", graph.display());
    eprintln!("  weights:  {}", weights.display());
    eprintln!("  optimize: {:?}", nn_import::OptLevel::from(optimize));
    eprintln!("  verify:   {:?}", nn_import::VerifyLevel::from(verify));
    if let Some(ref r) = reference {
        eprintln!("  reference: {}", r.display());
    }
    if let Some(ref o) = output {
        eprintln!("  output:   {}", o.display());
    }
    if let Some(ref report_path) = report_output {
        eprintln!("  report:   {}", report_path.display());
    }
    eprintln!();

    // Initialize Metal backend.
    let cache = init_metal_cache()?;

    // Build the convert pipeline.
    let mut builder = nn_import::convert_build(&graph, &weights, &cache)
        .optimize(optimize.into())
        .verify(verify.into());

    if let Some(intake_path) = intake_path {
        builder = builder.intake_path(intake_path);
    }
    if let Some(ref r) = reference {
        builder = builder.reference_trace(r);
    }

    eprintln!("Converting...");
    let result = builder.build().context("convert pipeline failed")?;

    // Print the detailed report.
    if json {
        println!("{}", result.report.to_json());
    } else {
        result.report.print();
    }

    if let Some(ref report_path) = report_output {
        write_report_json(&result.report, report_path)?;
    }

    // Save the compiled plan to disk if --output is specified.
    if let Some(ref o) = output {
        eprintln!("Saving compiled plan to {}...", o.display());
        save_compiled_plan(&result.result.graph, o, "convert")?;
    }

    Ok(())
}

/// Compile a graph JSON + weights into a .nnc precompiled plan file.
fn cmd_compile(
    graph: PathBuf,
    weights: PathBuf,
    output: PathBuf,
    report_output: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    // Validate input files exist.
    if !graph.exists() {
        bail!("graph file not found: {}", graph.display());
    }
    if !weights.exists() {
        bail!("weights file not found: {}", weights.display());
    }

    let default_report_path = default_compile_report_path(&output);
    let (report_path, report_is_default) = match report_output {
        Some(path) => (path, false),
        None => (default_report_path, true),
    };

    eprintln!("nn compile");
    eprintln!("  graph:   {}", graph.display());
    eprintln!("  weights: {}", weights.display());
    eprintln!("  output:  {}", output.display());
    eprintln!(
        "  report:  {}{}",
        report_path.display(),
        if report_is_default {
            " (default sibling report)"
        } else {
            ""
        }
    );
    eprintln!();

    let cache = init_metal_cache()?;
    let result = nn_import::convert_build(&graph, &weights, &cache)
        .optimize(nn_import::OptLevel::Full)
        .build()
        .context("compile pipeline failed")?;

    if json {
        println!("{}", result.report.to_json());
    } else {
        result.report.print();
    }

    write_report_json(&result.report, &report_path)?;

    eprintln!("Saving compiled plan...");
    save_compiled_plan(&result.result.graph, &output, "compile")?;

    Ok(())
}

/// Run a compiled model: import, compile, load inputs, execute, print stats.
///
/// When `compiled` is `Some`, loads a pre-compiled `.nnc` plan to skip
/// the trace compilation step.
fn cmd_run(
    graph: PathBuf,
    weights: PathBuf,
    compiled: Option<PathBuf>,
    input: PathBuf,
    output: Option<PathBuf>,
    optimize: Optimize,
) -> Result<()> {
    // Validate input files exist.
    if !graph.exists() {
        bail!("graph file not found: {}", graph.display());
    }
    if !weights.exists() {
        bail!("weights file not found: {}", weights.display());
    }
    if !input.exists() {
        bail!("input file not found: {}", input.display());
    }

    eprintln!("nn run");
    eprintln!("  graph:    {}", graph.display());
    eprintln!("  weights:  {}", weights.display());
    if let Some(ref c) = compiled {
        eprintln!("  compiled: {}", c.display());
    }
    eprintln!("  input:    {}", input.display());
    eprintln!("  optimize: {:?}", nn_import::OptLevel::from(optimize));
    if let Some(ref o) = output {
        eprintln!("  output:   {}", o.display());
    }
    eprintln!();

    // Initialize Metal backend.
    let _backend =
        nn_metal::MetalBackend::init().context("failed to initialize Metal GPU backend")?;
    nn_metal::register_metal_dyn_backend();
    let cache =
        nn_metal::PipelineCache::new_global().context("failed to create Metal pipeline cache")?;

    // Import the model graph (needed for edge map and input metadata).
    eprintln!("Importing model...");
    let imported = nn_import::import_model(&graph, &weights).context("model import failed")?;

    // Build the CompiledModel: either from a pre-compiled plan or via the
    // full convert pipeline.
    let model;
    if let Some(ref compiled_path) = compiled {
        if !compiled_path.exists() {
            bail!("compiled plan not found: {}", compiled_path.display());
        }
        eprintln!(
            "Loading pre-compiled plan from {}...",
            compiled_path.display()
        );
        let plan = nn_dsl::load_plan(compiled_path).context("failed to load compiled plan")?;
        eprintln!("  {} steps from plan", plan.steps.len());
        model = nn_metal::compiled_model::CompiledModel::from_plan(&plan, &imported.graph, &cache)
            .context("failed to build model from compiled plan")?;
    } else {
        eprintln!("Compiling model...");
        let result = nn_import::convert_build(&graph, &weights, &cache)
            .optimize(optimize.into())
            .verify(nn_import::VerifyLevel::None)
            .build()
            .context("model compilation failed")?;
        model = result.result.model;
    }
    eprintln!(
        "  {} dispatches, {} steps",
        model.num_dispatches(),
        model.num_steps()
    );
    eprintln!();

    // Load input tensors from safetensors file.
    eprintln!("Loading inputs from {}...", input.display());
    let input_tensors =
        nn_core::load_safetensors(&input).context("failed to load input safetensors")?;

    // Map input names to ordered DynTensor refs for execute_dyn.
    let num_inputs = imported.num_user_inputs;
    let input_names = &imported.user_input_names;
    eprintln!("  Model expects {num_inputs} input(s): {input_names:?}");
    eprintln!(
        "  File contains {} tensor(s): {:?}",
        input_tensors.len(),
        input_tensors.keys().collect::<Vec<_>>()
    );

    // Build ordered input list: match by name, or by position if names match count.
    let ordered_inputs = build_ordered_inputs(&input_tensors, input_names, num_inputs)?;

    // Transfer inputs to GPU.
    let gpu_inputs: Vec<nn_core::DynTensor> = ordered_inputs
        .iter()
        .map(|t| {
            t.to_device(&nn_core::Device::metal())
                .context("failed to move input to GPU")
        })
        .collect::<Result<Vec<_>>>()?;
    let gpu_input_refs: Vec<&nn_core::DynTensor> = gpu_inputs.iter().collect();

    // Execute the model.
    eprintln!("Executing model...");
    let outputs = model
        .execute_dyn_outputs(&cache, &gpu_input_refs)
        .context("model execution failed")?;

    // Print output statistics.
    eprintln!();
    eprintln!("Outputs ({}):", outputs.len());
    let mut output_map = HashMap::new();
    for (i, tensor) in outputs.iter().enumerate() {
        let name = imported
            .output_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("output_{i}"));
        print_tensor_stats(&name, tensor)?;
        output_map.insert(name, tensor.clone());
    }

    // Optionally save outputs.
    if let Some(ref out_path) = output {
        eprintln!();
        eprintln!("Saving outputs to {}...", out_path.display());
        nn_core::save_safetensors(&output_map, out_path)
            .context("failed to save output safetensors")?;
        eprintln!("Done.");
    }

    Ok(())
}

/// Run the self-optimizing compiler on a compiled plan.
///
/// Imports the model from graph JSON + safetensors weights (to get the
/// `ComputationGraph` needed for re-compilation), loads the `.nnc` plan
/// as the baseline, then exhaustively searches PeepholeConfig space to
/// find the configuration that minimizes dispatch count and estimated cost.
///
/// Prints a summary, saves the optimized plan, and optionally persists
/// the optimal `PeepholeConfig` as a JSON sidecar file.
fn cmd_optimize(
    plan_path: PathBuf,
    graph_path: PathBuf,
    weights_path: PathBuf,
    budget_secs: u64,
    cost_model_choice: CostModelChoice,
    output: Option<PathBuf>,
    persist: bool,
) -> Result<()> {
    // Validate input files exist.
    if !plan_path.exists() {
        bail!("plan file not found: {}", plan_path.display());
    }
    if !graph_path.exists() {
        bail!("graph file not found: {}", graph_path.display());
    }
    if !weights_path.exists() {
        bail!("weights file not found: {}", weights_path.display());
    }

    let output_path = output.as_ref().unwrap_or(&plan_path);

    eprintln!("nn optimize");
    eprintln!("  plan:       {}", plan_path.display());
    eprintln!("  graph:      {}", graph_path.display());
    eprintln!("  weights:    {}", weights_path.display());
    eprintln!("  budget:     {budget_secs}s");
    eprintln!("  cost-model: {cost_model_choice:?}");
    eprintln!("  output:     {}", output_path.display());
    if persist {
        eprintln!("  persist:    yes");
    }
    eprintln!();

    // Phase 1: Load the baseline plan.
    eprintln!("Loading baseline plan...");
    let baseline_plan = nn_dsl::load_plan(&plan_path).context("failed to load compiled plan")?;
    let baseline_dispatches = nn_dsl::count_dispatches(&baseline_plan);
    eprintln!(
        "  {} steps, {} dispatches, {} weight(s)",
        baseline_plan.steps.len(),
        baseline_dispatches,
        baseline_plan.weight_names.len(),
    );

    // Phase 2: Import the model to get the ComputationGraph.
    eprintln!("Importing model graph...");
    let imported =
        nn_import::import_model(&graph_path, &weights_path).context("model import failed")?;
    eprintln!(
        "  {} graph nodes, {} user input(s)",
        imported.graph.len(),
        imported.num_user_inputs,
    );

    // Phase 3: Select cost model.
    let cost_model = match cost_model_choice {
        CostModelChoice::AppleM4 => nn_dsl::CostModel::apple_m4(),
        CostModelChoice::AppleM4Max => nn_dsl::CostModel::apple_m4_max(),
    };
    let baseline_cost = cost_model.estimate(&baseline_plan);

    // Phase 4: Run the optimizer.
    let budget = std::time::Duration::from_secs(budget_secs);
    eprintln!(
        "Searching PeepholeConfig space (budget: {budget_secs}s)..."
    );
    let opt_result = nn_dsl::optimize_plan_with_cost(&imported.graph, &cost_model, budget)
        .context("optimization search failed")?;

    // Phase 5: Print summary to stdout.
    println!("{}", opt_result.summarize());

    // Also print cost comparison from baseline plan file vs optimized.
    println!();
    println!("Baseline plan (loaded from .nnc):");
    println!("  Dispatches: {baseline_dispatches}");
    println!("  Cost:       {:.1} us", baseline_cost.total_ns / 1e3);
    println!("Optimized plan:");
    println!("  Dispatches: {}", opt_result.dispatch_count);
    println!("  Cost:       {:.1} us", opt_result.best_cost_ns / 1e3);

    if baseline_dispatches > 0 {
        let saved = baseline_dispatches.saturating_sub(opt_result.dispatch_count);
        let pct = (saved as f64 / baseline_dispatches as f64) * 100.0;
        println!(
            "  vs file:    {saved} fewer dispatches ({pct:.1}% reduction)",
        );
    }

    // Phase 6: Save the optimized plan.
    eprintln!();
    eprintln!("Saving optimized plan to {}...", output_path.display());
    nn_dsl::save_plan(&opt_result.plan, output_path).context("failed to save optimized plan")?;
    let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("Saved {} ({} bytes)", output_path.display(), file_size);

    // Phase 7: Optionally persist the PeepholeConfig as a JSON sidecar.
    if persist {
        let config_path = output_path.with_extension("peephole.json");
        eprintln!("Saving PeepholeConfig to {}...", config_path.display());
        nn_dsl::save_peephole_config(&opt_result.config, &config_path)
            .context("failed to save PeepholeConfig")?;
        eprintln!("Done.");
    }

    Ok(())
}

/// Build ordered input tensor list from a name-keyed map.
///
/// Matching strategy:
/// 1. If all model input names are present in the map, use name-based lookup.
/// 2. Otherwise, if the map has exactly `num_inputs` entries, use sorted order.
/// 3. Otherwise, error with a helpful message.
fn build_ordered_inputs(
    tensors: &HashMap<String, nn_core::DynTensor>,
    input_names: &[String],
    num_inputs: usize,
) -> Result<Vec<nn_core::DynTensor>> {
    // Strategy 1: Name-based lookup.
    let all_names_present = input_names.iter().all(|n| tensors.contains_key(n));
    if all_names_present && !input_names.is_empty() {
        return input_names
            .iter()
            .map(|name| {
                tensors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing input tensor: {name}"))
            })
            .collect();
    }

    // Strategy 2: Count-based with sorted order.
    if tensors.len() == num_inputs {
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        return names
            .into_iter()
            .map(|name| {
                tensors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing input tensor: {name}"))
            })
            .collect();
    }

    bail!(
        "cannot match input tensors: model expects {} input(s) named {:?}, \
         but safetensors file contains {} tensor(s) named {:?}",
        num_inputs,
        input_names,
        tensors.len(),
        tensors.keys().collect::<Vec<_>>()
    )
}

/// Print shape, dtype, and summary statistics for a tensor.
fn print_tensor_stats(name: &str, tensor: &nn_core::DynTensor) -> Result<()> {
    let shape = tensor.dims();
    let dtype = tensor.dtype();
    let elem_count: usize = shape.iter().product();

    // Read to CPU for statistics.
    let cpu_tensor = tensor
        .to_device(&nn_core::Device::Cpu)
        .context("failed to read output tensor to CPU")?;
    let values: Vec<f32> = cpu_tensor
        .to_flat_vec::<f32>()
        .context("failed to extract f32 values from output tensor")?;

    let (min, max, sum) = values.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY, 0.0f64),
        |(mn, mx, s), &v| (mn.min(v), mx.max(v), s + f64::from(v)),
    );
    let mean = if elem_count > 0 {
        sum / elem_count as f64
    } else {
        0.0
    };

    eprintln!(
        "  {name}: shape={shape:?}, dtype={dtype:?}, \
         min={min:.6}, max={max:.6}, mean={mean:.6}",
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};

    fn render_subcommand_help(name: &str) -> String {
        let mut cmd = Cli::command();
        let subcommand = cmd
            .find_subcommand_mut(name)
            .unwrap_or_else(|| panic!("expected {name} subcommand"));
        let mut help = Vec::new();
        subcommand
            .write_long_help(&mut help)
            .expect("subcommand help should render");
        String::from_utf8(help).expect("help should be valid UTF-8")
    }

    // ---- Argument Parsing: Device subcommand ----

    #[test]
    fn test_device_subcommand_parses_with_no_arguments() {
        let cli = Cli::try_parse_from(["nn", "device"]);
        assert!(
            cli.is_ok(),
            "device should parse with no arguments: {cli:?}"
        );
        match cli.unwrap().command {
            Commands::Device { json } => assert!(!json, "json should default to false"),
            other => panic!("expected Device, got {other:?}"),
        }
    }

    #[test]
    fn test_device_subcommand_accepts_json_flag() {
        let cli =
            Cli::try_parse_from(["nn", "device", "--json"]).expect("device --json should parse");
        match cli.command {
            Commands::Device { json } => assert!(json, "--json should set json"),
            other => panic!("expected Device, got {other:?}"),
        }
    }

    #[test]
    fn test_device_subcommand_rejects_unknown_flag() {
        // Rule: unsupported flags fail loudly rather than being ignored.
        let cli = Cli::try_parse_from(["nn", "device", "--target", "metal"]);
        assert!(cli.is_err(), "device should reject --target: {cli:?}");
    }

    #[test]
    fn test_device_subcommand_rejects_positional_arguments() {
        let cli = Cli::try_parse_from(["nn", "device", "graph.json"]);
        assert!(
            cli.is_err(),
            "device takes no positional arguments: {cli:?}"
        );
    }

    #[test]
    fn test_device_help_documents_the_zero_input_contract() {
        let help = render_subcommand_help("device");
        assert!(
            help.contains("no input files") || help.contains("zero-input"),
            "device help should say it needs no input files: {help}"
        );
    }

    // ---- Argument Parsing: Convert subcommand ----

    #[test]
    fn test_convert_subcommand_parses_with_graph_and_weights() {
        let cli = Cli::try_parse_from(["nn", "convert", "graph.json", "weights.safetensors"]);
        assert!(
            cli.is_ok(),
            "convert with graph + weights should parse: {cli:?}"
        );
        let cli = cli.unwrap();
        match cli.command {
            Commands::Convert {
                ref graph,
                ref weights,
                ref report_output,
                json,
                ..
            } => {
                assert_eq!(graph, &PathBuf::from("graph.json"));
                assert_eq!(weights.as_deref(), Some(Path::new("weights.safetensors")));
                assert!(report_output.is_none());
                assert!(!json, "json flag should default to false");
            }
            _ => panic!("expected Convert subcommand"),
        }
    }

    #[test]
    fn test_convert_subcommand_graph_only_parses() {
        // weights is optional in the clap definition
        let cli = Cli::try_parse_from(["nn", "convert", "graph.json"]);
        assert!(cli.is_ok(), "convert with graph only should parse: {cli:?}");
        let cli = cli.unwrap();
        match cli.command {
            Commands::Convert { ref weights, .. } => {
                assert!(
                    weights.is_none(),
                    "weights should be None when not provided"
                );
            }
            _ => panic!("expected Convert subcommand"),
        }
    }

    #[test]
    fn test_convert_json_flag() {
        let cli = Cli::try_parse_from([
            "nn",
            "convert",
            "graph.json",
            "weights.safetensors",
            "--json",
        ])
        .expect("convert with --json should parse");
        match cli.command {
            Commands::Convert { json, .. } => {
                assert!(json, "--json flag should be true when provided");
            }
            _ => panic!("expected Convert subcommand"),
        }
    }

    #[test]
    fn test_convert_from_pytorch_flag() {
        let cli = Cli::try_parse_from([
            "nn",
            "convert",
            "model.pt",
            "--from-pytorch",
            "--model-spec",
            "mymod:NnModel",
        ])
        .expect("convert with --from-pytorch should parse");
        match cli.command {
            Commands::Convert {
                from_pytorch,
                model_spec,
                ..
            } => {
                assert!(from_pytorch, "--from-pytorch should be true");
                assert_eq!(model_spec.as_deref(), Some("mymod:NnModel"));
            }
            _ => panic!("expected Convert subcommand"),
        }
    }

    #[test]
    fn test_convert_input_shape_multi_values() {
        let cli = Cli::try_parse_from([
            "nn",
            "convert",
            "model.pt",
            "--from-pytorch",
            "--model-spec",
            "m:M",
            "--input-shape",
            "1",
            "3",
            "224",
            "224",
        ])
        .expect("convert with --input-shape should parse");
        match cli.command {
            Commands::Convert { input_shape, .. } => {
                assert_eq!(input_shape, Some(vec![1, 3, 224, 224]));
            }
            _ => panic!("expected Convert subcommand"),
        }
    }

    #[test]
    fn test_convert_all_options() {
        let cli = Cli::try_parse_from([
            "nn",
            "convert",
            "graph.json",
            "weights.safetensors",
            "--target",
            "metal",
            "--optimize",
            "aggressive",
            "--verify",
            "full",
            "--reference",
            "trace.safetensors",
            "--output",
            "model.nnc",
            "--report-output",
            "model.convert.json",
            "--json",
        ])
        .expect("convert with all options should parse");
        match cli.command {
            Commands::Convert {
                optimize,
                verify,
                reference,
                output,
                report_output,
                json,
                ..
            } => {
                assert!(matches!(optimize, Optimize::Aggressive));
                assert!(matches!(verify, Verify::Full));
                assert_eq!(reference, Some(PathBuf::from("trace.safetensors")));
                assert_eq!(output, Some(PathBuf::from("model.nnc")));
                assert_eq!(report_output, Some(PathBuf::from("model.convert.json")));
                assert!(json);
            }
            _ => panic!("expected Convert subcommand"),
        }
    }

    // ---- Argument Parsing: Compile subcommand ----

    #[test]
    fn test_compile_subcommand_parses() {
        let cli = Cli::try_parse_from([
            "nn",
            "compile",
            "graph.json",
            "weights.safetensors",
            "-o",
            "model.nnc",
        ])
        .expect("compile subcommand should parse");
        match cli.command {
            Commands::Compile {
                ref graph,
                ref weights,
                ref output,
                ref report_output,
                json,
            } => {
                assert_eq!(graph, &PathBuf::from("graph.json"));
                assert_eq!(weights, &PathBuf::from("weights.safetensors"));
                assert_eq!(output, &PathBuf::from("model.nnc"));
                assert!(report_output.is_none());
                assert!(!json, "--json should default to false");
            }
            _ => panic!("expected Compile subcommand"),
        }
    }

    #[test]
    fn test_compile_missing_output_fails() {
        let result = Cli::try_parse_from(["nn", "compile", "graph.json", "weights.safetensors"]);
        assert!(result.is_err(), "compile without -o/--output should fail");
    }

    #[test]
    fn test_compile_subcommand_with_report_output_and_json() {
        let cli = Cli::try_parse_from([
            "nn",
            "compile",
            "graph.json",
            "weights.safetensors",
            "--output",
            "model.nnc",
            "--report-output",
            "model.compile.json",
            "--json",
        ])
        .expect("compile with report flags should parse");
        match cli.command {
            Commands::Compile {
                output,
                report_output,
                json,
                ..
            } => {
                assert_eq!(output, PathBuf::from("model.nnc"));
                assert_eq!(report_output, Some(PathBuf::from("model.compile.json")));
                assert!(json, "--json should be true when provided");
            }
            _ => panic!("expected Compile subcommand"),
        }
    }

    #[test]
    fn test_default_compile_report_path_uses_output_stem() {
        let output = PathBuf::from("artifacts/model.nnc");
        assert_eq!(
            default_compile_report_path(&output),
            PathBuf::from("artifacts/model.compile.json")
        );
    }

    #[test]
    fn test_convert_help_mentions_exported_artifacts_and_not_raw_onnx() {
        let help = render_subcommand_help("convert");
        assert!(
            help.contains("torch.export"),
            "convert help should mention exported torch.export artifacts:\n{help}"
        );
        assert!(
            help.contains("not raw ONNX input"),
            "convert help should avoid claiming raw ONNX support:\n{help}"
        );
    }

    #[test]
    fn test_convert_help_describes_verify_full_without_inline_kani() {
        let help = render_subcommand_help("convert");
        assert!(
            help.contains("fullest report path"),
            "convert help should describe --verify full as a report request:\n{help}"
        );
        assert!(
            help.contains("optional reference parity"),
            "convert help should mention optional reference parity for --verify full:\n{help}"
        );
        assert!(
            help.contains("does not run inline Kani"),
            "convert help should make the missing inline Kani path explicit:\n{help}"
        );
        assert!(
            !help.contains("kernel-safety reporting, and reference parity"),
            "convert help should not claim --verify full includes inline Kani reporting:\n{help}"
        );
    }

    #[test]
    fn test_compile_help_mentions_exported_artifacts_only() {
        let help = render_subcommand_help("compile");
        assert!(
            help.contains("torch.export"),
            "compile help should mention exported torch.export artifacts:\n{help}"
        );
        assert!(
            help.contains("does not accept raw"),
            "compile help should make the exported-artifact-only contract explicit:\n{help}"
        );
        assert!(
            help.contains("compile.json"),
            "compile help should describe the default sibling report path:\n{help}"
        );
        assert!(
            help.contains("persists two artifacts"),
            "compile help should make the .nnc plus report artifact contract explicit:\n{help}"
        );
        assert!(
            help.contains("default verification/report"),
            "compile help should say it follows the default report request:\n{help}"
        );
        assert!(
            help.contains("does not currently expose a separate `--verify` flag"),
            "compile help should make the missing compile-time verify flag explicit:\n{help}"
        );
    }

    #[test]
    fn test_run_help_mentions_compiled_plan_contract() {
        let help = render_subcommand_help("run");
        assert!(
            help.contains("saved `.nnc` plan path produced by"),
            "run help should describe consuming the saved compiled plan path:\n{help}"
        );
        assert!(
            help.contains("skips the trace compilation step"),
            "run help should describe the compiled-plan execution path:\n{help}"
        );
    }

    // ---- Argument Parsing: Run subcommand ----

    #[test]
    fn test_run_subcommand_parses() {
        let cli = Cli::try_parse_from([
            "nn",
            "run",
            "graph.json",
            "weights.safetensors",
            "--input",
            "inputs.safetensors",
        ])
        .expect("run subcommand should parse");
        match cli.command {
            Commands::Run {
                ref graph,
                ref weights,
                ref input,
                ref compiled,
                ref output,
                ..
            } => {
                assert_eq!(graph, &PathBuf::from("graph.json"));
                assert_eq!(weights, &PathBuf::from("weights.safetensors"));
                assert_eq!(input, &PathBuf::from("inputs.safetensors"));
                assert!(compiled.is_none());
                assert!(output.is_none());
            }
            _ => panic!("expected Run subcommand"),
        }
    }

    #[test]
    fn test_run_with_compiled_flag() {
        let cli = Cli::try_parse_from([
            "nn",
            "run",
            "graph.json",
            "weights.safetensors",
            "--input",
            "inputs.safetensors",
            "--compiled",
            "plan.nnc",
        ])
        .expect("run with --compiled should parse");
        match cli.command {
            Commands::Run { compiled, .. } => {
                assert_eq!(compiled, Some(PathBuf::from("plan.nnc")));
            }
            _ => panic!("expected Run subcommand"),
        }
    }

    #[test]
    fn test_run_missing_input_fails() {
        let result = Cli::try_parse_from(["nn", "run", "graph.json", "weights.safetensors"]);
        assert!(result.is_err(), "run without --input should fail");
    }

    // ---- Error Cases ----

    #[test]
    fn test_no_subcommand_fails() {
        let result = Cli::try_parse_from(["nn"]);
        assert!(result.is_err(), "no subcommand should produce an error");
    }

    #[test]
    fn test_unknown_subcommand_fails() {
        let result = Cli::try_parse_from(["nn", "train"]);
        assert!(result.is_err(), "unknown subcommand 'train' should fail");
    }

    #[test]
    fn test_invalid_target_fails() {
        let result = Cli::try_parse_from([
            "nn",
            "convert",
            "graph.json",
            "weights.safetensors",
            "--target",
            "cuda",
        ]);
        assert!(
            result.is_err(),
            "invalid target 'cuda' should fail (only 'metal' is supported)"
        );
    }

    #[test]
    fn test_invalid_optimize_level_fails() {
        let result = Cli::try_parse_from([
            "nn",
            "convert",
            "graph.json",
            "weights.safetensors",
            "--optimize",
            "turbo",
        ]);
        assert!(
            result.is_err(),
            "invalid optimize level 'turbo' should fail"
        );
    }

    #[test]
    fn test_invalid_verify_level_fails() {
        let result = Cli::try_parse_from([
            "nn",
            "convert",
            "graph.json",
            "weights.safetensors",
            "--verify",
            "partial",
        ]);
        assert!(
            result.is_err(),
            "invalid verify level 'partial' should fail"
        );
    }

    // ---- ValueEnum defaults ----

    #[test]
    fn test_convert_defaults_target_metal_optimize_normal_verify_bounds() {
        let cli = Cli::try_parse_from(["nn", "convert", "graph.json", "weights.safetensors"])
            .expect("convert with defaults should parse");
        match cli.command {
            Commands::Convert {
                target,
                optimize,
                verify,
                ..
            } => {
                assert!(
                    matches!(target, Target::Metal),
                    "default target should be Metal"
                );
                assert!(
                    matches!(optimize, Optimize::Normal),
                    "default optimize should be Normal"
                );
                assert!(
                    matches!(verify, Verify::Bounds),
                    "default verify should be Bounds"
                );
            }
            _ => panic!("expected Convert subcommand"),
        }
    }

    // ---- Optimize / Verify From impls ----

    #[test]
    fn test_optimize_to_opt_level_conversion() {
        assert!(matches!(
            nn_import::OptLevel::from(Optimize::None),
            nn_import::OptLevel::None
        ));
        assert!(matches!(
            nn_import::OptLevel::from(Optimize::Normal),
            nn_import::OptLevel::Full
        ));
        assert!(matches!(
            nn_import::OptLevel::from(Optimize::Aggressive),
            nn_import::OptLevel::Aggressive
        ));
    }

    #[test]
    fn test_verify_to_verify_level_conversion() {
        assert!(matches!(
            nn_import::VerifyLevel::from(Verify::None),
            nn_import::VerifyLevel::None
        ));
        assert!(matches!(
            nn_import::VerifyLevel::from(Verify::Bounds),
            nn_import::VerifyLevel::Bounds
        ));
        assert!(matches!(
            nn_import::VerifyLevel::from(Verify::Full),
            nn_import::VerifyLevel::Full
        ));
    }

    // ---- build_ordered_inputs ----

    #[test]
    fn test_build_ordered_inputs_name_based_lookup() {
        let mut tensors = HashMap::new();
        let t_a = nn_core::DynTensor::zeros(&[2, 3], nn_core::DType::F32, &nn_core::Device::Cpu)
            .unwrap();
        let t_b = nn_core::DynTensor::zeros(&[4, 5], nn_core::DType::F32, &nn_core::Device::Cpu)
            .unwrap();
        tensors.insert("input_a".to_string(), t_a);
        tensors.insert("input_b".to_string(), t_b);

        let names = vec!["input_b".to_string(), "input_a".to_string()];
        let result = build_ordered_inputs(&tensors, &names, 2).unwrap();
        assert_eq!(result.len(), 2);
        // First should be input_b (shape [4,5]), second input_a (shape [2,3])
        assert_eq!(result[0].dims(), &[4, 5]);
        assert_eq!(result[1].dims(), &[2, 3]);
    }

    #[test]
    fn test_build_ordered_inputs_count_based_sorted() {
        let mut tensors = HashMap::new();
        let t_z = nn_core::DynTensor::zeros(&[1, 1], nn_core::DType::F32, &nn_core::Device::Cpu)
            .unwrap();
        let t_a = nn_core::DynTensor::zeros(&[2, 2], nn_core::DType::F32, &nn_core::Device::Cpu)
            .unwrap();
        tensors.insert("z_tensor".to_string(), t_z);
        tensors.insert("a_tensor".to_string(), t_a);

        // Names don't match, but count matches — falls through to sorted order
        let names = vec!["x".to_string(), "y".to_string()];
        let result = build_ordered_inputs(&tensors, &names, 2).unwrap();
        assert_eq!(result.len(), 2);
        // Sorted by name: "a_tensor" first (shape [2,2]), "z_tensor" second (shape [1,1])
        assert_eq!(result[0].dims(), &[2, 2]);
        assert_eq!(result[1].dims(), &[1, 1]);
    }

    #[test]
    fn test_build_ordered_inputs_count_mismatch_fails() {
        let mut tensors = HashMap::new();
        let t =
            nn_core::DynTensor::zeros(&[1], nn_core::DType::F32, &nn_core::Device::Cpu).unwrap();
        tensors.insert("only_one".to_string(), t);

        let names = vec!["x".to_string(), "y".to_string()];
        let result = build_ordered_inputs(&tensors, &names, 2);
        assert!(
            result.is_err(),
            "should fail when tensor count (1) != expected inputs (2)"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot match input tensors"),
            "error message should explain the mismatch: {err_msg}"
        );
    }

    // ---- Argument Parsing: Optimize subcommand ----

    #[test]
    fn test_optimize_subcommand_parses_with_defaults() {
        let cli = Cli::try_parse_from([
            "nn",
            "optimize",
            "model.nnc",
            "--graph",
            "graph.json",
            "--weights",
            "weights.safetensors",
        ])
        .expect("optimize with defaults should parse");
        match cli.command {
            Commands::Optimize {
                ref plan,
                ref graph,
                ref weights,
                budget,
                cost_model,
                ref output,
                persist,
            } => {
                assert_eq!(plan, &PathBuf::from("model.nnc"));
                assert_eq!(graph, &PathBuf::from("graph.json"));
                assert_eq!(weights, &PathBuf::from("weights.safetensors"));
                assert_eq!(budget, 30, "default budget should be 30s");
                assert!(
                    matches!(cost_model, CostModelChoice::AppleM4),
                    "default cost model should be apple-m4"
                );
                assert!(output.is_none(), "output should default to None");
                assert!(!persist, "persist should default to false");
            }
            _ => panic!("expected Optimize subcommand"),
        }
    }

    #[test]
    fn test_optimize_subcommand_with_budget() {
        let cli = Cli::try_parse_from([
            "nn",
            "optimize",
            "model.nnc",
            "--graph",
            "g.json",
            "--weights",
            "w.safetensors",
            "--budget",
            "60",
        ])
        .expect("optimize with --budget should parse");
        match cli.command {
            Commands::Optimize { budget, .. } => {
                assert_eq!(budget, 60);
            }
            _ => panic!("expected Optimize subcommand"),
        }
    }

    #[test]
    fn test_optimize_subcommand_with_cost_model() {
        let cli = Cli::try_parse_from([
            "nn",
            "optimize",
            "model.nnc",
            "--graph",
            "g.json",
            "--weights",
            "w.safetensors",
            "--cost-model",
            "apple-m4-max",
        ])
        .expect("optimize with --cost-model should parse");
        match cli.command {
            Commands::Optimize { cost_model, .. } => {
                assert!(matches!(cost_model, CostModelChoice::AppleM4Max));
            }
            _ => panic!("expected Optimize subcommand"),
        }
    }

    #[test]
    fn test_optimize_subcommand_with_output() {
        let cli = Cli::try_parse_from([
            "nn",
            "optimize",
            "model.nnc",
            "--graph",
            "g.json",
            "--weights",
            "w.safetensors",
            "--output",
            "optimized.nnc",
        ])
        .expect("optimize with --output should parse");
        match cli.command {
            Commands::Optimize { output, .. } => {
                assert_eq!(output, Some(PathBuf::from("optimized.nnc")));
            }
            _ => panic!("expected Optimize subcommand"),
        }
    }

    #[test]
    fn test_optimize_subcommand_with_persist() {
        let cli = Cli::try_parse_from([
            "nn",
            "optimize",
            "model.nnc",
            "--graph",
            "g.json",
            "--weights",
            "w.safetensors",
            "--persist",
        ])
        .expect("optimize with --persist should parse");
        match cli.command {
            Commands::Optimize { persist, .. } => {
                assert!(persist, "--persist flag should be true");
            }
            _ => panic!("expected Optimize subcommand"),
        }
    }

    #[test]
    fn test_optimize_subcommand_all_options() {
        let cli = Cli::try_parse_from([
            "nn",
            "optimize",
            "kokoro.nnc",
            "--graph",
            "kokoro_graph.json",
            "--weights",
            "kokoro_weights.safetensors",
            "--budget",
            "120",
            "--cost-model",
            "apple-m4-max",
            "--output",
            "kokoro_optimized.nnc",
            "--persist",
        ])
        .expect("optimize with all options should parse");
        match cli.command {
            Commands::Optimize {
                ref plan,
                ref graph,
                ref weights,
                budget,
                cost_model,
                ref output,
                persist,
            } => {
                assert_eq!(plan, &PathBuf::from("kokoro.nnc"));
                assert_eq!(graph, &PathBuf::from("kokoro_graph.json"));
                assert_eq!(weights, &PathBuf::from("kokoro_weights.safetensors"));
                assert_eq!(budget, 120);
                assert!(matches!(cost_model, CostModelChoice::AppleM4Max));
                assert_eq!(output.as_deref(), Some(Path::new("kokoro_optimized.nnc")));
                assert!(persist);
            }
            _ => panic!("expected Optimize subcommand"),
        }
    }

    #[test]
    fn test_optimize_subcommand_missing_plan_fails() {
        let result = Cli::try_parse_from([
            "nn",
            "optimize",
            "--graph",
            "g.json",
            "--weights",
            "w.safetensors",
        ]);
        assert!(
            result.is_err(),
            "optimize without plan positional arg should fail"
        );
    }

    #[test]
    fn test_optimize_subcommand_missing_graph_fails() {
        let result = Cli::try_parse_from([
            "nn",
            "optimize",
            "model.nnc",
            "--weights",
            "w.safetensors",
        ]);
        assert!(result.is_err(), "optimize without --graph should fail");
    }

    #[test]
    fn test_optimize_subcommand_missing_weights_fails() {
        let result = Cli::try_parse_from(["nn", "optimize", "model.nnc", "--graph", "g.json"]);
        assert!(result.is_err(), "optimize without --weights should fail");
    }

    #[test]
    fn test_optimize_subcommand_invalid_cost_model_fails() {
        let result = Cli::try_parse_from([
            "nn",
            "optimize",
            "model.nnc",
            "--graph",
            "g.json",
            "--weights",
            "w.safetensors",
            "--cost-model",
            "nvidia-h100",
        ]);
        assert!(
            result.is_err(),
            "invalid cost model 'nvidia-h100' should fail"
        );
    }

    // ---- Help flag does not panic ----

    #[test]
    fn test_help_flag_produces_error_not_panic() {
        // clap treats --help as an error (it exits), but it should not panic.
        let result = Cli::try_parse_from(["nn", "--help"]);
        assert!(
            result.is_err(),
            "--help should produce a clap error (display help)"
        );
    }

    #[test]
    fn test_version_flag_produces_error_not_panic() {
        let result = Cli::try_parse_from(["nn", "--version"]);
        assert!(
            result.is_err(),
            "--version should produce a clap error (display version)"
        );
    }
}
