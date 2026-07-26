// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for the verification runner: source path resolution,
//! workspace root computation, and layer bounds extraction.

use std::path::{Path, PathBuf};

use nn_verify::{extract_layer_bounds, kernel_to_graph, LayerBoundRecord, ScalarInputBounds};

/// Workspace root anchored at compile time via `CARGO_MANIFEST_DIR`.
///
/// `CARGO_MANIFEST_DIR` for the `nn-verify` crate is `crates/nn-verify`,
/// so the workspace root is two directories up. This matches the logic in
/// `main.rs::status_path()`.
pub(super) fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Resolve the source file path for a kernel config name (#1327).
///
/// Maps config names like `"snake"`, `"adain_wide"`, `"fusion_adain_snake"`
/// to their kernel source file under `crates/nn-dsl/src/`. Returns `None`
/// if the mapping cannot be determined (best-effort).
///
/// Paths are anchored to the workspace root via `CARGO_MANIFEST_DIR` so
/// `path.exists()` succeeds regardless of the process's current working
/// directory. Previously used relative paths, causing `source_hash` to be
/// absent in all 31 proof certificates.
pub(super) fn source_path_for_config(config_name: &str) -> Option<PathBuf> {
    let base = config_name.strip_prefix("fusion_").unwrap_or(config_name);

    let filename = match base {
        s if s.starts_with("snake") => "snake.rs",
        s if s.starts_with("silu_mul") => "silu_mul.rs",
        s if s.starts_with("adain_snake") => "adain.rs",
        s if s.starts_with("adain") => "adain.rs",
        s if s.starts_with("gelu") => "gelu.rs",
        s if s.starts_with("sigmoid") => "sigmoid.rs",
        s if s.starts_with("relu") => "relu.rs",
        s if s.starts_with("tanh") => "tanh_kernel.rs",
        s if s.starts_with("rope") => "rope.rs",
        s if s.starts_with("layer_norm") => "layer_norm.rs",
        s if s.starts_with("rms_norm") => "rms_norm.rs",
        s if s.starts_with("instance_norm") => "instance_norm.rs",
        _ => return None,
    };

    let path = workspace_root().join(format!("crates/nn-dsl/src/{filename}"));
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Source path for model-level verification configs (#1327, #1696 AC7).
pub(super) fn source_path_for_model(model_name: &str) -> Option<PathBuf> {
    let root = workspace_root();
    let path = match model_name {
        "silero_vad_full" => root.join("crates/nn-verify/examples/verify_all/model_configs.rs"),
        "htdemucs_full" | "whisper_full" | "qwen3_full" | "kokoro_decoder" => {
            root.join("crates/nn-verify/examples/verify_all/model_configs_extra.rs")
        }
        _ => return None,
    };
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Rebuild the NY graph for a kernel and extract per-layer bounds.
///
/// Returns `Some(layer_bounds)` on success, `None` on any error. Errors
/// are logged to stderr but do not prevent certificate generation.
pub(super) fn extract_layer_bounds_for_kernel(
    kernel: &nn_dsl::ir::KernelDef,
    constant_params: &[f32],
    bounds: ScalarInputBounds,
) -> Option<Vec<LayerBoundRecord>> {
    let graph = match kernel_to_graph(kernel, constant_params) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  layer_bounds: graph build failed: {e}");
            return None;
        }
    };
    let input_bounds = match nn_verify::scalar_input_bounds(bounds.lower(), bounds.upper()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  layer_bounds: input bounds failed: {e}");
            return None;
        }
    };
    match extract_layer_bounds(&graph, &input_bounds) {
        Ok(records) => Some(records),
        Err(e) => {
            eprintln!("  layer_bounds: extraction failed: {e}");
            None
        }
    }
}
