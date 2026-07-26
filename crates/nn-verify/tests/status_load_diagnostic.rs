// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic test: verify all per-model status files load without error.
//! Catches silent deserialization failures hidden by `unwrap_or_default()`.

use std::path::Path;

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn all_status_files_load_without_error() {
    let ws = workspace_root();
    let mut total_kernels = 0;

    for &model in nn_verify::MODEL_CATEGORIES {
        let path = nn_verify::model_status_path(&ws, model);
        if !path.exists() {
            eprintln!("[{model}] status file not found: {}", path.display());
            continue;
        }

        match nn_verify::VerifyStatus::load(&path) {
            Ok(status) => {
                let count = status.kernel_count();
                eprintln!("[{model}] OK: {count} kernels");
                total_kernels += count;
            }
            Err(e) => {
                panic!(
                    "[{model}] DESERIALIZATION FAILED: {e}\n  path: {}",
                    path.display()
                );
            }
        }
    }

    eprintln!("Total kernels across all models: {total_kernels}");
    assert!(
        total_kernels >= 90,
        "Expected >= 90 total kernels, found {total_kernels}. \
         If a status file fails to load, the error is printed above."
    );
}
