// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Workspace source hash computation for moonshot certificates.
//!
//! Computes a deterministic SHA-256 hash over the verification-relevant Rust
//! source files in the workspace. This hash is embedded in the
//! [`MoonshotCertificate`] and verified by [`validate_certificate()`] to ensure
//! the certificate's claims correspond to the actual codebase.
//!
//! # Algorithm
//!
//! 1. Recursively walk `crates/` collecting all `.rs` files.
//! 2. Sort file paths lexicographically (for determinism across platforms).
//! 3. For each file, hash `relative_path + "\n" + file_contents`.
//! 4. Hash the concatenation of all per-file hashes into a single composite.
//!
//! The path prefix is included so renaming a file changes the hash even if
//! contents are identical.

use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Error type for source hash computation.
#[derive(Debug)]
pub enum SourceHashError {
    /// The `crates/` directory does not exist at the given repo root.
    CratesDirNotFound(PathBuf),
    /// I/O error reading a source file.
    Io(std::io::Error),
}

impl std::fmt::Display for SourceHashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CratesDirNotFound(p) => {
                write!(f, "crates/ directory not found at {}", p.display())
            }
            Self::Io(e) => write!(f, "I/O error reading source file: {e}"),
        }
    }
}

impl std::error::Error for SourceHashError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SourceHashError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Compute the deterministic SHA-256 hash of the workspace source code.
///
/// Walks `repo_root/crates/` for all `.rs` files, sorts them by relative
/// path, and computes a composite hash.
///
/// # Errors
///
/// Returns [`SourceHashError::CratesDirNotFound`] if `repo_root/crates/`
/// does not exist, or [`SourceHashError::Io`] if any file cannot be read.
pub fn compute_workspace_source_hash(repo_root: &Path) -> Result<String, SourceHashError> {
    let crates_dir = repo_root.join("crates");
    if !crates_dir.is_dir() {
        return Err(SourceHashError::CratesDirNotFound(crates_dir));
    }

    let mut rs_files = walk_rs_files(&crates_dir)?;
    // Sort for determinism — file system traversal order varies by OS/FS.
    rs_files.sort();

    let mut composite_hasher = Sha256::new();

    for file_path in &rs_files {
        // Use the path relative to repo_root for the hash input so the hash
        // is independent of the absolute checkout location.
        let relative = file_path.strip_prefix(repo_root).unwrap_or(file_path);

        let mut file_hasher = Sha256::new();
        // Include relative path as prefix so renaming changes the hash.
        file_hasher.update(relative.to_string_lossy().as_bytes());
        file_hasher.update(b"\n");

        // Read file contents in 8 KiB chunks (matches nn-verify pattern).
        let mut file = std::fs::File::open(file_path)?;
        let mut buf = [0u8; 8192];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file_hasher.update(&buf[..n]);
        }

        let file_hash = file_hasher.finalize();
        composite_hasher.update(file_hash);
    }

    Ok(composite_hasher
        .finalize()
        .iter()
        .fold(String::new(), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        }))
}

/// Validate that a string looks like a SHA-256 hex digest (64 lowercase hex).
#[must_use]
pub fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Recursively walk a directory collecting `.rs` file paths.
fn walk_rs_files(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut result = Vec::new();
    walk_rs_files_inner(dir, &mut result)?;
    Ok(result)
}

fn walk_rs_files_inner(dir: &Path, result: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files_inner(&path, result)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            result.push(path);
        }
    }
    Ok(())
}
