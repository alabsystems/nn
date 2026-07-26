// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Single function call API: `load_kokoro()` -> `CompiledKokoro`.
//!
//! ```rust,ignore
//! use nn_import::load_kokoro;
//! use nn_metal::cache::PipelineCache;
//!
//! let cache = PipelineCache::new()?;
//! let mut kokoro = load_kokoro("kokoro_v1_0.safetensors")?;
//! let (audio, cert) = kokoro.synthesize(&input_ids, &style, 1.0, &cache)?;
//! assert!(cert.overall_passed);
//! ```
//!
//! Part of #2465, #2218.

use std::path::Path;

use nn_metal::compiled_kokoro::CompiledKokoro;

use crate::kokoro_weights::validate_kokoro_safetensors;
use crate::ImportError;

/// Load Kokoro TTS model from a safetensors file.
///
/// Validates that the safetensors file contains all required Kokoro weight
/// groups, then loads and returns a [`CompiledKokoro`] pipeline ready for
/// GPU-accelerated synthesis.
///
/// # Weight Validation
///
/// Checks that all 6 required weight prefixes are present:
/// `plbert.`, `bert_encoder.`, `text_encoder.`, `prosody_predictor.`,
/// `predictor.`, `decoder.`
///
/// # Safety
///
/// Uses mmap for zero-copy weight loading. The safetensors file must not be
/// modified or truncated while the returned `CompiledKokoro` is alive.
///
/// # Example
///
/// ```rust,ignore
/// let mut kokoro = nn_import::load_kokoro("kokoro_v1_0.safetensors")?;
/// let (audio, cert) = kokoro.synthesize(&input_ids, &style, 1.0, &cache)?;
/// assert!(cert.overall_passed);
/// ```
pub fn load_kokoro(path: impl AsRef<Path>) -> Result<CompiledKokoro, ImportError> {
    let path = path.as_ref();

    // Step 1: Validate safetensors keys before loading the full model.
    let keys = read_safetensors_keys(path)?;
    let _mapped = validate_kokoro_safetensors(&keys)?;

    // Step 2: Load via CompiledKokoro::load (mmap, zero-copy).
    // SAFETY: standard mmap contract — file must not be modified while alive.
    let kokoro =
        unsafe { CompiledKokoro::load(path) }.map_err(|e| ImportError::CompiledModelLoad {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;

    Ok(kokoro)
}

/// Read safetensors metadata to extract key names without loading tensors.
///
/// Header-only parsing: reads the 8-byte length prefix + JSON header
/// (typically <100 KB) instead of the entire file (~300 MB for Kokoro).
/// Fix for #2488.
fn read_safetensors_keys(path: &Path) -> Result<Vec<String>, ImportError> {
    use std::io::Read;

    let io_err = |e: std::io::Error| ImportError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    };

    let mut file = std::fs::File::open(path).map_err(io_err)?;

    // Safetensors format: [8 bytes: header_len as u64 LE] [header_len bytes: JSON] [tensor data]
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(io_err)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    // Sanity check: header should be well under 1 MB for any model.
    const MAX_HEADER: usize = 10 * 1024 * 1024; // 10 MB upper bound
    if header_len > MAX_HEADER {
        return Err(ImportError::Io {
            path: path.display().to_string(),
            detail: format!("safetensors header too large: {header_len} bytes"),
        });
    }

    // Read only the header portion (length prefix + JSON body).
    let mut buf = vec![0u8; 8 + header_len];
    buf[..8].copy_from_slice(&len_buf);
    file.read_exact(&mut buf[8..]).map_err(io_err)?;

    let (_, metadata) =
        safetensors::SafeTensors::read_metadata(&buf).map_err(|e| ImportError::Io {
            path: path.display().to_string(),
            detail: format!("safetensors header parse: {e}"),
        })?;

    Ok(metadata.tensors().into_keys().collect())
}

#[cfg(test)]
#[path = "kokoro_load_tests.rs"]
mod tests;
