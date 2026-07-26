// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GGUF file header parsing.

use std::io::Read;

use crate::error::GgufError;

/// GGUF magic number: "GGUF" in little-endian.
pub(crate) const GGUF_MAGIC: u32 = 0x4647_5547;

/// Maximum allowed string length (16 MiB). Prevents OOM from crafted GGUF files.
pub(crate) const MAX_STRING_LENGTH: u64 = 16 * 1024 * 1024;

/// Maximum allowed byte size for a single tensor allocation (8 GiB).
///
/// Even the largest single tensors in production models (e.g., Llama-405B
/// embedding at 128k vocab x 16k hidden) are under 8 GiB in f32. This cap
/// prevents a crafted GGUF file from triggering extreme memory allocation.
pub(crate) const MAX_TENSOR_BYTE_SIZE: u64 = 8 * 1024 * 1024 * 1024;

/// Maximum allowed tensor count. Largest models have ~1000 tensors.
pub(crate) const MAX_TENSOR_COUNT: u64 = 100_000;

/// Maximum allowed metadata key-value count.
pub(crate) const MAX_METADATA_KV_COUNT: u64 = 100_000;

/// Maximum allowed metadata array length.
pub(crate) const MAX_ARRAY_LENGTH: u64 = 10_000_000;

/// Maximum tensor dimensions. GGUF spec allows up to 4, we allow 8 for safety.
pub(crate) const MAX_DIMENSIONS: u32 = 8;

/// Parsed GGUF file header.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// File format version (must be 3).
    pub version: u32,
    /// Number of metadata key-value pairs.
    pub metadata_kv_count: u64,
    /// Number of tensors in the file.
    pub tensor_count: u64,
}

impl GgufHeader {
    /// Parse the GGUF header from a reader.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, GgufError> {
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic { found: magic });
        }

        let version = read_u32(reader)?;
        if version != 3 {
            return Err(GgufError::UnsupportedVersion { version });
        }

        let tensor_count = read_u64(reader)?;
        let metadata_kv_count = read_u64(reader)?;

        if tensor_count > MAX_TENSOR_COUNT {
            return Err(GgufError::TensorCountExceeded {
                count: tensor_count,
                max: MAX_TENSOR_COUNT,
            });
        }
        if metadata_kv_count > MAX_METADATA_KV_COUNT {
            return Err(GgufError::MetadataCountExceeded {
                count: metadata_kv_count,
                max: MAX_METADATA_KV_COUNT,
            });
        }

        Ok(Self {
            version,
            metadata_kv_count,
            tensor_count,
        })
    }
}

/// Read a little-endian u32.
pub(crate) fn read_u32<R: Read>(reader: &mut R) -> Result<u32, GgufError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian u64.
pub(crate) fn read_u64<R: Read>(reader: &mut R) -> Result<u64, GgufError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read a little-endian i64.
pub(crate) fn read_i64<R: Read>(reader: &mut R) -> Result<i64, GgufError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

/// Read a little-endian f32.
pub(crate) fn read_f32<R: Read>(reader: &mut R) -> Result<f32, GgufError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// Read a little-endian f64.
pub(crate) fn read_f64<R: Read>(reader: &mut R) -> Result<f64, GgufError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

/// Read a GGUF string (u64 length prefix + UTF-8 bytes).
pub(crate) fn read_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let len = read_u64(reader)?;
    if len > MAX_STRING_LENGTH {
        return Err(GgufError::StringLengthExceeded {
            len,
            max: MAX_STRING_LENGTH,
        });
    }
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8 { offset: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_header() {
        // GGUF magic + version 3 + 2 tensors + 1 metadata
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count
        let header = GgufHeader::read_from(&mut &data[..]).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 2);
        assert_eq!(header.metadata_kv_count, 1);
    }

    #[test]
    fn test_invalid_magic() {
        let data = 0xDEAD_BEEFu32.to_le_bytes();
        let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
        assert!(matches!(
            err,
            GgufError::InvalidMagic { found: 0xDEAD_BEEF }
        ));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // version 2
        let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion { version: 2 }));
    }
}
