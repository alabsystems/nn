// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Format versioning for `.nnc` compiled model files.
//!
//! Every `.nnc` file begins with a fixed-size [`NncHeader`] that encodes
//! magic bytes, a format version, and a creation timestamp. This enables
//! safe forward/backward compatibility: older nn versions detect
//! incompatible files, and newer versions can migrate old formats.
//!
//! # Wire format (16 bytes, little-endian)
//!
//! | Offset | Size | Field         |
//! |--------|------|---------------|
//! | 0      | 4    | magic `NNC\0` |
//! | 4      | 4    | version (u32) |
//! | 8      | 8    | created_at (u64, Unix epoch secs) |

/// Magic bytes identifying a `.nnc` file.
pub(crate) const NNC_MAGIC: [u8; 4] = *b"NNC\0";

/// Current format version.
///
/// Bump this when the serialized layout changes in a backward-incompatible
/// way. The loader rejects files with `version > NNC_CURRENT_VERSION`.
pub(crate) const NNC_CURRENT_VERSION: u32 = 1;

/// Size of the header in bytes (4 magic + 4 version + 8 created_at).
pub(crate) const NNC_HEADER_SIZE: usize = 16;

/// Minimum compatible format version that this loader can read.
///
/// Files with `version < NNC_MIN_VERSION` are rejected.
pub(crate) const NNC_MIN_VERSION: u32 = 1;

/// Header prepended to every `.nnc` file.
///
/// Stored as 16 little-endian bytes before the JSON payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NncHeader {
    /// Magic bytes — must equal [`NNC_MAGIC`].
    pub(crate) magic: [u8; 4],
    /// Format version.
    pub(crate) version: u32,
    /// Unix timestamp (seconds since epoch) when the file was created.
    pub(crate) created_at: u64,
}

/// Errors specific to `.nnc` header validation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NncError {
    /// The file does not start with the expected `NNC\0` magic bytes.
    #[error("invalid .nnc magic bytes: expected {:?}, got {got:?}", NNC_MAGIC)]
    InvalidMagic { got: [u8; 4] },

    /// The file was written by a newer version of nn that this loader
    /// cannot read.
    #[error(
        ".nnc version {version} is newer than the maximum supported version {max}; \
         upgrade nn to read this file"
    )]
    VersionTooNew { version: u32, max: u32 },

    /// The file was written by an older version of nn that is no longer
    /// supported by this loader.
    #[error(
        ".nnc version {version} is older than the minimum supported version {min}; \
         re-export the model with a newer nn"
    )]
    VersionTooOld { version: u32, min: u32 },

    /// The header is truncated (file too short).
    #[error(".nnc file too short: expected at least {expected} header bytes, got {got}")]
    Truncated { expected: usize, got: usize },
}

impl NncHeader {
    /// Create a new header with the current version and the given timestamp.
    pub(crate) fn new(created_at: u64) -> Self {
        Self {
            magic: NNC_MAGIC,
            version: NNC_CURRENT_VERSION,
            created_at,
        }
    }

    /// Create a header using the current wall-clock time.
    pub(crate) fn now() -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self::new(created_at)
    }

    /// Serialize the header to a 16-byte little-endian buffer.
    pub(crate) fn to_bytes(&self) -> [u8; NNC_HEADER_SIZE] {
        let mut buf = [0u8; NNC_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..16].copy_from_slice(&self.created_at.to_le_bytes());
        buf
    }

    /// Deserialize a header from a byte slice.
    ///
    /// Returns [`NncError::Truncated`] if `data` is shorter than
    /// [`NNC_HEADER_SIZE`].
    pub(crate) fn from_bytes(data: &[u8]) -> Result<Self, NncError> {
        if data.len() < NNC_HEADER_SIZE {
            return Err(NncError::Truncated {
                expected: NNC_HEADER_SIZE,
                got: data.len(),
            });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&data[0..4]);
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let created_at = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);
        Ok(Self {
            magic,
            version,
            created_at,
        })
    }

    /// Validate magic bytes and version compatibility.
    ///
    /// Returns `Ok(())` if the header is valid and the version is within
    /// the supported range `[NNC_MIN_VERSION, NNC_CURRENT_VERSION]`.
    pub(crate) fn validate(&self) -> Result<(), NncError> {
        if self.magic != NNC_MAGIC {
            return Err(NncError::InvalidMagic { got: self.magic });
        }
        if self.version > NNC_CURRENT_VERSION {
            return Err(NncError::VersionTooNew {
                version: self.version,
                max: NNC_CURRENT_VERSION,
            });
        }
        if self.version < NNC_MIN_VERSION {
            return Err(NncError::VersionTooOld {
                version: self.version,
                min: NNC_MIN_VERSION,
            });
        }
        Ok(())
    }

    /// Check whether a given version is compatible with this loader.
    pub(crate) fn is_compatible(version: u32) -> bool {
        version >= NNC_MIN_VERSION && version <= NNC_CURRENT_VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = NncHeader::new(1_700_000_000);
        let bytes = header.to_bytes();
        let restored = NncHeader::from_bytes(&bytes).expect("from_bytes should succeed");
        assert_eq!(header, restored);
    }

    #[test]
    fn test_header_now_has_valid_magic_and_version() {
        let header = NncHeader::now();
        assert_eq!(header.magic, NNC_MAGIC);
        assert_eq!(header.version, NNC_CURRENT_VERSION);
        assert!(header.created_at > 0, "timestamp should be non-zero");
    }

    #[test]
    fn test_validate_good_header() {
        let header = NncHeader::new(12345);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_magic() {
        let header = NncHeader {
            magic: *b"BAAD",
            version: NNC_CURRENT_VERSION,
            created_at: 0,
        };
        let err = header.validate().unwrap_err();
        assert!(
            matches!(err, NncError::InvalidMagic { got } if got == *b"BAAD"),
            "expected InvalidMagic, got {err:?}"
        );
    }

    #[test]
    fn test_validate_future_version() {
        let header = NncHeader {
            magic: NNC_MAGIC,
            version: NNC_CURRENT_VERSION + 1,
            created_at: 0,
        };
        let err = header.validate().unwrap_err();
        assert!(
            matches!(err, NncError::VersionTooNew { version, max }
                if version == NNC_CURRENT_VERSION + 1 && max == NNC_CURRENT_VERSION),
            "expected VersionTooNew, got {err:?}"
        );
    }

    #[test]
    fn test_validate_ancient_version() {
        let header = NncHeader {
            magic: NNC_MAGIC,
            version: 0,
            created_at: 0,
        };
        let err = header.validate().unwrap_err();
        assert!(
            matches!(err, NncError::VersionTooOld { version: 0, min: 1 }),
            "expected VersionTooOld, got {err:?}"
        );
    }

    #[test]
    fn test_from_bytes_truncated() {
        let err = NncHeader::from_bytes(&[0u8; 10]).unwrap_err();
        assert!(
            matches!(
                err,
                NncError::Truncated {
                    expected: 16,
                    got: 10
                }
            ),
            "expected Truncated, got {err:?}"
        );
    }

    #[test]
    fn test_from_bytes_empty() {
        let err = NncHeader::from_bytes(&[]).unwrap_err();
        assert!(
            matches!(
                err,
                NncError::Truncated {
                    expected: 16,
                    got: 0
                }
            ),
            "expected Truncated, got {err:?}"
        );
    }

    #[test]
    fn test_is_compatible_current() {
        assert!(NncHeader::is_compatible(NNC_CURRENT_VERSION));
    }

    #[test]
    fn test_is_compatible_future() {
        assert!(!NncHeader::is_compatible(NNC_CURRENT_VERSION + 1));
    }

    #[test]
    fn test_is_compatible_zero() {
        assert!(!NncHeader::is_compatible(0));
    }

    #[test]
    fn test_header_bytes_layout() {
        let header = NncHeader::new(0x0102_0304_0506_0708);
        let bytes = header.to_bytes();
        // Magic bytes
        assert_eq!(&bytes[0..4], b"NNC\0");
        // Version = 1 in little-endian
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
        // Timestamp in little-endian
        assert_eq!(&bytes[8..16], &0x0102_0304_0506_0708u64.to_le_bytes());
    }

    #[test]
    fn test_corrupt_magic_mid_bytes() {
        let mut bytes = NncHeader::new(100).to_bytes();
        bytes[2] = 0xFF; // corrupt third byte of magic
        let header = NncHeader::from_bytes(&bytes).expect("parse succeeds");
        let err = header.validate().unwrap_err();
        assert!(
            matches!(err, NncError::InvalidMagic { .. }),
            "expected InvalidMagic, got {err:?}"
        );
    }
}
