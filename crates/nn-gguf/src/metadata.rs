// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GGUF metadata key-value parsing.

use std::collections::HashMap;
use std::io::Read;

use crate::error::GgufError;
use crate::header::{
    read_f32, read_f64, read_i64, read_string, read_u32, read_u64, MAX_ARRAY_LENGTH,
    MAX_METADATA_KV_COUNT,
};

/// Maximum number of elements in a metadata array value.
///
/// A metadata value in a GGUF file.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
}

impl GgufMetadataValue {
    /// Try to extract as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as u64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) => Some(*v),
            // Allow widening from f32 for convenience (GGUF files store
            // rope.freq_base as f32 in some producers, f64 in others).
            Self::F32(v) => Some(f64::from(*v)),
            _ => None,
        }
    }
}

/// Collection of GGUF metadata key-value pairs.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub entries: HashMap<String, GgufMetadataValue>,
}

impl GgufMetadata {
    /// Parse `count` metadata key-value pairs from the reader.
    pub fn read_from<R: Read>(reader: &mut R, count: u64) -> Result<Self, GgufError> {
        let capped = count.min(MAX_METADATA_KV_COUNT) as usize;
        let mut entries = HashMap::with_capacity(capped);
        for _ in 0..count {
            let key = read_string(reader)?;
            let value = read_metadata_value(reader)?;
            entries.insert(key, value);
        }
        Ok(Self { entries })
    }

    /// Get a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.entries.get(key)
    }

    /// Get a string metadata value, returning None if missing or wrong type.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(GgufMetadataValue::as_str)
    }

    /// Get a u32 metadata value, returning None if missing or wrong type.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key).and_then(GgufMetadataValue::as_u32)
    }

    /// Get a u64 metadata value, returning None if missing or wrong type.
    ///
    /// Also accepts U32 values (widening to u64) since some GGUF producers
    /// store counts as u32 while the spec allows u64.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| match v {
            GgufMetadataValue::U64(n) => Some(*n),
            GgufMetadataValue::U32(n) => Some(u64::from(*n)),
            _ => None,
        })
    }

    /// Get an f64 metadata value, returning None if missing or wrong type.
    ///
    /// Also accepts F32 values (widening to f64).
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(GgufMetadataValue::as_f64)
    }
}

/// Read a single metadata value based on its type tag.
fn read_metadata_value<R: Read>(reader: &mut R) -> Result<GgufMetadataValue, GgufError> {
    let type_id = read_u32(reader)?;
    read_typed_value(reader, type_id)
}

/// Read a value of a known type (used for array elements and top-level values).
fn read_typed_value<R: Read>(reader: &mut R, type_id: u32) -> Result<GgufMetadataValue, GgufError> {
    match type_id {
        0 => {
            // UINT8
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::U8(buf[0]))
        }
        1 => {
            // INT8
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::I8(buf[0] as i8))
        }
        2 => {
            // UINT16
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::U16(u16::from_le_bytes(buf)))
        }
        3 => {
            // INT16
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::I16(i16::from_le_bytes(buf)))
        }
        4 => {
            // UINT32
            Ok(GgufMetadataValue::U32(read_u32(reader)?))
        }
        5 => {
            // INT32
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::I32(i32::from_le_bytes(buf)))
        }
        6 => {
            // FLOAT32
            Ok(GgufMetadataValue::F32(read_f32(reader)?))
        }
        7 => {
            // BOOL
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::Bool(buf[0] != 0))
        }
        8 => {
            // STRING
            Ok(GgufMetadataValue::String(read_string(reader)?))
        }
        9 => {
            // ARRAY
            let elem_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            if count > MAX_ARRAY_LENGTH {
                return Err(GgufError::ArrayLengthExceeded {
                    count,
                    max: MAX_ARRAY_LENGTH,
                });
            }
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let val = read_typed_value(reader, elem_type)?;
                values.push(val);
            }
            Ok(GgufMetadataValue::Array(values))
        }
        10 => {
            // UINT64
            Ok(GgufMetadataValue::U64(read_u64(reader)?))
        }
        11 => {
            // INT64
            Ok(GgufMetadataValue::I64(read_i64(reader)?))
        }
        12 => {
            // FLOAT64
            Ok(GgufMetadataValue::F64(read_f64(reader)?))
        }
        _ => Err(GgufError::UnknownMetadataType { type_id }),
    }
}
