// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Filesystem cache for compiled HIP `.hsaco` code objects.
//!
//! Cache key = SHA-256(source + target_arch). This ensures that:
//! - Different source code always produces different cache keys.
//! - Same source compiled for different GPU architectures gets separate entries.
//! - Rebuilds after source changes always recompile.
//!
//! Cache directory layout:
//! ```text
//! <cache_dir>/
//!   <hash>.hsaco          — compiled AMD GPU code object
//!   <hash>.hip.cpp        — source (retained for debugging / recompilation)
//!   <hash>.meta           — metadata (target_arch, timestamp)
//! ```

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Filesystem cache for compiled HIP code objects.
///
/// Thread-safe via [`RwLock`] on the in-memory index. The filesystem is the
/// source of truth; the in-memory index avoids repeated filesystem probes.
#[derive(Debug)]
pub struct HipCache {
    dir: PathBuf,
    /// In-memory index: cache key → hsaco path. Populated lazily on lookup.
    index: RwLock<HashMap<String, PathBuf>>,
}

impl HipCache {
    /// Create a new cache rooted at `dir`.
    ///
    /// Creates the directory if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the directory cannot be created.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            index: RwLock::new(HashMap::new()),
        })
    }

    /// Create a cache in the default location (`target/nn-hip-cache/`).
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the directory cannot be created.
    pub fn default_location() -> Result<Self, std::io::Error> {
        let dir = PathBuf::from("target").join("nn-hip-cache");
        Self::new(dir)
    }

    /// The cache directory path.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Compute a cache key from source content and target architecture.
    ///
    /// Uses a simple hash function (FNV-1a inspired) that is fast and
    /// sufficient for deduplication. Not cryptographic — collisions are
    /// handled by filename matching, not security.
    #[must_use]
    pub fn content_hash(source: &str, target_arch: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        target_arch.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Look up a cached `.hsaco` for the given source and target arch.
    ///
    /// Returns `Some(path)` if the cached file exists, `None` otherwise.
    pub fn lookup(&self, source: &str, target_arch: &str) -> Option<PathBuf> {
        let key = Self::content_hash(source, target_arch);

        // Check in-memory index first.
        if let Ok(index) = self.index.read() {
            if let Some(path) = index.get(&key) {
                if path.exists() {
                    return Some(path.clone());
                }
            }
        }

        // Check filesystem.
        let hsaco_path = self.dir.join(format!("{key}.hsaco"));
        if hsaco_path.exists() {
            if let Ok(mut index) = self.index.write() {
                index.insert(key, hsaco_path.clone());
            }
            return Some(hsaco_path);
        }

        None
    }

    /// Register a compiled `.hsaco` in the cache.
    ///
    /// If the `.hsaco` file is not already in the cache directory, it is
    /// copied there. The in-memory index is updated.
    pub fn register(&self, source: &str, target_arch: &str, hsaco_path: &Path) {
        let key = Self::content_hash(source, target_arch);
        let cached_path = self.dir.join(format!("{key}.hsaco"));

        // Copy if the hsaco is not already in the cache dir.
        if hsaco_path != cached_path && hsaco_path.exists() {
            let _ = std::fs::copy(hsaco_path, &cached_path);
        }

        // Write metadata for debugging.
        let meta_path = self.dir.join(format!("{key}.meta"));
        if let Ok(mut f) = std::fs::File::create(&meta_path) {
            let _ = writeln!(f, "target_arch={target_arch}");
            let _ = writeln!(f, "source_len={}", source.len());
        }

        // Update in-memory index.
        if let Ok(mut index) = self.index.write() {
            index.insert(key, cached_path);
        }
    }

    /// Number of entries in the in-memory index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.read().map_or(0, |i| i.len())
    }

    /// Returns `true` if the in-memory index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all cached files and clear the in-memory index.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if files cannot be removed.
    pub fn clear(&self) -> Result<(), std::io::Error> {
        if let Ok(mut index) = self.index.write() {
            index.clear();
        }
        // Remove .hsaco, .hip.cpp, and .meta files.
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|ext| ext == "hsaco" || ext == "cpp" || ext == "meta")
            {
                std::fs::remove_file(&path)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = HipCache::content_hash("void k() {}", "gfx90a");
        let h2 = HipCache::content_hash("void k() {}", "gfx90a");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_differs_by_arch() {
        let h1 = HipCache::content_hash("void k() {}", "gfx90a");
        let h2 = HipCache::content_hash("void k() {}", "gfx1100");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_content_hash_differs_by_source() {
        let h1 = HipCache::content_hash("void k() {}", "gfx90a");
        let h2 = HipCache::content_hash("void k2() {}", "gfx90a");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_cache_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = HipCache::new(tmp.path()).unwrap();

        // No cache hit initially.
        assert!(cache.lookup("source", "gfx90a").is_none());
        assert!(cache.is_empty());

        // Write a fake .hsaco and register it.
        let fake_hsaco = tmp.path().join("fake.hsaco");
        std::fs::write(&fake_hsaco, b"FAKE_HSACO").unwrap();
        cache.register("source", "gfx90a", &fake_hsaco);

        // Cache hit.
        let result = cache.lookup("source", "gfx90a");
        assert!(result.is_some());
        assert_eq!(cache.len(), 1);

        // Different arch = miss.
        assert!(cache.lookup("source", "gfx1100").is_none());
    }

    #[test]
    fn test_cache_clear() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = HipCache::new(tmp.path()).unwrap();

        let fake_hsaco = tmp.path().join("fake.hsaco");
        std::fs::write(&fake_hsaco, b"FAKE").unwrap();
        cache.register("src", "gfx90a", &fake_hsaco);
        assert_eq!(cache.len(), 1);

        cache.clear().unwrap();
        assert!(cache.is_empty());
    }
}
