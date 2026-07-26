// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe lazy GPU weight cache with atomic replace/invalidate.
//!
//! Extracted from `lib.rs` to keep it under 450 lines.

/// RAII guard that holds a lock on cached GPU weights, preventing use-after-free.
///
/// Returned by [`GpuWeightCache::get_or_init_with`]. Dereferences to `T`, keeping the
/// underlying `RwLock` guard alive for the duration of the borrow. Callers use it
/// identically to a `&T` reference via `Deref`.
pub(crate) enum GpuWeightRef<'a, T> {
    Read(std::sync::RwLockReadGuard<'a, Option<Result<T, String>>>),
    Write(std::sync::RwLockWriteGuard<'a, Option<Result<T, String>>>),
}

impl<T> std::ops::Deref for GpuWeightRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        let opt = match self {
            GpuWeightRef::Read(g) => g.as_ref(),
            GpuWeightRef::Write(g) => g.as_ref(),
        };
        // The guard is only constructed when inner is Some(Ok(..)), so this is safe.
        opt.expect("GpuWeightRef only constructed when inner is Some")
            .as_ref()
            .expect("GpuWeightRef only constructed when inner is Ok")
    }
}

/// Thread-safe lazy GPU weight cache with atomic replace/invalidate.
///
/// Encapsulates the `RwLock<Option<Result<T, String>>>` + `get_or_init` + error
/// conversion pattern used by all 4 model structs (SileroVad, DemucsTransformer,
/// DemucsTemporalEncoder, DemucsTemporalDecoder).
///
/// Unlike the previous `OnceLock` design, this supports:
/// - **Replace**: swap cached weights atomically (for live weight editing).
/// - **Invalidate**: force re-initialization on next access.
/// - **Generation counter**: monotonically increasing on each replace/invalidate,
///   enabling stale-cache detection in KV caches and downstream consumers.
///
/// Usage:
/// ```ignore
/// struct NnModel {
///     gpu_weights: GpuWeightCache<NnGpuWeights>,
/// }
/// impl NnModel {
///     fn ensure_gpu(&self, cache: &PipelineCache) -> Result<GpuWeightRef<'_, NnGpuWeights>, NnError> {
///         self.gpu_weights.get_or_init_with(
///             || build_weights(cache),
///             |e| NnError::GpuBufferAlloc(e),
///         )
///     }
/// }
/// ```
pub(crate) struct GpuWeightCache<T> {
    inner: std::sync::RwLock<Option<Result<T, String>>>,
    #[allow(dead_code)] // used by replace/invalidate/generation methods (test-only callers)
    generation: std::sync::atomic::AtomicU64,
}

impl<T> GpuWeightCache<T> {
    /// Create an empty cache.
    pub(crate) fn new() -> Self {
        Self {
            inner: std::sync::RwLock::new(None),
            generation: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get or initialize the cached GPU weights.
    ///
    /// Returns a [`GpuWeightRef`] RAII guard that dereferences to `&T` and keeps
    /// the underlying lock held. The lock is released when the guard is dropped.
    ///
    /// `init` builds the weights (called at most once, or once after each
    /// [`invalidate`](Self::invalidate)/[`replace`](Self::replace)).
    /// `map_err` converts the cached `String` error to the caller's error type.
    ///
    /// Recovers from poisoned locks (another thread panicked while holding
    /// the lock) by clearing the poison — the cached data is still valid.
    pub(crate) fn get_or_init_with<E>(
        &self,
        init: impl FnOnce() -> Result<T, String>,
        map_err: impl FnOnce(String) -> E,
    ) -> Result<GpuWeightRef<'_, T>, E> {
        // Fast path: read lock to check if already initialized.
        // clear_poison: a panicked thread doesn't invalidate cached GPU weights.
        {
            let guard = self
                .inner
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(result) = guard.as_ref() {
                return match result {
                    Ok(_) => Ok(GpuWeightRef::Read(guard)),
                    Err(e) => Err(map_err(e.clone())),
                };
            }
        }
        // Slow path: write lock to initialize.
        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.is_none() {
            *guard = Some(init());
        }
        match guard.as_ref().expect("guard set to Some by init() above") {
            Ok(_) => Ok(GpuWeightRef::Write(guard)),
            Err(e) => Err(map_err(e.clone())),
        }
    }

    /// Replace cached weights atomically. Returns the previous generation number.
    ///
    /// After replacement, the next call to [`get_or_init_with`](Self::get_or_init_with)
    /// returns the new weights without re-initializing.
    ///
    /// # Safety contract
    ///
    /// Callers must ensure no in-flight GPU command buffers reference the old
    /// weight buffers. Use `commit_and_wait()` before calling this method.
    #[allow(dead_code)] // called by #[cfg(test)] apply_weight_edit_with_generation
    pub(crate) fn replace(&self, new_weights: T) -> u64 {
        let prev = self
            .generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut guard = self.inner.write().expect("GpuWeightCache lock poisoned");
        *guard = Some(Ok(new_weights));
        prev
    }

    /// Invalidate cached weights. The next call to [`get_or_init_with`](Self::get_or_init_with)
    /// will re-run the initialization closure.
    #[allow(dead_code)] // called by #[cfg(test)] apply_weight_edit_with_generation
    pub(crate) fn invalidate(&self) {
        self.generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut guard = self.inner.write().expect("GpuWeightCache lock poisoned");
        *guard = None;
    }

    /// Current generation number (monotonically increasing on each replace/invalidate).
    #[allow(dead_code)] // called by #[cfg(test)] apply_weight_edit_with_generation
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::SeqCst)
    }
}
