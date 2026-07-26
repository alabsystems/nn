// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic data types for the Kokoro compiled pipeline.
//!
//! Extracted from `compiled_kokoro_diagnostics.rs` for code structure
//! compliance (wave 9 D2). Contains [`DispatchSummary`], [`TimingReport`],
//! and [`DiagnosticOutput`].

use std::fmt;
use std::time::Duration;

use crate::arena::ArenaStats;
use crate::dispatch_stats::DispatchStats;
use crate::rss::RssTracker;

/// Per-segment GPU dispatch counts for the Kokoro pipeline.
///
/// Each field gives the logical dispatch count for the most-recently-used
/// compiled model in that segment cache. Segments not yet compiled show 0.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DispatchSummary {
    pub plbert: usize,
    pub text_encoder: usize,
    pub prosody: usize,
    pub f0_energy: usize,
    pub generator: usize,
    /// Regulate elementwise chain (segment 5). Part of #1815 Tier 6 D2b.
    pub regulate: usize,
    /// SineGen pre-cumsum: F0 → rad_frames + voiced. Part of #1815 D2.
    pub sinegen_pre: usize,
    /// SineGen post-cumsum: phase → excitation. Part of #1815 D3.
    pub sinegen_post: usize,
}

impl DispatchSummary {
    /// Total logical dispatches across all segments.
    #[must_use]
    pub fn total(&self) -> usize {
        self.plbert
            + self.text_encoder
            + self.prosody
            + self.f0_energy
            + self.generator
            + self.regulate
            + self.sinegen_pre
            + self.sinegen_post
    }

    /// Expected Metal command buffer count for two-phase pipelining.
    ///
    /// The production pipeline (TwoPhase mode) creates command buffers at:
    /// 1. Phase 1: after encode (submit)
    /// 2. Phase 1: after prosody (submit)
    /// 3. Phase 1: regulate inherent submit+sync (prefix-sum readback)
    /// 4. Phase 1→2 boundary: fence submit
    /// 5. Phase 2: after f0+harmonic (single batched submit)
    /// 6. Phase 2: pipeline-exit commit (generator+iSTFT in one batch)
    ///
    /// Returns 6 for the production pipeline. On the regulate cache-hit
    /// hot path, the regulate sync is skipped but the command buffer is
    /// still created (the prefix-sum dispatch stays in the batch).
    ///
    /// Part of #4264.
    #[must_use]
    pub fn expected_submit_count(&self) -> usize {
        // Phase 1: encode, prosody, regulate, phase1→2 boundary = 4
        // Phase 2: f0+harmonic, pipeline-exit (generator+iSTFT) = 2
        6
    }
}

/// Per-stage wall-clock timing for a single Kokoro synthesis call.
///
/// Wraps each pipeline step with [`Instant::now()`] to decompose end-to-end
/// latency. Includes cache miss count so callers can distinguish compilation
/// overhead from steady-state execution time.
///
/// Part of #2781.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TimingReport {
    /// PlBert + bert_encoder + TextEncoder (steps 1–2).
    pub encode: Duration,
    /// ProsodyPredictor (step 3).
    pub prosody: Duration,
    /// Duration + length_regulate (step 4). Compiled segment + 1 micro-sync.
    pub regulate: Duration,
    /// F0EnergyPredictor (step 5).
    pub f0_energy: Duration,
    /// Harmonic source generation (step 6).
    pub harmonic: Duration,
    /// Generator / FullDecoder (step 7).
    pub generate: Duration,
    /// GPU iSTFT → PCM audio (step 8). Terminal GPU sync.
    pub istft: Duration,
    /// Audio quality verification (step 9). CPU-only.
    pub verify: Duration,
    /// Sum of all stage durations.
    pub total: Duration,
    /// Number of segment cache misses (recompilations) during this call.
    /// Steady-state should be 0; first call or shape changes cause misses.
    pub cache_misses: usize,
}

impl fmt::Display for TimingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn ms(d: Duration) -> f64 {
            d.as_secs_f64() * 1000.0
        }
        writeln!(f, "Kokoro TimingReport")?;
        writeln!(f, "  encode:     {:>8.2} ms", ms(self.encode))?;
        writeln!(f, "  prosody:    {:>8.2} ms", ms(self.prosody))?;
        writeln!(
            f,
            "  regulate:   {:>8.2} ms  (1 submit+sync, 4-byte readback)",
            ms(self.regulate)
        )?;
        writeln!(f, "  f0_energy:  {:>8.2} ms", ms(self.f0_energy))?;
        writeln!(
            f,
            "  harmonic:   {:>8.2} ms  (SineGen GPU, Kahan cumsum #2909)",
            ms(self.harmonic)
        )?;
        writeln!(f, "  generate:   {:>8.2} ms", ms(self.generate))?;
        writeln!(
            f,
            "  istft:      {:>8.2} ms  (terminal GPU sync)",
            ms(self.istft)
        )?;
        writeln!(f, "  verify:     {:>8.2} ms  (CPU only)", ms(self.verify))?;
        writeln!(f, "  total:      {:>8.2} ms", ms(self.total))?;
        write!(f, "  cache_misses: {}", self.cache_misses)
    }
}

/// Per-domain memory attribution for the Kokoro pipeline (#3079 D7).
///
/// Breaks down process RSS into known memory domains to identify where
/// the 2.3x gap vs PyTorch originates. Known GPU domains (weights, arena,
/// pool) are subtracted from total RSS to reveal unaccounted memory.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct MemoryBreakdown {
    /// Total GPU weight buffer bytes across all compiled segments.
    pub gpu_weight_bytes: usize,
    /// Pre-allocated arena capacity in bytes (default 64 MB).
    pub arena_capacity_bytes: usize,
    /// Peak arena bytes actually used (high-water mark).
    /// If much less than `arena_capacity_bytes`, the arena is oversized.
    pub arena_peak_bytes: usize,
    /// Pool retained bytes (Metal buffers held for reuse).
    pub pool_retained_bytes: usize,
    /// Total cached planned buffer bytes across all segment cache entries.
    /// Each `CompiledModel` holds a contiguous GPU buffer for intermediate
    /// sub-allocation (`BufferPlan`). With up to 4 cached shapes × 8 segments,
    /// these can sum to hundreds of MB.
    pub planned_buf_bytes: usize,
    /// Whether CPU weight copies have been released.
    pub cpu_weights_released: bool,
    /// Current process RSS in bytes at time of measurement.
    pub process_rss_bytes: Option<usize>,
    /// Metal device `current_allocated_size()` in bytes at time of measurement.
    /// Cross-check against `known_gpu_bytes()` — the gap reveals untracked
    /// Metal allocations (pipeline cache shaders, OS overhead, etc.).
    /// `None` on non-macOS platforms.
    pub metal_allocated_bytes: Option<usize>,
    /// Number of cached CompiledModel instances across all segments.
    /// With byte-budget eviction (#3079), this is less than
    /// `capacity × 8 segments` when large models are cached.
    pub cached_model_count: usize,
}

impl MemoryBreakdown {
    /// Sum of known GPU memory domains (weights + arena + pool + planned bufs).
    #[must_use]
    pub fn known_gpu_bytes(&self) -> usize {
        self.gpu_weight_bytes
            + self.arena_capacity_bytes
            + self.pool_retained_bytes
            + self.planned_buf_bytes
    }

    /// RSS minus known GPU domains. `None` if RSS unavailable.
    #[must_use]
    pub fn unaccounted_bytes(&self) -> Option<usize> {
        self.process_rss_bytes
            .map(|rss| rss.saturating_sub(self.known_gpu_bytes()))
    }

    /// Metal allocations NOT tracked by known categories (PSOs, framework, etc.).
    /// = metal_allocated - known_gpu. `None` if Metal telemetry unavailable.
    #[must_use]
    pub fn metal_overhead_bytes(&self) -> Option<usize> {
        self.metal_allocated_bytes
            .map(|metal| metal.saturating_sub(self.known_gpu_bytes()))
    }

    /// CPU-only RSS: process base, Rust heap, OS overhead.
    /// = process_rss - metal_allocated. `None` if either is unavailable.
    #[must_use]
    pub fn cpu_overhead_bytes(&self) -> Option<usize> {
        match (self.process_rss_bytes, self.metal_allocated_bytes) {
            (Some(rss), Some(metal)) => Some(rss.saturating_sub(metal)),
            _ => None,
        }
    }

    /// Decomposition validity: known_gpu + metal_overhead + cpu_overhead == rss.
    /// Returns `false` if any component is unavailable or saturating_sub distorted values.
    #[must_use]
    pub fn decomposition_valid(&self) -> bool {
        match (self.process_rss_bytes, self.metal_allocated_bytes) {
            (Some(rss), Some(metal)) => {
                let known = self.known_gpu_bytes();
                let overhead = metal.saturating_sub(known);
                let cpu = rss.saturating_sub(metal);
                known + overhead + cpu == rss
            }
            _ => false,
        }
    }
}

impl fmt::Display for MemoryBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn mb(b: usize) -> f64 {
            b as f64 / (1024.0 * 1024.0)
        }
        writeln!(f, "Memory Breakdown (#3079 D7)")?;
        writeln!(f, "  gpu weights:    {:>8.1} MB", mb(self.gpu_weight_bytes))?;
        writeln!(
            f,
            "  arena capacity: {:>8.1} MB",
            mb(self.arena_capacity_bytes)
        )?;
        let util_pct = if self.arena_capacity_bytes > 0 {
            self.arena_peak_bytes as f64 / self.arena_capacity_bytes as f64 * 100.0
        } else {
            0.0
        };
        writeln!(
            f,
            "  arena peak:     {:>8.1} MB ({:.0}% utilization)",
            mb(self.arena_peak_bytes),
            util_pct,
        )?;
        writeln!(
            f,
            "  pool retained:  {:>8.1} MB",
            mb(self.pool_retained_bytes)
        )?;
        writeln!(
            f,
            "  planned bufs:   {:>8.1} MB  ({} cached models)",
            mb(self.planned_buf_bytes),
            self.cached_model_count,
        )?;
        writeln!(
            f,
            "  known GPU total:{:>8.1} MB",
            mb(self.known_gpu_bytes())
        )?;
        writeln!(
            f,
            "  cpu weights:    {}",
            if self.cpu_weights_released {
                "released"
            } else {
                "held"
            },
        )?;
        if let Some(metal) = self.metal_allocated_bytes {
            writeln!(f, "  Metal alloc:    {:>8.1} MB", mb(metal))?;
            if let Some(overhead) = self.metal_overhead_bytes() {
                writeln!(
                    f,
                    "  Metal overhead: {:>8.1} MB  (PSOs, framework)",
                    mb(overhead),
                )?;
            }
        }
        if let Some(cpu) = self.cpu_overhead_bytes() {
            writeln!(f, "  CPU overhead:   {:>8.1} MB  (base, heap, OS)", mb(cpu))?;
        }
        if let Some(rss) = self.process_rss_bytes {
            writeln!(f, "  process RSS:    {:>8.1} MB", mb(rss))?;
            if let Some(unaccounted) = self.unaccounted_bytes() {
                write!(f, "  unaccounted:    {:>8.1} MB", mb(unaccounted))?;
            }
        } else {
            write!(f, "  process RSS:    unavailable")?;
        }
        Ok(())
    }
}

/// Per-stage GPU execution timing for a single Kokoro synthesis call.
///
/// Unlike [`TimingReport`] which measures CPU encoding time (GPU work is
/// batched lazily), this report flushes GPU work after each step to measure
/// actual GPU execution time per segment. This is a profiling-only tool
/// and imposes significant overhead (1 flush per step instead of 1 total).
///
/// Part of #4264.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GpuTimingReport {
    /// PlBert + bert_encoder + TextEncoder (steps 1-2), including GPU wait.
    pub encode: Duration,
    /// ProsodyPredictor (step 3), including GPU wait.
    pub prosody: Duration,
    /// Duration + length_regulate (step 4), including GPU wait.
    pub regulate: Duration,
    /// F0EnergyPredictor (step 5), including GPU wait.
    pub f0_energy: Duration,
    /// Harmonic source generation (step 6), including GPU wait.
    pub harmonic: Duration,
    /// Generator / FullDecoder (step 7), including GPU wait.
    pub generate: Duration,
    /// GPU iSTFT (step 8), including GPU wait.
    pub istft: Duration,
    /// Audio quality verification (step 9). CPU-only.
    pub verify: Duration,
    /// Wall-clock total including all flushes.
    pub total: Duration,
    /// Number of segment cache misses during this call.
    pub cache_misses: usize,
}

impl fmt::Display for GpuTimingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn ms(d: Duration) -> f64 {
            d.as_secs_f64() * 1000.0
        }
        writeln!(f, "Kokoro GpuTimingReport (per-step GPU flush)")?;
        writeln!(f, "  encode:     {:>8.2} ms  (GPU wait)", ms(self.encode))?;
        writeln!(f, "  prosody:    {:>8.2} ms  (GPU wait)", ms(self.prosody))?;
        writeln!(f, "  regulate:   {:>8.2} ms  (GPU wait)", ms(self.regulate))?;
        writeln!(f, "  f0_energy:  {:>8.2} ms  (GPU wait)", ms(self.f0_energy))?;
        writeln!(f, "  harmonic:   {:>8.2} ms  (GPU wait)", ms(self.harmonic))?;
        writeln!(f, "  generate:   {:>8.2} ms  (GPU wait)", ms(self.generate))?;
        writeln!(f, "  istft:      {:>8.2} ms  (GPU wait)", ms(self.istft))?;
        writeln!(f, "  verify:     {:>8.2} ms  (CPU only)", ms(self.verify))?;
        writeln!(f, "  total:      {:>8.2} ms", ms(self.total))?;
        write!(f, "  cache_misses: {}", self.cache_misses)
    }
}

/// Per-segment dispatch census with categorized step counts.
///
/// Provides a detailed breakdown of each segment's compiled steps by
/// operation type: NativeOps, IR Dispatches, RuntimeOps, and zero-cost
/// steps (Passthrough, NarrowView, InputForward, IdentityPassthrough,
/// ConstantValue). Each category includes per-variant counts.
///
/// Use [`CompiledKokoro::dispatch_census()`] to obtain this report.
/// Part of #4264.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DispatchCensus {
    /// Per-segment census entries.
    pub segments: Vec<SegmentCensus>,
    /// Total logical dispatches (NativeOp + IR Dispatch + RuntimeOp).
    pub total_dispatches: usize,
    /// Total estimated Metal kernel launches across all segments.
    pub total_metal_dispatches: usize,
    /// Total zero-cost steps (no GPU work).
    pub total_zero_cost: usize,
    /// Total compiled steps across all segments.
    pub total_steps: usize,
}

/// Census for a single compiled segment.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SegmentCensus {
    /// Segment name (e.g., "plbert", "generator").
    pub name: String,
    /// Logical dispatch count (NativeOp + IR Dispatch + RuntimeOp).
    pub dispatches: usize,
    /// Estimated Metal kernel launches.
    pub metal_dispatches: usize,
    /// NativeOp step count and per-variant breakdown.
    pub native_ops: Vec<(String, usize)>,
    /// IR Dispatch step count and per-kernel breakdown.
    pub ir_dispatches: Vec<(String, usize)>,
    /// RuntimeOp step count.
    pub runtime_ops: usize,
    /// Zero-cost steps (Passthrough, NarrowView, etc.).
    pub zero_cost: usize,
    /// Total compiled steps in this segment.
    pub total_steps: usize,
    /// Adjacent dispatch pairs that are candidates for fusion.
    /// Each entry is (step_a_detail, step_b_detail).
    pub fusion_candidates: Vec<(String, String)>,
}

impl DispatchCensus {
    /// Segments sorted by dispatch count (heaviest first).
    #[must_use]
    pub fn heaviest_segments(&self) -> Vec<&SegmentCensus> {
        let mut sorted: Vec<&SegmentCensus> = self.segments.iter().collect();
        sorted.sort_by_key(|s| std::cmp::Reverse(s.dispatches));
        sorted
    }

    /// Total fusion candidate pairs across all segments.
    #[must_use]
    pub fn total_fusion_candidates(&self) -> usize {
        self.segments
            .iter()
            .map(|s| s.fusion_candidates.len())
            .sum()
    }

    /// Dispatches that must be eliminated to reach the target.
    #[must_use]
    pub fn gap_to_target(&self, target: usize) -> usize {
        self.total_dispatches.saturating_sub(target)
    }
}

impl fmt::Display for DispatchCensus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Kokoro Dispatch Census")?;
        writeln!(
            f,
            "  Total dispatches: {}  |  Metal launches: {}  |  Zero-cost: {}  |  Steps: {}",
            self.total_dispatches, self.total_metal_dispatches, self.total_zero_cost, self.total_steps,
        )?;
        writeln!(f, "  Gap to target (<60): {}", self.gap_to_target(60))?;
        writeln!(f)?;
        for seg in &self.segments {
            writeln!(
                f,
                "  [{:<14}] dispatches={:<4} metal={:<4} native={:<3} ir={:<3} runtime={:<2} zero_cost={:<3}",
                seg.name,
                seg.dispatches,
                seg.metal_dispatches,
                seg.native_ops.iter().map(|(_, c)| c).sum::<usize>(),
                seg.ir_dispatches.iter().map(|(_, c)| c).sum::<usize>(),
                seg.runtime_ops,
                seg.zero_cost,
            )?;
            // Top NativeOps by count
            for (variant, count) in &seg.native_ops {
                if *count > 0 {
                    writeln!(f, "    NativeOp: {variant} x{count}")?;
                }
            }
            for (kernel, count) in &seg.ir_dispatches {
                if *count > 0 {
                    writeln!(f, "    IR Dispatch: {kernel} x{count}")?;
                }
            }
            if !seg.fusion_candidates.is_empty() {
                writeln!(
                    f,
                    "    Fusion candidates: {} pairs",
                    seg.fusion_candidates.len()
                )?;
            }
        }
        Ok(())
    }
}

/// Combined diagnostic output from a single Kokoro synthesis call.
///
/// Wraps [`TimingReport`] (per-stage wall-clock timing) and
/// [`DispatchStats`] (GPU flush/submit/encoding counts) into a single
/// return value for comprehensive pipeline analysis.
///
/// Part of #2781.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DiagnosticOutput {
    /// Per-stage wall-clock timing breakdown.
    pub timing: TimingReport,
    /// GPU dispatch statistics (flushes, submits, encodings).
    pub stats: DispatchStats,
    /// Peak arena bytes used during synthesis (high-water mark).
    /// `None` when no explicit arena was active (#2914).
    pub arena_peak_bytes: Option<usize>,
    /// Arena allocation hit/miss counts during synthesis.
    /// Misses indicate overflow to standalone Metal buffers — input for D3
    /// buffer pool sizing (#3079).
    pub arena_stats: ArenaStats,
    /// RSS memory checkpoints across the synthesis pipeline (#3079).
    /// `None` when memory profiling was not requested.
    pub rss: Option<RssTracker>,
    /// Per-domain memory attribution (#3079 D7).
    /// `None` when memory breakdown was not requested.
    pub memory: Option<MemoryBreakdown>,
}

impl fmt::Display for DiagnosticOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.timing)?;
        writeln!(f)?;
        writeln!(f, "  GPU stats:")?;
        writeln!(f, "    flushes:   {}", self.stats.flushes)?;
        writeln!(f, "    submits:   {}", self.stats.submits)?;
        writeln!(f, "    compute:   {}", self.stats.compute_encodings)?;
        writeln!(f, "    blits:     {}", self.stats.blits)?;
        if self.stats.blits_eliminated > 0 {
            writeln!(f, "    blits_eliminated: {}", self.stats.blits_eliminated)?;
        }
        let capacity = crate::arena::arena_capacity();
        if let Some(peak) = self.arena_peak_bytes {
            let util_pct = peak as f64 / capacity as f64 * 100.0;
            writeln!(
                f,
                "  arena: {:.1} MB peak / {:.1} MB capacity ({:.0}% utilization)",
                peak as f64 / (1024.0 * 1024.0),
                capacity as f64 / (1024.0 * 1024.0),
                util_pct,
            )?;
        }
        let total_allocs = self.arena_stats.hits + self.arena_stats.misses;
        if total_allocs > 0 {
            writeln!(
                f,
                "  arena stats: {} hits, {} misses ({:.0}% hit rate)",
                self.arena_stats.hits,
                self.arena_stats.misses,
                self.arena_stats.hit_rate() * 100.0,
            )?;
            let fresh = self.arena_stats.fresh_allocs();
            let pool = &self.arena_stats.pool;
            if pool.hits > 0 || pool.pooled_buffers > 0 || fresh > 0 {
                writeln!(
                    f,
                    "  buffer pool: {} reuses, {} fresh allocs, {} discards, {} entries ({:.1} MB retained)",
                    pool.hits,
                    fresh,
                    pool.discards,
                    pool.pooled_buffers,
                    pool.pooled_bytes as f64 / (1024.0 * 1024.0),
                )?;
            }
        }
        if let Some(ref rss) = self.rss {
            writeln!(f)?;
            write!(f, "{rss}")?;
        }
        if let Some(ref mem) = self.memory {
            writeln!(f)?;
            write!(f, "{mem}")?;
        }
        Ok(())
    }
}
