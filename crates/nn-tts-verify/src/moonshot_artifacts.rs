// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Artifact-building functions for the moonshot verification tracker.
//!
//! Each function returns verification artifacts for a category of moonshot
//! properties. These are pure data builders with no reverse dependencies.

use super::{VerificationArtifact, VerificationLevel};

/// Properties 1+2: Non-silence and Non-clipping verification artifacts.
pub(super) fn audio_quality_artifacts() -> Vec<VerificationArtifact> {
    vec![
        VerificationArtifact {
            description: "Kokoro ISTFTNet vocoder CROWN bounds (exp > 0)",
            file: "crates/nn-verify/tests/compose_kokoro_decoder.rs",
            properties: &[0, 1],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description: "Hard bound checks (RMS, amplitude, DC offset, clicks)",
            file: "crates/nn-tts-verify/src/bounds.rs",
            properties: &[0, 1],
            level: VerificationLevel::Empirical,
        },
        VerificationArtifact {
            description: "CROWN vocoder output range proof bridge",
            file: "crates/nn-tts-verify/src/crown.rs",
            properties: &[0, 1],
            level: VerificationLevel::CrownPartial,
        },
    ]
}

/// Property 3: Intelligibility (attention monotonicity) artifacts.
pub(super) fn intelligibility_artifacts() -> Vec<VerificationArtifact> {
    vec![
        VerificationArtifact {
            description: "Duration positivity certificate (exp(dur_logits) > 0)",
            file: "crates/nn-tts-verify/src/monotonicity.rs",
            properties: &[2],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description: "ProsodyPredictor CROWN composition (1-block, 3-block, T=4)",
            file: "crates/nn-verify/tests/compose_kokoro_duration.rs",
            properties: &[2],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description: "ProsodyPredictor T=4 temporal unrolling",
            file: "crates/nn-verify/tests/compose_kokoro_duration_t4.rs",
            properties: &[2],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description: "ICLR sensitivity analysis (weight/input sweep)",
            file: "crates/nn-verify/tests/compose_kokoro_duration_sensitivity.rs",
            properties: &[2],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description: "Attention monotonicity upgrade — diagonal dominance CROWN certificate upgrades P3 to CrownProven",
            file: "crates/nn-tts-verify/src/moonshot_crown_attention.rs",
            properties: &[2],
            level: VerificationLevel::CrownProven,
        },
    ]
}

/// Properties 4+5+6: Design-phase artifacts (speaker, temporal, streaming).
pub(super) fn design_phase_artifacts() -> Vec<VerificationArtifact> {
    vec![
        VerificationArtifact {
            description: "ECAPA-TDNN speaker encoder design doc",
            file: "designs/archive/2026-03-10-ecapa-tdnn-speaker-encoder.md",
            properties: &[3],
            level: VerificationLevel::None,
        },
        VerificationArtifact {
            description:
                "ECAPA-TDNN-512 model (nn primitives + model struct, Phase 1 inference)",
            file: "crates/nn-models/src/ecapa_tdnn.rs",
            properties: &[3],
            level: VerificationLevel::Empirical,
        },
        VerificationArtifact {
            description:
                "CROWN speaker consistency — worst-case L2 distance from ECAPA-TDNN embedding bounds",
            file: "crates/nn-tts-verify/src/moonshot_crown_speaker.rs",
            properties: &[3],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description:
                "D=192 composed speaker pipeline — 4-stage CROWN (text→prosody→vocoder→speaker) + 6-property bundle",
            file: "crates/nn-tts-verify/src/moonshot_crown_tests_speaker.rs",
            properties: &[3],
            level: VerificationLevel::CrownProven,
        },
        VerificationArtifact {
            description:
                "D=192 composed 3-stage temporal pipeline — CROWN composition + per-stage cost profiling",
            file: "crates/nn-tts-verify/src/moonshot_crown_tests_temporal_composed.rs",
            properties: &[0, 1, 2, 4, 5],
            level: VerificationLevel::CrownProven,
        },
        VerificationArtifact {
            description:
                "Full 7-property D=192 bundle with attention monotonicity — all 6 CROWN properties CrownProven",
            file: "crates/nn-tts-verify/src/moonshot_crown_tests_temporal_composed.rs",
            properties: &[0, 1, 2, 3, 4, 5],
            level: VerificationLevel::CrownProven,
        },
        VerificationArtifact {
            description: "Computational boundedness design doc",
            file: "designs/archive/2026-03-10-computational-boundedness.md",
            properties: &[4],
            level: VerificationLevel::Empirical,
        },
        VerificationArtifact {
            description: "Roofline cost model + timing certificate",
            file: "crates/nn-tts-verify/src/pipeline_hybrid.rs",
            properties: &[4],
            level: VerificationLevel::Empirical,
        },
        VerificationArtifact {
            description: "Roofline calibration — predicted vs measured GPU timing (15 tests)",
            file: "crates/nn-tts-verify/src/cost_model_calibration.rs",
            properties: &[4],
            level: VerificationLevel::Empirical,
        },
        VerificationArtifact {
            description:
                "CROWN-coupled timing certificate — per-layer CROWN + roofline cost bounds",
            file: "crates/nn-tts-verify/src/pipeline_hybrid.rs",
            properties: &[4],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description:
                "Moonshot Property 5 bridge — check_temporal_boundedness + verify_properties_with_timing",
            file: "crates/nn-tts-verify/src/moonshot_crown.rs",
            properties: &[4],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description:
                "Conservative roofline model with empirical correction factors (5x compute, 2x BW)",
            file: "crates/nn-tts-verify/src/cost_model.rs",
            properties: &[4],
            level: VerificationLevel::Empirical,
        },
        VerificationArtifact {
            description:
                "Kokoro-scale 9-step dispatch plan timing certificate (100ms target on M4 Max)",
            file: "crates/nn-tts-verify/src/cost_propagation_tests.rs",
            properties: &[4],
            level: VerificationLevel::Empirical,
        },
        VerificationArtifact {
            description: "Streaming boundary verification",
            file: "crates/nn-tts-verify/src/streaming.rs",
            properties: &[5],
            level: VerificationLevel::Empirical,
        },
        VerificationArtifact {
            description:
                "CROWN streaming safety — bounded crossfade discontinuity via output bounds",
            file: "crates/nn-tts-verify/src/moonshot_crown.rs",
            properties: &[5],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description: "Streaming verification design doc",
            file: "designs/archive/2026-03-10-streaming-voice-verification.md",
            properties: &[5],
            level: VerificationLevel::None,
        },
    ]
}

/// Property 7: Memory safety (Kani model checking) artifacts.
pub(super) fn memory_safety_artifacts() -> Vec<VerificationArtifact> {
    vec![
        VerificationArtifact {
            description: "491 Kani harnesses across nn workspace",
            file: "crates/",
            properties: &[6],
            level: VerificationLevel::KaniProven,
        },
        VerificationArtifact {
            description: "Kani bounds proofs (arithmetic, structural, ULP)",
            file: "crates/nn-core/src/kani_bounds.rs",
            properties: &[6],
            level: VerificationLevel::KaniProven,
        },
        VerificationArtifact {
            description: "Kani backward derivative proofs (35 harnesses)",
            file: "crates/nn-autodiff/src/kani_backward_proofs.rs",
            properties: &[6],
            level: VerificationLevel::KaniProven,
        },
    ]
}

/// Property 8: Correct implementation (ay SMT + NY) artifacts.
pub(super) fn correctness_artifacts() -> Vec<VerificationArtifact> {
    vec![
        VerificationArtifact {
            description: "ay SMT proofs (15/15 linear kernel assertions Proven)",
            file: "crates/nn-verify/src/ay/",
            properties: &[7],
            level: VerificationLevel::SmtProven,
        },
        VerificationArtifact {
            description: "14 BOUNDS_REGISTRY entries with analytical bounds",
            file: "crates/nn-verify/src/ay/prove_dispatch.rs",
            properties: &[7],
            level: VerificationLevel::SmtProven,
        },
        VerificationArtifact {
            description: "33 KernelConfigs verified (30 Pending + 3 Fusion)",
            file: "crates/nn-verify/examples/verify_all/configs.rs",
            properties: &[7],
            level: VerificationLevel::CrownPartial,
        },
    ]
}

/// Cross-cutting artifacts spanning multiple properties.
pub(super) fn cross_cutting_artifacts() -> Vec<VerificationArtifact> {
    vec![
        VerificationArtifact {
            description: "Pipeline composition verification framework",
            file: "crates/nn-tts-verify/src/pipeline.rs",
            // Framework covers audio quality (0,1), intelligibility (2),
            // speaker consistency (3) via embedding bounds, and
            // streaming (5) via pipeline CROWN bounds.
            properties: &[0, 1, 2, 3, 5],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description: "Prosody disentanglement CROWN verification",
            file: "crates/nn-tts-verify/src/disentanglement.rs",
            properties: &[2],
            level: VerificationLevel::CrownPartial,
        },
        VerificationArtifact {
            description: "CROWN-verified fairness bounds (per-group)",
            file: "crates/nn-tts-verify/src/fairness_crown.rs",
            // Fairness verifies equal quality across demographics (properties 0,1)
            // — not speaker embedding distance (property 3). Separate concern.
            properties: &[0, 1],
            level: VerificationLevel::CrownPartial,
        },
    ]
}

/// Full model verification artifacts (end-to-end CROWN composition).
pub(super) fn full_model_artifacts() -> Vec<VerificationArtifact> {
    vec![
        VerificationArtifact {
            description: "Silero VAD full model composition (7 tests)",
            file: "crates/nn-verify/tests/compose_silero_vad_full.rs",
            properties: &[6, 7],
            level: VerificationLevel::CrownProven,
        },
        VerificationArtifact {
            description: "Whisper encoder+decoder composition (11 tests)",
            file: "crates/nn-verify/tests/compose_whisper_full.rs",
            properties: &[6, 7],
            level: VerificationLevel::CrownProven,
        },
        VerificationArtifact {
            description: "D=192 production-scale CROWN composition + moonshot bridge (6 tests)",
            file: "crates/nn-tts-verify/src/pipeline_tests.rs",
            // P1 (non-silence) and P6 (streaming) proven at D=192 with NY.
            properties: &[0, 5],
            level: VerificationLevel::CrownProven,
        },
        VerificationArtifact {
            description: "Qwen3 decoder composition (5 tests)",
            file: "crates/nn-verify/tests/compose_qwen3_decoder.rs",
            properties: &[6, 7],
            level: VerificationLevel::CrownProven,
        },
    ]
}
