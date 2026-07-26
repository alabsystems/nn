// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated nn-tts-verify tests: DSP, quality metrics, hard bounds,
//! pipeline verification, fairness, and gap detection.

#![allow(dead_code, unreachable_pub)]

#[path = "tts_verify/dsp_tests.rs"]
mod dsp_tests;

#[path = "tts_verify/fairness_crown_integration.rs"]
mod fairness_crown_integration;

#[path = "tts_verify/hard_bounds_tests.rs"]
mod hard_bounds_tests;

#[path = "tts_verify/kokoro_bound_gap_detector.rs"]
mod kokoro_bound_gap_detector;

#[path = "tts_verify/pipeline_crown.rs"]
mod pipeline_crown;

#[path = "tts_verify/pipeline_disentanglement.rs"]
mod pipeline_disentanglement;

#[path = "tts_verify/pipeline_unicode_adversarial.rs"]
mod pipeline_unicode_adversarial;

#[path = "tts_verify/quality_tests.rs"]
mod quality_tests;

#[path = "tts_verify/tts_quality_pipeline_tests.rs"]
mod tts_quality_pipeline_tests;

#[path = "tts_verify/crown_certificate_wiring.rs"]
mod crown_certificate_wiring;

#[cfg(feature = "ny")]
#[path = "tts_verify/dead_neuron_certificate.rs"]
mod dead_neuron_certificate;
