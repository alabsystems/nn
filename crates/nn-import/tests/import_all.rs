// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated nn-import tests: Kokoro decoder and PyanNet import/parity.

#![allow(dead_code, unreachable_pub)]

#[path = "import/kokoro_decoder_convert.rs"]
mod kokoro_decoder_convert;

#[path = "import/kokoro_decoder_import.rs"]
mod kokoro_decoder_import;

#[path = "import/pyannet_import.rs"]
mod pyannet_import;

#[path = "import/pyannet_l3_parity.rs"]
mod pyannet_l3_parity;

#[path = "import/kokoro_segment_convert.rs"]
mod kokoro_segment_convert;

#[path = "import/kokoro_converter_parity.rs"]
mod kokoro_converter_parity;

#[path = "import/kokoro_convert_builder_parity.rs"]
mod kokoro_convert_builder_parity;

#[path = "import/kokoro_e2e_parity.rs"]
mod kokoro_e2e_parity;

#[path = "import/whisper_e2e_parity.rs"]
mod whisper_e2e_parity;

#[path = "import/gamma_crown_bounds_parity.rs"]
mod gamma_crown_bounds_parity;

#[path = "import/kokoro_real_weights_parity.rs"]
mod kokoro_real_weights_parity;
