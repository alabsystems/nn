// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Standard tensor input names for builder + dispatch APIs.
//!
//! Tensor-level builders (`build_conv1d`, `build_linear`, `build_lstm_cell_decomposed`,
//! etc.) and model forward passes (`SileroVad::forward`, `DemucsTemporalDecoder::forward`)
//! must agree on input naming. Previously three conventions coexisted (`"data"`, `"input"`,
//! `"x"`), causing silent dispatch failures on key mismatch.
//!
//! All tensor-level builders now use these constants. Scalar kernel builders
//! (`build_snake_scalar_kernel`, etc.) use `"x"` internally and are not affected.
//!
//! Part of #790.

/// Primary data input tensor.
pub const DATA: &str = "data";

/// Weight matrix input.
pub const WEIGHT: &str = "weight";

/// Bias vector input.
pub const BIAS: &str = "bias";

/// LSTM hidden state input.
pub const HIDDEN_STATE: &str = "hidden_state";

/// LSTM cell state input.
pub const CELL_STATE: &str = "cell_state";

/// LSTM input-hidden weight matrix.
pub const WEIGHT_IH: &str = "weight_ih";

/// LSTM hidden-hidden weight matrix.
pub const WEIGHT_HH: &str = "weight_hh";

/// Skip connection tensor (Demucs decoder).
pub const SKIP: &str = "skip";
