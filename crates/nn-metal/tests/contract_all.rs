// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![cfg(feature = "verify")]
#![allow(dead_code, unreachable_pub)]

//! Consolidated cross-backend contract tests: GPU output within NY
//! verified bounds for each tensor op.

mod test_utils;

#[path = "contract/causal_conv1d.rs"]
mod contract_causal_conv1d;
#[path = "contract/composed.rs"]
mod contract_composed;
#[path = "contract/conv1d.rs"]
mod contract_conv1d;
#[path = "contract/conv2d.rs"]
mod contract_conv2d;
#[path = "contract/conv_transpose_1d.rs"]
mod contract_conv_transpose_1d;
#[path = "contract/decoder_composed.rs"]
mod contract_decoder_composed;
#[path = "contract/embedding.rs"]
mod contract_embedding;
#[path = "contract/glu.rs"]
mod contract_glu;
#[path = "contract/group_norm.rs"]
mod contract_group_norm;
#[path = "contract/index_select.rs"]
mod contract_index_select;
#[path = "contract/instance_norm_decomposed.rs"]
mod contract_instance_norm_decomposed;
#[path = "contract/linear.rs"]
mod contract_linear;
#[path = "contract/matmul.rs"]
mod contract_matmul;
#[path = "contract/narrow.rs"]
mod contract_narrow;
#[path = "contract/softmax.rs"]
mod contract_softmax;
#[path = "contract/spectral_decoder.rs"]
mod contract_spectral_decoder;
