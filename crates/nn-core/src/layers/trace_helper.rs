// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Re-export of `traced_forward` from the canonical definition in
//! `dyn_tensor/trace.rs`. All nn modules import via `super::traced_forward`.

pub(crate) use crate::dyn_tensor::trace::traced_forward;
