// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Score-bound tightening regression: the per-head zonotope de-fused `Q@Kᵀ` score
//! must be meaningfully tighter than the plain-IBP multi-head score, for the
//! attention configs (within the de-fusion size gate) used by DETR / SVTR /
//! table_transformer encoder blocks. Run with `-- --nocapture` to print deltas.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

fn uniform(shape: &[usize], r: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), -r),
        ArrayD::from_elem(IxDyn(shape), r),
    )
    .unwrap()
}
fn width(b: &BoundedTensor) -> f32 {
    let (lo, hi) = b.lower_upper();
    lo.iter().zip(hi.iter()).map(|(l, u)| u - l).fold(0.0, f32::max)
}
fn rand_w(seed: u64, rows: usize, cols: usize) -> ArrayD<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut data = vec![0.0f32; rows * cols];
    for v in data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
        *v = (u - 0.5) * 0.6;
    }
    ArrayD::from_shape_vec(IxDyn(&[rows, cols]), data).unwrap()
}

/// Per-head-2D (zonotope) scores from a LayerNorm base — the de-fused structure.
fn build_zono(seq: usize, d: usize, h: usize) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let hd = d / h;
    let mut b = TensorBlockBuilder::new("z");
    let x = b.add_input("x", &[seq, d]);
    let eps = b.add_input("eps", &[1]);
    let lnw = b.add_input("lnw", &[d]);
    let lnb = b.add_input("lnb", &[d]);
    let base = b.add_layer_norm(x, eps, 1, lnw, lnb, &[seq, d]);
    let scale = 1.0 / (hd as f32).sqrt();
    let qw = rand_w(1, d, d);
    let kw = rand_w(2, d, d);
    let mut hs = Vec::new();
    let mut bind = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
    ];
    for hh in 0..h {
        let qwh = b.add_input(&format!("qw{hh}"), &[hd, d]);
        let kwh = b.add_input(&format!("kw{hh}"), &[hd, d]);
        let qh = b.add_linear(base, qwh, None, &[seq, hd]);
        let kh = b.add_linear(base, kwh, None, &[seq, hd]);
        hs.push(b.add_matmul(qh, kh, true, Some(scale), &[seq, seq]));
        let split = |w: &ArrayD<f32>| {
            w.view().into_dimensionality::<ndarray::Ix2>().unwrap()
                .slice(ndarray::s![hh * hd..(hh + 1) * hd, ..]).to_owned().into_dyn()
        };
        bind.push(TensorParamBinding::ConstantTensor(split(&qw)));
        bind.push(TensorParamBinding::ConstantTensor(split(&kw)));
    }
    let out = b.add_stack(&hs, 0, &[h, seq, seq]);
    (b.build(out).expect("valid"), bind)
}

/// Fused-equivalent 3-D scores (plain IBP) from a LayerNorm base.
fn build_ibp(seq: usize, d: usize, h: usize) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let hd = d / h;
    let mut b = TensorBlockBuilder::new("i");
    let x = b.add_input("x", &[seq, d]);
    let eps = b.add_input("eps", &[1]);
    let lnw = b.add_input("lnw", &[d]);
    let lnb = b.add_input("lnb", &[d]);
    let base = b.add_layer_norm(x, eps, 1, lnw, lnb, &[seq, d]);
    let qw = b.add_input("qw", &[d, d]);
    let kw = b.add_input("kw", &[d, d]);
    let q = b.add_linear(base, qw, None, &[seq, d]);
    let k = b.add_linear(base, kw, None, &[seq, d]);
    let q = b.add_reshape(q, &[seq, h, hd]);
    let k = b.add_reshape(k, &[seq, h, hd]);
    let q = b.add_transpose(q, &[1, 0, 2], &[h, seq, hd]);
    let k = b.add_transpose(k, &[1, 0, 2], &[h, seq, hd]);
    let scale = 1.0 / (hd as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[h, seq, seq]);
    let bind = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        TensorParamBinding::ConstantTensor(rand_w(1, d, d)),
        TensorParamBinding::ConstantTensor(rand_w(2, d, d)),
    ];
    (b.build(scores).expect("valid"), bind)
}

#[test]
fn measure_score_tightening() {
    // Real per-model encoder-block attention dimensions. This measures the
    // zonotope-vs-IBP SCORE width that the de-fusion achieves (the graphs are built
    // explicitly, so the measurement is independent of the runtime de-fusion size
    // gate). `[gated]` marks configs whose `num_heads*seq²` exceeds the default
    // budget (4096), so the live verifier keeps those FUSED (CROWN-safe) — the
    // numbers below show the score tightening they WOULD get, and which a higher
    // `NN_VERIFY_DEFUSE_SCORE_BUDGET` would unlock for IBP-only verification.
    let configs = [
        ("DETR-encoder-small  d=64  h=4 seq=16       ", 16usize, 64usize, 4usize),
        ("DETR-encoder-medium d=128 h=8 seq=32 [gated]", 32, 128, 8),
        ("SVTR-block          d=64  h=8 seq=16       ", 16, 64, 8),
        ("table_transformer   d=256 h=8 seq=64 [gated]", 64, 256, 8),
    ];
    for (name, seq, d, h) in configs {
        let (dz, bz) = build_zono(seq, d, h);
        let (di, bi) = build_ibp(seq, d, h);
        let input = uniform(&[seq, d], 0.05);
        let wz = width(&tensor_kernel_to_graph(&dz, &bz).unwrap().propagate_ibp(&input).unwrap());
        let wi = width(&tensor_kernel_to_graph(&di, &bi).unwrap().propagate_ibp(&input).unwrap());
        eprintln!(
            "MEASURE {name}: score width  IBP(fused)={wi:10.4}  zonotope(de-fused)={wz:10.4}  ratio={:6.2}x tighter",
            wi / wz.max(1e-12)
        );
        assert!(wz <= wi + 1e-4, "{name}: de-fused score must not be looser");
        assert!(
            wz * 2.0 <= wi,
            "{name}: expected >= 2x score tightening, got {:.2}x",
            wi / wz.max(1e-12)
        );
    }
}
