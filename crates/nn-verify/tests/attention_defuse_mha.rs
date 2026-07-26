// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the per-head multi-head-attention de-fusion that routes
//! the `Q@Kᵀ` score through NY's tight zonotope bound. See
//! `graph_tensor_attention_defuse.rs`.
//!
//! Covers: (1) de-fusion FIRES on a standard `add_multi_head_attention` (per-head
//! 2-D MatMul score nodes appear); (2) sound ENCLOSURE of the true forward over
//! sampled inputs (standard + causal); (3) CROWN does not error; (4) tighter
//! per-head score vs plain-IBP; (5) direct-2-D (`Linear->add_attention`) de-fuses
//! and stays 2-D; (6) cross-attention (distinct Q/KV base) does NOT de-fuse.
//!
//! All configs keep `num_heads*seq² <= 4096` so the de-fusion size gate fires.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding,
};
use ndarray::{Array2, ArrayD, IxDyn};

const SEQ: usize = 16;
const D: usize = 64;
const H: usize = 4;
const HD: usize = D / H; // 16

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

/// Deterministic ~U(-0.3, 0.3) `[D,D]` weight.
fn rand_w(seed: u64) -> ArrayD<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut data = vec![0.0f32; D * D];
    for v in data.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
        *v = (u - 0.5) * 0.6;
    }
    ArrayD::from_shape_vec(IxDyn(&[D, D]), data).unwrap()
}

fn build_real_mha(mask: AttentionMask) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("mha");
    let x = b.add_input("x", &[SEQ, D]);
    let qw = b.add_input("qw", &[D, D]);
    let kw = b.add_input("kw", &[D, D]);
    let vw = b.add_input("vw", &[D, D]);
    let ow = b.add_input("ow", &[D, D]);
    let out = b
        .add_multi_head_attention(x, qw, kw, vw, ow, H, mask, &[SEQ, D])
        .expect("valid MHA");
    b.build(out).expect("valid")
}

fn mha_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(rand_w(1)),
        TensorParamBinding::ConstantTensor(rand_w(2)),
        TensorParamBinding::ConstantTensor(rand_w(3)),
        TensorParamBinding::ConstantTensor(rand_w(4)),
    ]
}

/// (1) De-fusion fires: per-head 2-D score nodes appear in the translated graph.
#[test]
fn defuse_fires_on_standard_mha() {
    let g = tensor_kernel_to_graph(&build_real_mha(AttentionMask::Standard), &mha_bindings()).unwrap();
    let n_score = g.node_names().iter().filter(|n| n.contains("_dfmha_s")).count();
    assert_eq!(n_score, H, "expected one per-head score node per head");
}

/// (1c) Causal de-fusion fires with per-head causal-softmax nodes; propagates.
#[test]
fn defuse_fires_on_causal_mha() {
    let g = tensor_kernel_to_graph(&build_real_mha(AttentionMask::Causal), &mha_bindings()).unwrap();
    let n_probs = g.node_names().iter().filter(|n| n.contains("_dfmha_p")).count();
    assert_eq!(n_probs, H);
    let out = g.propagate_ibp(&uniform(&[SEQ, D], 0.05)).unwrap();
    assert_eq!(out.lower_upper().0.shape(), &[SEQ, D]);
    let (lo, hi) = out.lower_upper();
    assert!(lo.iter().chain(hi.iter()).all(|v| v.is_finite()));
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u);
    }
}

/// (3) CROWN through the de-fused graph must not error: succeeds or falls back.
#[test]
fn defuse_crown_does_not_error() {
    let g = tensor_kernel_to_graph(&build_real_mha(AttentionMask::Standard), &mha_bindings()).unwrap();
    let input = uniform(&[SEQ, D], 0.05);
    let (_m, out, _f) =
        propagate_with_crown_fallback(&g, &input).expect("CROWN/IBP propagation must not error");
    let (lo, hi) = out.lower_upper();
    assert!(lo.iter().chain(hi.iter()).all(|v| v.is_finite()));
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u);
    }
}

/// Independent reference multi-head SDPA forward + output projection.
fn ref_forward(
    x: &Array2<f32>,
    qw: &Array2<f32>,
    kw: &Array2<f32>,
    vw: &Array2<f32>,
    ow: &Array2<f32>,
    causal: bool,
) -> Array2<f32> {
    let q = x.dot(&qw.t());
    let k = x.dot(&kw.t());
    let v = x.dot(&vw.t());
    let scale = 1.0 / (HD as f32).sqrt();
    let mut concat = Array2::<f32>::zeros((SEQ, D));
    for h in 0..H {
        let cols = h * HD..(h + 1) * HD;
        let qh = q.slice(ndarray::s![.., cols.clone()]);
        let kh = k.slice(ndarray::s![.., cols.clone()]);
        let vh = v.slice(ndarray::s![.., cols.clone()]);
        let scores = qh.dot(&kh.t()) * scale;
        for i in 0..SEQ {
            let end = if causal { i + 1 } else { SEQ };
            let mut mx = f32::NEG_INFINITY;
            for j in 0..end {
                mx = mx.max(scores[[i, j]]);
            }
            let mut sum = 0.0;
            let mut p = vec![0.0f32; end];
            for j in 0..end {
                let e = (scores[[i, j]] - mx).exp();
                p[j] = e;
                sum += e;
            }
            for j in 0..end {
                let w = p[j] / sum;
                for d in 0..HD {
                    concat[[i, h * HD + d]] += w * vh[[j, d]];
                }
            }
        }
    }
    concat.dot(&ow.t())
}

/// (2) Soundness: the de-fused IBP bound encloses the true forward at every
/// sampled point in the input box (standard and causal).
#[test]
fn defuse_bound_encloses_true_forward() {
    for causal in [false, true] {
        let mask = if causal {
            AttentionMask::Causal
        } else {
            AttentionMask::Standard
        };
        let g = tensor_kernel_to_graph(&build_real_mha(mask), &mha_bindings()).unwrap();
        let r = 0.05f32;
        let out = g.propagate_ibp(&uniform(&[SEQ, D], r)).unwrap();
        let (lo, hi) = out.lower_upper();
        let lo = lo.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let hi = hi.clone().into_dimensionality::<ndarray::Ix2>().unwrap();

        let qw = rand_w(1).into_dimensionality::<ndarray::Ix2>().unwrap();
        let kw = rand_w(2).into_dimensionality::<ndarray::Ix2>().unwrap();
        let vw = rand_w(3).into_dimensionality::<ndarray::Ix2>().unwrap();
        let ow = rand_w(4).into_dimensionality::<ndarray::Ix2>().unwrap();

        for seed in 0..150u64 {
            let mut s = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            let mut xd = vec![0.0f32; SEQ * D];
            for e in xd.iter_mut() {
                s = s.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
                *e = (u * 2.0 - 1.0) * r;
            }
            let x = Array2::from_shape_vec((SEQ, D), xd).unwrap();
            let y = ref_forward(&x, &qw, &kw, &vw, &ow, causal);
            for i in 0..SEQ {
                for j in 0..D {
                    let tol = 1e-4 + 1e-3 * y[[i, j]].abs();
                    assert!(
                        y[[i, j]] >= lo[[i, j]] - tol && y[[i, j]] <= hi[[i, j]] + tol,
                        "enclosure violation (causal={causal}) at [{i},{j}]: y={} not in [{},{}]",
                        y[[i, j]],
                        lo[[i, j]],
                        hi[[i, j]]
                    );
                }
            }
        }
    }
}

/// (4) Tightness: the de-fused per-head score bound (zonotope) is strictly tighter
/// than the plain-IBP multi-head 3-D score bound, on a realistic LayerNorm base.
#[test]
fn defuse_score_is_tighter_than_ibp() {
    let mut bz = TensorBlockBuilder::new("scores_zono");
    let x = bz.add_input("x", &[SEQ, D]);
    let eps = bz.add_input("eps", &[1]);
    let lnw = bz.add_input("lnw", &[D]);
    let lnb = bz.add_input("lnb", &[D]);
    let base = bz.add_layer_norm(x, eps, 1, lnw, lnb, &[SEQ, D]);
    let scale = 1.0 / (HD as f32).sqrt();
    let mut hs = Vec::new();
    for h in 0..H {
        let qwh = bz.add_input(&format!("qw{h}"), &[HD, D]);
        let kwh = bz.add_input(&format!("kw{h}"), &[HD, D]);
        let qh = bz.add_linear(base, qwh, None, &[SEQ, HD]);
        let kh = bz.add_linear(base, kwh, None, &[SEQ, HD]);
        hs.push(bz.add_matmul(qh, kh, true, Some(scale), &[SEQ, SEQ]));
    }
    let scores_z = bz.add_stack(&hs, 0, &[H, SEQ, SEQ]);
    let def_z = bz.build(scores_z).expect("valid");

    let mut bi = TensorBlockBuilder::new("scores_ibp");
    let x = bi.add_input("x", &[SEQ, D]);
    let eps = bi.add_input("eps", &[1]);
    let lnw = bi.add_input("lnw", &[D]);
    let lnb = bi.add_input("lnb", &[D]);
    let base = bi.add_layer_norm(x, eps, 1, lnw, lnb, &[SEQ, D]);
    let qw = bi.add_input("qw", &[D, D]);
    let kw = bi.add_input("kw", &[D, D]);
    let q = bi.add_linear(base, qw, None, &[SEQ, D]);
    let k = bi.add_linear(base, kw, None, &[SEQ, D]);
    let q = bi.add_reshape(q, &[SEQ, H, HD]);
    let k = bi.add_reshape(k, &[SEQ, H, HD]);
    let q = bi.add_transpose(q, &[1, 0, 2], &[H, SEQ, HD]);
    let k = bi.add_transpose(k, &[1, 0, 2], &[H, SEQ, HD]);
    let scores_i = bi.add_matmul(q, k, true, Some(scale), &[H, SEQ, SEQ]);
    let def_i = bi.build(scores_i).expect("valid");

    let qw = rand_w(1);
    let kw = rand_w(2);
    let mut bind_z = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D]), 0.0f32)),
    ];
    for h in 0..H {
        let split = |w: &ArrayD<f32>| {
            w.view()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap()
                .slice(ndarray::s![h * HD..(h + 1) * HD, ..])
                .to_owned()
                .into_dyn()
        };
        bind_z.push(TensorParamBinding::ConstantTensor(split(&qw)));
        bind_z.push(TensorParamBinding::ConstantTensor(split(&kw)));
    }
    let bind_i = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D]), 0.0f32)),
        TensorParamBinding::ConstantTensor(qw),
        TensorParamBinding::ConstantTensor(kw),
    ];

    let input = uniform(&[SEQ, D], 0.05);
    let wz = width(&tensor_kernel_to_graph(&def_z, &bind_z).unwrap().propagate_ibp(&input).unwrap());
    let wi = width(&tensor_kernel_to_graph(&def_i, &bind_i).unwrap().propagate_ibp(&input).unwrap());
    assert!(wz <= wi + 1e-4, "zonotope score {wz} must be <= IBP score {wi}");
    assert!(wz * 2.0 <= wi, "expected >= 2x score tightening, got {wz} vs {wi}");
}

/// (5) Direct-2-D self-attention (`Linear(base) -> add_attention` on full-D
/// tensors) de-fuses to a single head and stays 2-D `[S,D]`; sound enclosure with
/// an EXPLICIT scale passed through.
#[test]
fn defuse_direct_2d_self_attention() {
    let custom_scale = 1.0 / (HD as f32).sqrt(); // != 1/sqrt(D)
    let mut b = TensorBlockBuilder::new("direct2d");
    let x = b.add_input("x", &[SEQ, D]);
    let eps = b.add_input("eps", &[1]);
    let lnw = b.add_input("lnw", &[D]);
    let lnb = b.add_input("lnb", &[D]);
    let base = b.add_layer_norm(x, eps, 1, lnw, lnb, &[SEQ, D]);
    let qw = b.add_input("qw", &[D, D]);
    let kw = b.add_input("kw", &[D, D]);
    let vw = b.add_input("vw", &[D, D]);
    let q = b.add_linear(base, qw, None, &[SEQ, D]);
    let k = b.add_linear(base, kw, None, &[SEQ, D]);
    let v = b.add_linear(base, vw, None, &[SEQ, D]);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(custom_scale), &[SEQ, D]);
    let def = b.build(attn).expect("valid");
    let bind = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D]), 0.0f32)),
        TensorParamBinding::ConstantTensor(rand_w(1)),
        TensorParamBinding::ConstantTensor(rand_w(2)),
        TensorParamBinding::ConstantTensor(rand_w(3)),
    ];
    let g = tensor_kernel_to_graph(&def, &bind).unwrap();
    let n_score = g.node_names().iter().filter(|n| n.contains("_dfmha_s")).count();
    assert_eq!(n_score, 1, "direct-2-D self-attention should de-fuse to one score node");
    let input = uniform(&[SEQ, D], 0.05);
    let out = g.propagate_ibp(&input).unwrap();
    assert_eq!(out.lower_upper().0.shape(), &[SEQ, D], "output stays 2-D [S,D]");

    let (lo, hi) = out.lower_upper();
    let lo = lo.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let hi = hi.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let qw = rand_w(1).into_dimensionality::<ndarray::Ix2>().unwrap();
    let kw = rand_w(2).into_dimensionality::<ndarray::Ix2>().unwrap();
    let vw = rand_w(3).into_dimensionality::<ndarray::Ix2>().unwrap();
    for seed in 0..120u64 {
        let mut s = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        let mut xd = vec![0.0f32; SEQ * D];
        for e in xd.iter_mut() {
            s = s.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
            *e = (u * 2.0 - 1.0) * 0.05;
        }
        let xx = Array2::from_shape_vec((SEQ, D), xd).unwrap();
        let mut ln = Array2::<f32>::zeros((SEQ, D));
        for i in 0..SEQ {
            let row = xx.row(i);
            let mean = row.sum() / D as f32;
            let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / D as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..D {
                ln[[i, j]] = (xx[[i, j]] - mean) * inv;
            }
        }
        let q = ln.dot(&qw.t());
        let k = ln.dot(&kw.t());
        let v = ln.dot(&vw.t());
        let scores = q.dot(&k.t()) * custom_scale;
        let mut y = Array2::<f32>::zeros((SEQ, D));
        for i in 0..SEQ {
            let mut mx = f32::NEG_INFINITY;
            for j in 0..SEQ {
                mx = mx.max(scores[[i, j]]);
            }
            let mut sum = 0.0;
            let mut p = vec![0.0f32; SEQ];
            for j in 0..SEQ {
                let e = (scores[[i, j]] - mx).exp();
                p[j] = e;
                sum += e;
            }
            for j in 0..SEQ {
                let w = p[j] / sum;
                for d in 0..D {
                    y[[i, d]] += w * v[[j, d]];
                }
            }
        }
        for i in 0..SEQ {
            for j in 0..D {
                let tol = 1e-3 + 5e-3 * y[[i, j]].abs();
                assert!(
                    y[[i, j]] >= lo[[i, j]] - tol && y[[i, j]] <= hi[[i, j]] + tol,
                    "direct-2-D enclosure violation at [{i},{j}]: y={} not in [{},{}]",
                    y[[i, j]],
                    lo[[i, j]],
                    hi[[i, j]]
                );
            }
        }
    }
}

/// (6) Cross-attention (Q from one base, K/V from another) must NOT de-fuse.
#[test]
fn cross_attention_does_not_defuse() {
    let mut b = TensorBlockBuilder::new("cross");
    let q_in = b.add_input("q_in", &[SEQ, D]);
    let kv_in = b.add_input("kv_in", &[SEQ, D]);
    let qw = b.add_input("qw", &[D, D]);
    let kw = b.add_input("kw", &[D, D]);
    let vw = b.add_input("vw", &[D, D]);
    let ow = b.add_input("ow", &[D, D]);
    let out = b
        .add_multi_head_cross_attention(q_in, kv_in, qw, kw, vw, ow, H, AttentionMask::Standard, &[SEQ, D])
        .expect("valid cross-attn");
    let def = b.build(out).expect("valid");
    let bind = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(rand_w(1)),
        TensorParamBinding::ConstantTensor(rand_w(2)),
        TensorParamBinding::ConstantTensor(rand_w(3)),
        TensorParamBinding::ConstantTensor(rand_w(4)),
    ];
    let g = tensor_kernel_to_graph(&def, &bind).unwrap();
    let n_score = g.node_names().iter().filter(|n| n.contains("_dfmha_s")).count();
    assert_eq!(n_score, 0, "cross-attention (distinct Q/KV base) must not de-fuse");
    let input = uniform(&[2, SEQ, D], 0.05);
    let out = g.propagate_ibp(&input).unwrap();
    let (lo, hi) = out.lower_upper();
    assert!(lo.iter().chain(hi.iter()).all(|v| v.is_finite()));
}

/// (7) Safety ceiling: attention above the (env-tunable) `H·S²` ceiling keeps the
/// fused node. Force a tiny ceiling so the standard config (4·16²=1024) trips it,
/// exercising the bail path. The default ceiling is generous (1<<20) so all real
/// models de-fuse — at realistic sizes the de-fused path is lighter than fused.
#[test]
fn defuse_size_ceiling_keeps_large_fused() {
    std::env::set_var("NN_VERIFY_DEFUSE_SCORE_BUDGET", "100"); // 1024 > 100 -> no de-fuse
    let g = tensor_kernel_to_graph(&build_real_mha(AttentionMask::Standard), &mha_bindings()).unwrap();
    std::env::remove_var("NN_VERIFY_DEFUSE_SCORE_BUDGET");

    let n_score = g.node_names().iter().filter(|n| n.contains("_dfmha_s")).count();
    assert_eq!(n_score, 0, "attention over the size ceiling must keep the fused node");
    let out = g.propagate_ibp(&uniform(&[SEQ, D], 0.05)).unwrap();
    assert_eq!(out.lower_upper().0.shape(), &[SEQ, D]);
}
