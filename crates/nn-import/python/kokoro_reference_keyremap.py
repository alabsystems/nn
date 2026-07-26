#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# SPDX-License-Identifier: Apache-2.0

"""Weight key remapping for Kokoro PyTorch → Rust nn VarBuilder.

Extracted from kokoro_reference.py to stay under the 500-line file limit.
Part of #2466, #2218, #2681.
"""

import logging
import re

import torch

log = logging.getLogger(__name__)


def decompose_weight_norm(state_dict):
    """Remove weight_norm parametrization by merging weight_g and weight_v.

    PyTorch weight_norm stores weights as:
      - name.weight_g: [out_ch, 1, ...] magnitude
      - name.weight_v: [out_ch, in_ch, ...] direction

    The actual weight is: weight = weight_v * (weight_g / ||weight_v||)

    This function merges them into a single .weight tensor and removes
    the _g/_v suffixes.
    """
    decomposed = {}
    g_keys = {k for k in state_dict if k.endswith(".weight_g")}
    v_keys = {k for k in state_dict if k.endswith(".weight_v")}

    # Find matched pairs
    merged_bases = set()
    for gk in g_keys:
        base = gk[:-len(".weight_g")]
        vk = base + ".weight_v"
        if vk in v_keys:
            g = state_dict[gk]
            v = state_dict[vk]
            # weight = v * (g / ||v||) where norm is over all dims except 0
            dims = tuple(range(1, v.dim()))
            norm = v.norm(dim=dims, keepdim=True)
            weight = v * (g / norm.clamp(min=1e-12))
            decomposed[base + ".weight"] = weight.contiguous()
            merged_bases.add(base)
            log.info("Decomposed weight_norm: %s (%s)", base, list(weight.shape))

    # Copy all non-weight_norm keys
    for k, v in state_dict.items():
        if k.endswith(".weight_g") or k.endswith(".weight_v"):
            base = k.rsplit(".", 1)[0]
            if base.endswith(".weight"):
                base = base[:-len(".weight")]
            else:
                base = k[:-len(".weight_g")] if k.endswith(".weight_g") else k[:-len(".weight_v")]
            if base in merged_bases:
                continue
        decomposed[k] = v.contiguous()

    log.info("Weight norm decomposition: %d -> %d tensors (%d merged)",
             len(state_dict), len(decomposed), len(merged_bases))
    return decomposed


def _remap_encoder_keys(key):
    """Remap bert.* → plbert.* and text_encoder.cnn.* → convs/norms.

    Special case: bert.weight [d_en, hidden] and bert.bias [d_en] are the
    PlBert→d_en Linear projection, NOT PlBert parameters. They map to
    bert_encoder.weight/bias in Rust. All other bert.* keys (bert.encoder.*,
    bert.embeddings.*, bert.pooler.*) map to plbert.*. Part of #2691.
    """
    if key in ("bert.weight", "bert.bias"):
        return "bert_encoder." + key[len("bert."):]
    if key.startswith("bert.") and not key.startswith("bert_encoder."):
        return "plbert." + key[len("bert."):]
    m = re.match(r'^text_encoder\.cnn\.(\d+)\.0\.(weight|bias)$', key)
    if m:
        return f"text_encoder.convs.{m.group(1)}.{m.group(2)}"
    m = re.match(r'^text_encoder\.cnn\.(\d+)\.1\.(gamma|beta)$', key)
    if m:
        suffix = "weight" if m.group(2) == "gamma" else "bias"
        return f"text_encoder.norms.{m.group(1)}.{suffix}"
    return key


def _remap_prosody_keys(key):
    """Remap predictor.text_encoder/duration_proj/lstm/duration → prosody_predictor."""
    m = re.match(r'^predictor\.text_encoder\.lstms\.(\d+)\.(.+)$', key)
    if m:
        idx, rest = int(m.group(1)), m.group(2)
        if idx in (0, 2, 4):
            return f"prosody_predictor.duration.lstms.{idx // 2}.{rest}"
        if idx in (1, 3, 5):
            return f"prosody_predictor.duration.norms.{(idx - 1) // 2}.{rest}"
    m = re.match(r'^predictor\.duration_proj\.linear_layer\.(weight|bias)$', key)
    if m:
        return f"prosody_predictor.duration.duration_proj.{m.group(1)}"
    if key.startswith("predictor.lstm."):
        return "prosody_predictor.lstm." + key[len("predictor.lstm."):]
    # Kokoro v1.0: predictor.duration_lstm → prosody_predictor.lstm (Part of #2691).
    if key.startswith("predictor.duration_lstm."):
        return "prosody_predictor.lstm." + key[len("predictor.duration_lstm."):]
    if key.startswith("predictor.duration."):
        return "prosody_predictor.duration." + key[len("predictor.duration."):]
    return key


_ADAIN_LAYER_RENAME = {
    "conv1": "c1", "conv2": "c2",
    "norm1": "n1", "norm2": "n2",
    "conv1x1": "skip",
}


def _remap_f0_energy_keys(key):
    """Remap F0/N AdainResBlk1d keys. Shared BiLSTM passes through unchanged.

    The shared BiLSTM uses PyTorch-native names (weight_ih_l0, *_reverse)
    which BiLstm::load() handles directly. No transformation needed.
    Part of #2691.
    """
    for prefix in ("predictor.F0.", "predictor.N."):
        m = re.match(
            rf'^{re.escape(prefix)}(\d+)\.(conv1|conv2|norm1|norm2|conv1x1)\.(.+)$',
            key,
        )
        if m:
            idx, layer, rest = m.group(1), m.group(2), m.group(3)
            return f"{prefix}{idx}.{_ADAIN_LAYER_RENAME.get(layer, layer)}.{rest}"
    return key


def _remap_decoder_keys(key):
    """Remap decoder keys to match Rust FullDecoder + Generator structure.

    - decoder.asr_res.0.* → decoder.asr_res.* (strip extra index)
    - decoder.{encode,decode}.norm*.fc.* → .style_linear.*
    - decoder.{ups,resblocks,noise_res,noise_convs,conv_post}.* → decoder.generator.*
      (PyTorch nests these directly under decoder; Rust nests under decoder.generator)
    Part of #2691.
    """
    m = re.match(r'^decoder\.asr_res\.0\.(weight|bias)$', key)
    if m:
        return f"decoder.asr_res.{m.group(1)}"
    m = re.match(
        r'^(decoder\.(?:encode|decode\.\d+))\.norm([12])\.fc\.(weight|bias)$', key,
    )
    if m:
        return f"{m.group(1)}.norm{m.group(2)}.style_linear.{m.group(3)}"
    # Generator sub-modules: PyTorch stores directly under decoder.*,
    # Rust Generator loads under decoder.generator.*
    _GENERATOR_PREFIXES = ("decoder.ups.", "decoder.resblocks.", "decoder.noise_res.",
                           "decoder.noise_convs.", "decoder.conv_post.", "decoder.m_source.")
    for gp in _GENERATOR_PREFIXES:
        if key.startswith(gp):
            return "decoder.generator." + key[len("decoder."):]
    return key


def _remap_resblock_paths(key):
    """Remap ResBlock paths.{i}.{layer}.* to Rust convs1/convs2/adain1/adain2/alpha naming.

    PyTorch ResBlock stores dilated conv paths as:
      paths.{i}.c1/c2   → Rust convs1.{i}/convs2.{i}
      paths.{i}.n1/n2   → Rust adain1.{i}/adain2.{i}
      paths.{i}.s1.alpha → Rust alpha1.{i} (alpha tensor, no sub-key)
      paths.{i}.s2.alpha → Rust alpha2.{i}
    Part of #2691.
    """
    # paths.{i}.c1/c2 → convs1/convs2.{i}
    m = re.match(r'^(.+)\.paths\.(\d+)\.(c[12])\.(.+)$', key)
    if m:
        prefix, idx, conv, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        rust_name = "convs1" if conv == "c1" else "convs2"
        return f"{prefix}.{rust_name}.{idx}.{rest}"
    # paths.{i}.n1/n2 → adain1/adain2.{i}
    m = re.match(r'^(.+)\.paths\.(\d+)\.(n[12])\.(.+)$', key)
    if m:
        prefix, idx, norm, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        rust_name = "adain1" if norm == "n1" else "adain2"
        return f"{prefix}.{rust_name}.{idx}.{rest}"
    # paths.{i}.s1.alpha / paths.{i}.s2.alpha → alpha1.{i} / alpha2.{i}
    m = re.match(r'^(.+)\.paths\.(\d+)\.(s[12])\.alpha$', key)
    if m:
        prefix, idx, snake = m.group(1), m.group(2), m.group(3)
        rust_name = "alpha1" if snake == "s1" else "alpha2"
        return f"{prefix}.{rust_name}.{idx}"
    return key


_DROP_PATTERN = re.compile(r'\.(norm[12]|adain[12]\.\d+)\.norm\.(weight|bias)$')
_REMAP_CHAIN = [_remap_encoder_keys, _remap_prosody_keys, _remap_f0_energy_keys, _remap_decoder_keys, _remap_resblock_paths]


def _add_synthetic_weights(out):
    """Add identity weights for layers missing in v1.0 Kokoro."""
    d_en = 512
    if "text_encoder.lstm.linear.weight" not in out:
        out["text_encoder.lstm.linear.weight"] = torch.eye(d_en, dtype=torch.float32)
        out["text_encoder.lstm.linear.bias"] = torch.zeros(d_en, dtype=torch.float32)
        log.info("Added synthetic: text_encoder.lstm.linear (identity %dx%d)", d_en, d_en)

    gen_ch = 512
    if "decoder.generator.conv_pre.weight" not in out:
        kernel = torch.zeros(gen_ch, gen_ch, 7, dtype=torch.float32)
        for c in range(gen_ch):
            kernel[c, c, 3] = 1.0
        out["decoder.generator.conv_pre.weight"] = kernel
        out["decoder.generator.conv_pre.bias"] = torch.zeros(gen_ch, dtype=torch.float32)
        log.info("Added synthetic: decoder.generator.conv_pre (identity %dx%dx7)", gen_ch, gen_ch)


def _squeeze_projection_weights(out):
    """Squeeze Conv1d [out, in, 1] → Linear [out, in] for F0/N projection layers."""
    for proj_key in ("predictor.F0_proj.weight", "predictor.N_proj.weight"):
        if proj_key in out and out[proj_key].dim() == 3 and out[proj_key].shape[-1] == 1:
            out[proj_key] = out[proj_key].squeeze(-1)
            log.info("Squeezed %s: Conv1d → Linear [%s]", proj_key, list(out[proj_key].shape))


def remap_keys_for_rust(state_dict):
    """Remap PyTorch KModel state_dict keys to Rust nn model expectations.

    Transforms the PyTorch key namespace to match what the Rust VarBuilder
    loaders expect. See _remap_encoder_keys, _remap_prosody_keys,
    _remap_f0_energy_keys, _remap_decoder_keys for individual transforms.
    Also adds synthetic identity weights and squeezes Conv1d projections.

    Drops PyTorch InstanceNorm affine params (*.norm.weight/bias inside
    adain blocks) since Rust AdaIn doesn't use affine InstanceNorm.
    """
    out = {}
    dropped = 0

    for key, val in state_dict.items():
        if _DROP_PATTERN.search(key):
            dropped += 1
            continue
        new_key = key
        for fn in _REMAP_CHAIN:
            new_key = fn(new_key)
        out[new_key] = val.contiguous()

    _add_synthetic_weights(out)
    _squeeze_projection_weights(out)
    log.info("Key remap: %d input → %d output keys (%d dropped InstanceNorm affine)",
             len(state_dict), len(out), dropped)
    return out


def _self_test():
    """Verify shared BiLSTM keys pass through unchanged (regression test for #2691)."""
    _SHARED_BILSTM_KEYS = [
        "predictor.shared.weight_ih_l0",
        "predictor.shared.weight_hh_l0",
        "predictor.shared.bias_ih_l0",
        "predictor.shared.bias_hh_l0",
        "predictor.shared.weight_ih_l0_reverse",
        "predictor.shared.weight_hh_l0_reverse",
        "predictor.shared.bias_ih_l0_reverse",
        "predictor.shared.bias_hh_l0_reverse",
    ]
    for key in _SHARED_BILSTM_KEYS:
        result = _remap_f0_energy_keys(key)
        assert result == key, (
            f"Shared BiLSTM key should pass through unchanged: "
            f"{key!r} -> {result!r}"
        )

    # Also verify AdainResBlk1d keys still remap correctly
    assert _remap_f0_energy_keys("predictor.F0.0.conv1.weight") == "predictor.F0.0.c1.weight"
    assert _remap_f0_energy_keys("predictor.N.1.norm2.bias") == "predictor.N.1.n2.bias"
    assert _remap_f0_energy_keys("predictor.F0.2.conv1x1.weight") == "predictor.F0.2.skip.weight"


if __name__ == "__main__":
    _self_test()
    print("keyremap self-test passed")
