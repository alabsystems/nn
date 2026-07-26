#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# SPDX-License-Identifier: Apache-2.0

"""Generate PyTorch Kokoro reference outputs for L3 parity testing.

Runs Kokoro v1.0 (kokoro package), saves input_ids, style, speed, audio
as safetensors. With --intermediates, saves inter_* tensors for layerwise
parity. With --weights-out, saves state_dict with weight_norm decomposed
and keys remapped for Rust VarBuilder. Part of #2466, #2218.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate Kokoro reference outputs")
    parser.add_argument("--output", required=True, help="Output reference safetensors path")
    parser.add_argument("--weights-out", help="Also save model weights as safetensors (with weight_norm decomposed)")
    parser.add_argument("--voice", default="af_heart", help="Voice name (default: af_heart)")
    parser.add_argument("--text", default="Hello world.", help="Input text")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--intermediates", action="store_true",
                        help="Save intermediate tensors (inter_*) for layerwise parity testing")
    parser.add_argument("--deterministic", action="store_true",
                        help="Zero out SourceModule noise for deterministic output "
                             "(matches nn inference which uses zero noise)")
    return parser.parse_args()


def _find_embedding_module(model):
    """Find the word embedding module in the model for hooking."""
    for name, mod in model.named_modules():
        if "word_embedding" in name.lower() or "embed_tokens" in name.lower():
            return mod
    # Fallback: first Embedding layer.
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            log.info("Hooking first Embedding layer: %s", name)
            return mod
    return None


def _install_input_hook(target_module):
    """Install a forward hook that captures the first input tensor."""
    captured = {}

    def hook_fn(_module, inputs, _output):
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["input_ids"] = inputs[0].detach().cpu().clone()

    handle = target_module.register_forward_hook(hook_fn)
    return captured, handle


def _load_voice_embedding(pipeline, voice):
    """Extract the style/voice embedding tensor from the pipeline."""
    if hasattr(pipeline, "load_voice"):
        return pipeline.load_voice(voice)
    if hasattr(pipeline, "voices") and voice in pipeline.voices:
        return pipeline.voices[voice]
    return None


def _ensure_3d(audio):
    """Reshape audio to [1, 1, T] if needed."""
    if audio.dim() == 1:
        return audio.unsqueeze(0).unsqueeze(0)
    if audio.dim() == 2:
        return audio.unsqueeze(0)
    return audio


def _ensure_2d(tensor):
    """Add batch dim if tensor is 1D -> [1, D]."""
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


from kokoro_reference_keyremap import decompose_weight_norm, remap_keys_for_rust


def _capture_model_inputs(pipeline, text, voice, speed):
    """Run KPipeline and capture input_ids, ref_s (style), and audio.

    Returns (input_ids, style, audio) where:
      - input_ids: [1, T] long tensor of phoneme token IDs
      - style: [1, 256] float tensor — the actual ref_s passed to model.forward()
      - audio: [1, 1, T_audio] float tensor of PCM waveform

    The voice pack is [510, 1, 256]; the pipeline selects pack[len(ps)-1]
    as the ref_s for each segment. We hook the embedding to capture input_ids
    and also monkey-patch model.forward to capture the actual ref_s used.
    """
    target = _find_embedding_module(pipeline.model)
    if target is None:
        log.error("Could not find embedding layer to hook")
        sys.exit(1)

    captured, handle = _install_input_hook(target)

    # Monkey-patch model.forward to capture ref_s (the actual style vector used).
    original_forward = pipeline.model.forward
    captured_ref_s = {}

    def patched_forward(phonemes, ref_s, speed_arg=1, **kwargs):
        captured_ref_s["ref_s"] = ref_s.detach().cpu().clone()
        return original_forward(phonemes, ref_s, speed_arg, **kwargs)

    pipeline.model.forward = patched_forward

    log.info("Generating audio for: '%s' at speed=%s", text, speed)
    generator = pipeline(text, voice=voice, speed=speed)

    # Capture first segment only.
    audio = None
    for _gs, _ps, audio_segment in generator:
        if audio_segment is not None:
            audio = audio_segment
        break

    handle.remove()
    pipeline.model.forward = original_forward

    if audio is None:
        log.error("No audio generated")
        sys.exit(1)

    input_ids = captured.get("input_ids")
    if input_ids is None:
        log.error("Failed to capture input_ids via forward hook")
        sys.exit(1)

    style_tensor = captured_ref_s.get("ref_s")
    if style_tensor is None:
        log.error("Failed to capture ref_s via model forward hook")
        sys.exit(1)

    input_ids = _ensure_2d(input_ids)
    style_tensor = _ensure_2d(style_tensor)
    audio = _ensure_3d(audio)

    log.info("Captured input_ids: %s (range: %d-%d)",
             list(input_ids.shape), input_ids.min().item(), input_ids.max().item())
    log.info("Captured style: %s", list(style_tensor.shape))
    log.info("Audio shape: %s, duration: %.2fs",
             list(audio.shape), audio.shape[-1] / 24000)

    return input_ids, style_tensor, audio


def _capture_decoder_blocks(model, input_ids, ref_s, speed, device, f0_pred, n_pred,
                            intermediates, save_fn):
    """Capture per-block decoder intermediates for layerwise parity (#2681).

    Stage 1: FullDecoder encode + decode blocks.
    Stage 2: Generator conv_pre + upsample stages + output.
    """
    import torch.nn.functional as F
    decoder = model.decoder
    s_decoder = ref_s[:, :128].to(device)
    asr = intermediates["inter_regulated"].to(device)

    # Capture har_source via stft.transform hook and magnitude/phase via
    # stft.inverse hook. This captures the ACTUAL tensors computed inside
    # Generator.forward, avoiding the double-computation bug (#2691).
    captured_hooks = {}
    gen = decoder.generator

    # Capture SourceModule output (before STFT) for D4 source signal parity.
    # m_source.forward(f0) returns (har_source, noise, uv), all [B, T_audio, 1].
    # We capture har_source (first element) — the tanh-bounded excitation signal.
    def _m_source_hook(module, inputs, output):
        # output is tuple: (sine_merge [B,T,1], noise [B,T,1], uv [B,T,1])
        captured_hooks["source_module_output"] = output[0].detach().cpu()
    m_source_handle = gen.m_source.register_forward_hook(_m_source_hook)

    original_stft_transform = gen.stft.transform
    def _stft_transform_hook(input_signal):
        result = original_stft_transform(input_signal)
        har_spec, har_phase = result
        har = torch.cat([har_spec, har_phase], dim=1)
        captured_hooks["har_source"] = har.detach().cpu()
        return result
    gen.stft.transform = _stft_transform_hook

    original_stft_inverse = gen.stft.inverse
    def _stft_inverse_hook(magnitude, phase):
        captured_hooks["magnitude"] = magnitude.detach().cpu()
        captured_hooks["phase"] = phase.detach().cpu()
        return original_stft_inverse(magnitude, phase)
    gen.stft.inverse = _stft_inverse_hook

    # Use forward_with_tokens (not model()) because input_ids is a tensor,
    # and model.forward() expects a phoneme string (#2691).
    with torch.no_grad():
        model.forward_with_tokens(input_ids.to(device), ref_s.to(device), speed)

    gen.stft.transform = original_stft_transform
    gen.stft.inverse = original_stft_inverse
    m_source_handle.remove()

    # Save source_module_output (D4: source signal parity, #2691).
    if "source_module_output" in captured_hooks:
        save_fn("inter_source_module_output", captured_hooks["source_module_output"])

    # Save har_source and magnitude/phase from hooked forward BEFORE replay.
    # The replay below may fail (dimension mismatch), so save eagerly.
    if "har_source" in captured_hooks:
        save_fn("inter_har_source", captured_hooks["har_source"])
    if "magnitude" in captured_hooks:
        save_fn("inter_decoder_magnitude", captured_hooks["magnitude"])
        save_fn("inter_decoder_phase", captured_hooks["phase"])

    # FullDecoder Stage 1: encode + decode blocks.
    # F0Ntrain returns [B, T] (2D); F0_conv/N_conv are Conv1d expecting [B, 1, T] (3D).
    f0_3d = f0_pred.unsqueeze(1) if f0_pred.dim() == 2 else f0_pred
    n_3d = n_pred.unsqueeze(1) if n_pred.dim() == 2 else n_pred
    f0_down = decoder.F0_conv(f0_3d)
    n_down = decoder.N_conv(n_3d)
    x = decoder.encode(torch.cat([asr, f0_down, n_down], dim=1), s_decoder)
    save_fn("inter_decoder_encode", x)

    asr_res = decoder.asr_res(asr)
    num_decode = len(decoder.decode) if hasattr(decoder.decode, '__len__') else 4
    for i in range(min(num_decode, 4)):
        skip = torch.cat([x, asr_res, f0_down, n_down], dim=1)
        x = decoder.decode[i](skip, s_decoder)
        save_fn(f"inter_decoder_decode_{i}", x)

    # Generator Stage 2: conv_pre → upsample stages → output.
    _capture_generator_stages(gen, x, s_decoder, captured_hooks, device, save_fn)


def _capture_generator_stages(gen, decoder_out, s_decoder, captured_hooks, device, save_fn):
    """Capture Generator per-upsample-block intermediates (#2681)."""
    import torch.nn.functional as F
    har_source = captured_hooks.get("har_source")
    if har_source is None:
        log.warning("har_source not captured; Generator per-stage skipped")
        return
    har_source_dev = har_source.to(device)
    save_fn("inter_har_source", har_source_dev)
    h = gen.conv_pre(decoder_out)
    save_fn("inter_gen_conv_pre", h)
    num_ups = len(gen.ups)
    for stage_idx in range(num_ups):
        h = F.leaky_relu(h, 0.1)
        h = gen.ups[stage_idx](h)
        noise = gen.noise_convs[stage_idx](har_source_dev)
        noise_out = gen.noise_res[stage_idx](noise, s_decoder)
        t_h, t_n = h.shape[-1], noise_out.shape[-1]
        if t_n > t_h:
            noise_out = noise_out[..., :t_h]
        elif t_n < t_h:
            noise_out = F.pad(noise_out, (0, t_h - t_n))
        h = h + noise_out
        rb_per = len(gen.resblocks) // num_ups
        rb_start = stage_idx * rb_per
        xs = sum(gen.resblocks[rb_start + j](h, s_decoder) for j in range(rb_per))
        h = xs / rb_per
        save_fn(f"inter_gen_upsample_{stage_idx}", h)
    # Output stage: conv_post → split magnitude/phase.
    h = F.leaky_relu(h, 0.01)
    out = gen.conv_post(h)
    n_bins = out.shape[1] // 2
    log_mag = out[:, :n_bins, :]
    phase_raw = out[:, n_bins:, :]
    save_fn("inter_log_magnitude", log_mag)
    save_fn("inter_phase_raw", phase_raw)
    save_fn("inter_magnitude", torch.exp(log_mag.clamp(-88.0, 88.0)))
    save_fn("inter_phase", torch.sin(phase_raw))


def _capture_intermediates(model, input_ids, ref_s, speed):
    """Replay model forward, capturing inter_* tensors for layerwise parity.

    Names match test_kokoro_l3_layerwise_parity expectations in kokoro_l3_parity.rs.
    """
    intermediates = {}
    device = next(model.parameters()).device

    def _save(name, t):
        intermediates[name] = t.detach().cpu().float()
        log.info("  %s: %s", name, list(t.shape))

    # PlBert (ALBERT encoder)
    mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    bert_output = model.bert(input_ids, attention_mask=mask)
    if isinstance(bert_output, (tuple, list)):
        bert_output = bert_output[0]
    _save("inter_bert_output", bert_output)

    # bert_encoder (Linear 768->512) + transpose to [B, 512, T]
    bert_dur = model.bert_encoder(bert_output).transpose(-1, -2)
    _save("inter_bert_features", bert_dur)

    # TextEncoder (Embedding + Conv + BiLSTM + projection)
    input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(device)
    text_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
    text_features = model.text_encoder(input_ids, input_lengths, text_mask)
    _save("inter_text_features", text_features)

    # Split style: decoder=[0:128], prosody=[128:256]
    ref_s = ref_s.to(device)
    s_content = ref_s[:, 128:]

    # DurationEncoder → BiLSTM → duration_proj (d_model+style=640-dim)
    d = model.predictor.text_encoder(bert_dur, s_content, input_lengths, text_mask)
    x = model.predictor.lstm(d)
    if isinstance(x, tuple):
        x = x[0]
    dur_logits = model.predictor.duration_proj(x)
    # Rust ProsodyPredictor returns features as [B, d_model+style_dim, T] (channel-first)
    _save("inter_prosody_features", d.transpose(-1, -2))
    _save("inter_dur_logits", dur_logits)

    # Duration: sigmoid → sum → /speed → clamp (matches kokoro_tts.rs:256-265)
    max_dur = 50.0
    durations = dur_logits.sigmoid().sum(-1) / speed
    durations = durations.clamp(min=1.0, max=max_dur)
    _save("inter_durations_raw", durations)

    # length_regulate via alignment matrix (matches PyTorch exactly)
    pred_dur = durations.round().clamp(min=1).long()
    t_mel = pred_dur.sum().item()
    pred_aln = torch.zeros(input_ids.shape[-1], t_mel, device=device)
    c_frame = 0
    for i in range(pred_dur.shape[-1]):
        count = pred_dur[0, i].item()
        pred_aln[i, c_frame:c_frame + count] = 1.0
        c_frame += count
    # d is [B, T, 640]; transpose to [B, 640, T] for matmul with [T, T_mel]
    aligned_dur = d.transpose(-1, -2) @ pred_aln  # [B, 640, T_mel]
    _save("inter_aligned_dur", aligned_dur)
    _save("inter_regulated", text_features @ pred_aln)

    # F0 + Energy prediction (aligned_dur already includes style).
    f0_pred, n_pred = model.predictor.F0Ntrain(aligned_dur, s_content)
    _save("inter_f0", f0_pred)
    _save("inter_energy", n_pred)

    # Capture Generator's pre-iSTFT magnitude/phase by hooking stft.inverse.
    gen = model.decoder.generator
    original_stft_inverse = gen.stft.inverse
    def _stft_inverse_hook(magnitude, phase):
        _save("inter_gen_magnitude", magnitude.clone())
        _save("inter_gen_phase", phase.clone())
        return original_stft_inverse(magnitude, phase)
    gen.stft.inverse = _stft_inverse_hook

    # Decoder per-block intermediates (#2681).
    try:
        _capture_decoder_blocks(
            model, input_ids, ref_s, speed, device, f0_pred, n_pred, intermediates, _save,
        )
    except (RuntimeError, AttributeError) as e:
        log.warning("Decoder block replay failed (non-fatal): %s", e)

    gen.stft.inverse = original_stft_inverse

    log.info("Captured %d intermediate tensors", len(intermediates))
    return intermediates


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        log.error("safetensors package not installed. pip install safetensors")
        sys.exit(1)

    try:
        from kokoro import KPipeline
    except ImportError:
        log.error("kokoro package not installed. pip install kokoro")
        sys.exit(1)

    log.info("Loading Kokoro pipeline (using package default weights)...")
    pipeline = KPipeline(lang_code="a")

    # Zero out SourceModule noise to match nn deterministic inference (#2691).
    # nn uses DynTensor::zeros for noise (no Gaussian). Patching randn_like to
    # return zeros during SineGen makes noise_amp * randn = 0.
    if args.deterministic:
        gen = pipeline.model.decoder.generator
        if hasattr(gen, 'm_source') and hasattr(gen.m_source, 'l_sin_gen'):
            sine_gen = gen.m_source.l_sin_gen
            original_sinegen_forward = sine_gen.forward

            def _zero_noise_forward(f0):
                # Patch both randn_like (Gaussian noise) and rand (initial
                # phase noise for harmonics 2-9) to make SineGen deterministic.
                # nn's SineGen uses zero initial phase for all harmonics.
                original_randn = torch.randn_like
                original_rand = torch.rand
                torch.randn_like = lambda x, **kw: torch.zeros_like(x)
                torch.rand = lambda *a, **kw: torch.zeros(*a, **kw)
                try:
                    result = original_sinegen_forward(f0)
                finally:
                    torch.randn_like = original_randn
                    torch.rand = original_rand
                return result

            sine_gen.forward = _zero_noise_forward
            log.info("Deterministic mode: zeroed SineGen noise (randn_like → zeros)")
        else:
            log.warning("Could not find SineGen (l_sin_gen) for deterministic mode")

    # Save model weights if requested (with weight_norm decomposed + key remap).
    if args.weights_out:
        raw_sd = {k: v.cpu() for k, v in pipeline.model.state_dict().items()}
        decomposed = decompose_weight_norm(raw_sd)
        remapped = remap_keys_for_rust(decomposed)
        weights_path = Path(args.weights_out)
        save_file(remapped, str(weights_path))
        log.info("Saved model weights to %s (%d tensors, %d bytes)",
                 weights_path, len(remapped), weights_path.stat().st_size)

    input_ids, style, audio = _capture_model_inputs(
        pipeline, args.text, args.voice, args.speed,
    )

    tensors = {
        "input_ids": input_ids.float().contiguous(),
        "style": style.float().contiguous(),
        "speed": torch.tensor([args.speed], dtype=torch.float32),
        "audio": audio.float().contiguous(),
    }

    # Capture intermediate tensors for layerwise parity testing.
    if args.intermediates:
        log.info("Capturing intermediate tensors for layerwise parity...")
        intermediates = _capture_intermediates(
            pipeline.model, input_ids.long().to(next(pipeline.model.parameters()).device),
            style, args.speed,
        )
        for name, t in intermediates.items():
            tensors[name] = t.float().contiguous()

    output_path = Path(args.output)
    save_file(tensors, str(output_path))
    log.info("Saved reference to %s (%d bytes)", output_path, output_path.stat().st_size)
    for name, t in tensors.items():
        log.info("  %s: %s", name, list(t.shape))


if __name__ == "__main__":
    main()
