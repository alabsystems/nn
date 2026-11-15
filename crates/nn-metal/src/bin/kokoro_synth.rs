// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Synthesize speech from Kokoro TTS and write a WAV file.
//!
//! Usage:
//!   KOKORO_WEIGHTS=/path/to/kokoro_v1_0.safetensors \
//!   targo --unverified run -p nn-metal --bin kokoro_synth -- [--voice af_heart] [--speed 1.0] [--out output.wav]

use std::io::IsTerminal;
use std::path::PathBuf;

use nn_metal::{register_metal_dyn_backend, MetalBackend, PipelineCache};

fn main() {
    let weights_path = std::env::var("KOKORO_WEIGHTS")
        .expect("Set KOKORO_WEIGHTS=/path/to/kokoro_v1_0.safetensors");

    // Parse CLI args.
    let args: Vec<String> = std::env::args().collect();
    let voice_name = get_arg(&args, "--voice").unwrap_or_else(|| "af_heart".into());
    let speed: f32 = get_arg(&args, "--speed")
        .map(|s| s.parse().expect("--speed must be a number"))
        .unwrap_or(1.0);
    let out_path = get_arg(&args, "--out").unwrap_or_else(|| "kokoro_output.wav".into());

    eprintln!("Loading Kokoro weights from: {weights_path}");

    // Token IDs: from --ids arg (comma-separated) or stdin.
    let phoneme_ids: Vec<i64> = if let Some(ids_str) = get_arg(&args, "--ids") {
        ids_str
            .split(',')
            .map(|s| s.trim().parse::<i64>().expect("invalid token ID"))
            .collect()
    } else {
        // Read from stdin (piped from kokoro_phonemize.py).
        let mut line = String::new();
        if std::io::stdin().is_terminal() {
            eprintln!("No --ids provided and stdin is a terminal.");
            eprintln!("Usage: python3 scripts/kokoro_phonemize.py \"Hello\" | targo --unverified run -p nn-metal --bin kokoro_synth");
            eprintln!("   or: targo --unverified run -p nn-metal --bin kokoro_synth -- --ids 0,50,83,54,57,156,65,86,123,54,46,0");
            std::process::exit(1);
        }
        std::io::stdin().read_line(&mut line).expect("read stdin");
        line.trim()
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<i64>()
                    .expect("invalid token ID from stdin")
            })
            .collect()
    };

    let _ = MetalBackend::init().expect("Metal GPU required");
    register_metal_dyn_backend();
    let cache = PipelineCache::new_global().expect("Metal pipeline cache");

    // Load model.
    // SAFETY: safetensors file is not modified while alive.
    let mut kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load(&weights_path)
            .expect("Failed to load Kokoro model")
    };

    // Create input tensor.
    let seq_len = phoneme_ids.len();

    // Load voice style embedding (selected by seq_len from voice pack).
    let voice_path = find_voice_file(&weights_path, &voice_name);
    eprintln!("Loading voice: {voice_name} from {}", voice_path.display());
    let style = load_voice_style(&voice_path, seq_len);
    eprintln!("Style shape: {:?}", style.dims());
    let input_ids = nn_core::dyn_tensor::DynTensor::from_vec_i64(
        phoneme_ids,
        &[1, seq_len],
        &nn_core::Device::Cpu,
    )
    .expect("input_ids creation");

    eprintln!("Synthesizing (speed={speed}, seq_len={seq_len})...");
    let start = std::time::Instant::now();
    let (audio, cert) = kokoro
        .synthesize(&input_ids, &style, speed, &cache)
        .expect("Synthesis failed");
    let elapsed = start.elapsed();

    let pcm = audio.to_flat_vec::<f32>().expect("PCM extraction");
    let n_samples = pcm.len();
    let duration_sec = n_samples as f64 / 24000.0;
    let rtf = elapsed.as_secs_f64() / duration_sec;

    eprintln!("Done in {elapsed:.2?}");
    eprintln!("  Audio: {n_samples} samples ({duration_sec:.2}s at 24kHz)");
    eprintln!("  RTF: {rtf:.3}");
    eprintln!("  Certificate: passed={}", cert.overall_passed);

    // Write WAV.
    write_wav(&out_path, &pcm, 24000);
    eprintln!("Wrote: {out_path}");

    // Print dispatch summary.
    let summary = kokoro.dispatch_summary();
    eprintln!(
        "  Dispatches: total={}, metal={} [plbert={}, text={}, prosody={}, f0={}, gen={}]",
        kokoro.total_dispatches(),
        kokoro.total_metal_dispatches(),
        summary.plbert,
        summary.text_encoder,
        summary.prosody,
        summary.f0_energy,
        summary.generator,
    );
}

/// The argument after the first occurrence of `flag`, if any.
fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().skip_while(|a| *a != flag).nth(1).cloned()
}

fn find_voice_file(weights_path: &str, voice_name: &str) -> PathBuf {
    let weights_dir = PathBuf::from(weights_path).parent().unwrap().to_path_buf();

    // Try: weights_dir/voices/<name>.safetensors
    let candidate = weights_dir
        .join("voices")
        .join(format!("{voice_name}.safetensors"));
    if candidate.exists() {
        return candidate;
    }

    // Try: models/kokoro-82m/voices/<name>.safetensors
    let models_dir = weights_dir
        .parent()
        .unwrap()
        .join("models")
        .join("kokoro-82m")
        .join("voices");
    let candidate = models_dir.join(format!("{voice_name}.safetensors"));
    if candidate.exists() {
        return candidate;
    }

    // Try .pt as fallback
    let candidate = weights_dir.join("voices").join(format!("{voice_name}.pt"));
    assert!(
        !candidate.exists(),
        "Found {voice_name}.pt but not .safetensors. Convert with: \
         python3 -c \"import torch; t=torch.load('{}.pt'); \
         from safetensors.torch import save_file; save_file({{'style': t}}, '{}.safetensors')\"",
        candidate.display(),
        candidate.with_extension("safetensors").display(),
    );

    panic!(
        "Voice file not found for '{voice_name}'. Searched:\n  {}\n  {}",
        weights_dir.join("voices").display(),
        models_dir.display(),
    );
}

fn load_voice_style(path: &PathBuf, seq_len: usize) -> nn_core::dyn_tensor::DynTensor {
    use nn_core::dyn_tensor::DynTensor;
    let tensors = nn_core::dyn_tensor::load_safetensors(path).expect("load voice safetensors");

    // Voice packs store embeddings as [N_segments, 1, style_dim] under "embedding".
    // Select the row matching seq_len (clamped to valid range), flatten to [1, style_dim].
    if let Some(emb) = tensors.get("embedding") {
        let dims = emb.dims();
        // Shape: [N, 1, style_dim] — select row min(seq_len, N-1).
        let n = dims[0];
        let style_dim = dims[dims.len() - 1];
        let idx = seq_len.min(n - 1);
        let flat = emb.to_flat_vec::<f32>().unwrap();
        let row_start = idx * style_dim;
        let row = &flat[row_start..row_start + style_dim];
        eprintln!("  Voice pack: {n} entries, style_dim={style_dim}, selected idx={idx}");
        return DynTensor::from_slice(row, &[1, style_dim], &nn_core::Device::Cpu).unwrap();
    }

    // Fallback: "style" key or single tensor.
    if let Some(style) = tensors.get("style") {
        return style
            .clone()
            .reshape([1, style.dims().iter().product::<usize>()])
            .unwrap();
    }
    if tensors.len() == 1 {
        let style = tensors.values().next().unwrap();
        return style
            .clone()
            .reshape([1, style.dims().iter().product::<usize>()])
            .unwrap();
    }
    panic!(
        "Voice file has {} tensors, no 'embedding' or 'style' key. Keys: {:?}",
        tensors.len(),
        tensors.keys().collect::<Vec<_>>()
    );
}

/// RIFF chunk size, data chunk size and byte rate of a 16-bit mono PCM WAV
/// file holding `num_samples` samples at `sample_rate` Hz.
///
/// RIFF sizes are 32-bit, so `None` when the audio does not fit one WAV
/// file (or the byte rate overflows) instead of a truncated or wrapped header.
fn wav_sizes(num_samples: usize, sample_rate: u32) -> Option<(u32, u32, u32)> {
    let data_bytes = u32::try_from(num_samples).ok()?.checked_mul(2)?; // 16-bit PCM
    let file_size = data_bytes.checked_add(36)?;
    let byte_rate = sample_rate.checked_mul(2)?;
    Some((file_size, data_bytes, byte_rate))
}

fn write_wav(path: &str, pcm: &[f32], sample_rate: u32) {
    use std::io::Write;

    let Some((file_size, data_bytes, byte_rate)) = wav_sizes(pcm.len(), sample_rate) else {
        panic!(
            "{} samples at {sample_rate} Hz do not fit a WAV file (32-bit RIFF sizes)",
            pcm.len()
        );
    };

    // Streamed through a buffered writer: the bytes are those of the whole
    // file assembled in memory, without an allocation sized by the audio.
    let file = std::fs::File::create(path).expect("Failed to write WAV file");
    let mut w = std::io::BufWriter::new(file);
    let mut put = |bytes: &[u8]| w.write_all(bytes).expect("Failed to write WAV file");
    // RIFF header
    put(b"RIFF");
    put(&file_size.to_le_bytes());
    put(b"WAVE");
    // fmt chunk
    put(b"fmt ");
    put(&16u32.to_le_bytes()); // chunk size
    put(&1u16.to_le_bytes()); // PCM format
    put(&1u16.to_le_bytes()); // mono
    put(&sample_rate.to_le_bytes());
    put(&byte_rate.to_le_bytes()); // byte rate
    put(&2u16.to_le_bytes()); // block align
    put(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    put(b"data");
    put(&data_bytes.to_le_bytes());
    for &s in pcm {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        put(&i16_val.to_le_bytes());
    }
    w.flush().expect("Failed to write WAV file");
}

#[cfg(test)]
mod tests {
    use super::{get_arg, wav_sizes};

    #[test]
    fn get_arg_returns_value_after_first_flag() {
        let args: Vec<String> = ["bin", "--voice", "a", "--voice", "b", "--out"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        assert_eq!(get_arg(&args, "--voice").as_deref(), Some("a"));
        assert_eq!(get_arg(&args, "--out"), None);
        assert_eq!(get_arg(&args, "--speed"), None);
    }

    #[test]
    fn wav_sizes_match_canonical_header() {
        assert_eq!(wav_sizes(3, 24_000), Some((42, 6, 48_000)));
        assert_eq!(wav_sizes(0, 24_000), Some((36, 0, 48_000)));
    }

    /// Regression: the sample count was cast with `as u32`, so 2^32 samples
    /// wrote a header claiming 0 data bytes, and 2^31 samples overflowed
    /// `n * 2`. Both are now rejected.
    #[test]
    fn wav_sizes_reject_audio_too_long_for_riff() {
        assert_eq!(wav_sizes(1 << 32, 24_000), None);
        assert_eq!(wav_sizes(1 << 31, 24_000), None);
        let max = (u32::MAX as usize - 36) / 2;
        assert!(wav_sizes(max, 24_000).is_some());
        assert_eq!(wav_sizes(max + 1, 24_000), None);
        assert_eq!(wav_sizes(1, u32::MAX), None);
    }
}
