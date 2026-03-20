#!/usr/bin/env python3
"""Subtract known backing tracks from audience recordings via spectral subtraction."""

import os
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, correlate, resample

FFMPEG = r"C:\Program Files\MediaHuman\Audio Converter\ffmpeg.exe"
BIN = "bin"
OUT = "bin/results"
SR = 44100

MOV1 = os.path.join(BIN, "MVI_8876.MP4")
MOV2 = os.path.join(BIN, "MVI_8877.MP4")
BACK1 = os.path.join(BIN, "iohtba_film_part1_computer_sound.aif")
BACK2 = os.path.join(BIN, "iohtba_film_part2_computer_sound.aif")

# Subtraction parameters - tuned after first run
ALPHA = 0.7          # was 1.0 - less aggressive to reduce spectral artifacts
FLOOR = 0.08         # was 0.01 - higher floor prevents musical noise / muffling
BLOCK_S = 15.0       # was 30.0 - shorter blocks adapt better to level changes

# Alignment regions: use sections where backing track dominates (no actress)
# These are (start_s, end_s) in each file's local time
ALIGN_REGIONS_PART1 = [
    (0, 33),       # hold music, loud and clear
    (40, 50),      # background sounds, hold music low volume
    (820, 828),    # 13:40-13:48 just narrator
]
ALIGN_REGIONS_PART2 = [
    (341, 352),    # 05:41-05:52 narrator + piano, no actress voice
    (926, 932),    # 15:26-15:32 just narrator
    (1325, 1365),  # 22:05-22:45 narrator then silence, no actress
]

os.makedirs(OUT, exist_ok=True)


def load_mono(path, target_sr=SR):
    data, sr = sf.read(path, dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = resample(data, int(len(data) * target_sr / sr))
    return data


def extract_audio(video_path, wav_path):
    if os.path.exists(wav_path):
        print(f"  {wav_path} already exists, skipping extraction")
        return
    print(f"  Extracting {video_path} -> {wav_path}")
    subprocess.run([
        FFMPEG, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
        wav_path,
    ], capture_output=True, check=True)


def find_delay_on_regions(mixture, reference, regions, max_delay_s=2.0):
    """Align using specific regions where backing track dominates."""
    max_samples = int(max_delay_s * SR)
    best_offset = 0
    best_confidence = 0

    for start_s, end_s in regions:
        s, e = int(start_s * SR), int(end_s * SR)
        if s >= len(mixture) or s >= len(reference):
            continue
        e = min(e, len(mixture), len(reference))
        mix_chunk = mixture[s:e]
        ref_chunk = reference[s:e]
        if len(mix_chunk) < SR:
            continue

        corr = correlate(mix_chunk, ref_chunk, mode="full")
        center = len(ref_chunk) - 1
        lo = max(0, center - max_samples)
        hi = min(len(corr), center + max_samples + 1)
        search = corr[lo:hi]
        offset = np.argmax(np.abs(search)) - (center - lo)
        confidence = np.abs(search).max() / (np.abs(search).mean() + 1e-10)

        print(f"    Region {start_s:.0f}-{end_s:.0f}s: delay={offset} "
              f"({offset/SR*1000:.1f}ms), confidence={confidence:.1f}")

        if confidence > best_confidence:
            best_confidence = confidence
            best_offset = offset

    return best_offset, best_confidence


def spectral_subtract(mixture, reference, alpha=ALPHA, floor=FLOOR,
                      block_s=BLOCK_S, nperseg=2048):
    noverlap = nperseg * 3 // 4
    min_len = min(len(mixture), len(reference))
    mixture, reference = mixture[:min_len], reference[:min_len]

    _, _, Zmix = stft(mixture, fs=SR, nperseg=nperseg, noverlap=noverlap)
    _, _, Zref = stft(reference, fs=SR, nperseg=nperseg, noverlap=noverlap)

    mag_mix = np.abs(Zmix)
    mag_ref = np.abs(Zref)
    phase_mix = np.angle(Zmix)

    block_frames = max(1, int(block_s * SR / (nperseg - noverlap)))
    n_frames = mag_mix.shape[1]
    gain = np.ones_like(mag_ref)

    for bs in range(0, n_frames, block_frames):
        be = min(bs + block_frames, n_frames)
        ref_e = np.sum(mag_ref[:, bs:be] ** 2, axis=1) + 1e-10
        mix_e = np.sum(mag_mix[:, bs:be] ** 2, axis=1) + 1e-10
        g = np.clip(np.sqrt(mix_e / ref_e), 0.0, 3.0)
        gain[:, bs:be] = g[:, np.newaxis]

    mag_result = np.maximum(mag_mix - alpha * gain * mag_ref, floor * mag_mix)
    _, result = istft(mag_result * np.exp(1j * phase_mix), fs=SR, nperseg=nperseg, noverlap=noverlap)
    return result


def process_pair(video_path, backing_path, label, align_regions):
    print(f"\n=== {label} ===")

    wav_path = os.path.join(OUT, f"{label}_mixture.wav")
    extract_audio(video_path, wav_path)

    print(f"  Loading mixture...")
    mixture = load_mono(wav_path)
    print(f"    {len(mixture)/SR:.1f}s")

    print(f"  Loading backing track...")
    backing = load_mono(backing_path)
    print(f"    {len(backing)/SR:.1f}s")

    print(f"  Finding alignment on backing-dominant regions...")
    delay, confidence = find_delay_on_regions(mixture, backing, align_regions)
    print(f"  Best: delay={delay} ({delay/SR*1000:.1f}ms), confidence={confidence:.1f}")

    if delay > 0:
        backing = backing[delay:]
    elif delay < 0:
        mixture = mixture[-delay:]

    print(f"  Spectral subtraction (alpha={ALPHA}, floor={FLOOR}, block={BLOCK_S}s)...")
    result = spectral_subtract(mixture, backing)

    peak = np.abs(result).max()
    if peak > 0.95:
        result = result * (0.95 / peak)
        print(f"    Normalized peak {peak:.3f} -> 0.95")

    out_path = os.path.join(OUT, f"{label}_subtracted.wav")
    sf.write(out_path, result, SR)
    print(f"  -> {out_path} ({len(result)/SR:.1f}s)")

    return wav_path, out_path


print("Spectral subtraction: backing track removal (v2)")
print(f"  alpha={ALPHA}, floor={FLOOR}, block={BLOCK_S}s")
print("=" * 50)

mix1_path, sub1_path = process_pair(MOV1, BACK1, "part1", ALIGN_REGIONS_PART1)
mix2_path, sub2_path = process_pair(MOV2, BACK2, "part2", ALIGN_REGIONS_PART2)

print(f"\n=== Concatenating ===")
sub1 = load_mono(sub1_path)
sub2 = load_mono(sub2_path)
concat = np.concatenate([sub1, sub2])
concat_path = os.path.join(OUT, "full_subtracted.wav")
sf.write(concat_path, concat, SR)
print(f"  -> {concat_path} ({len(concat)/SR:.1f}s / {len(concat)/SR/60:.1f}m)")

mix1 = load_mono(mix1_path)
mix2 = load_mono(mix2_path)
concat_mix = np.concatenate([mix1, mix2])
concat_mix_path = os.path.join(OUT, "full_mixture.wav")
sf.write(concat_mix_path, concat_mix, SR)
print(f"  -> {concat_mix_path} ({len(concat_mix)/SR:.1f}s / {len(concat_mix)/SR/60:.1f}m)")

print(f"\nDone. Results in {OUT}/")
