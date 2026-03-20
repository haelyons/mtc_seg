#!/usr/bin/env python3
"""
Voice separation for "Instructions On How To Be Alone"
=======================================================
Isolates the actress's voice from two audience-perspective recordings of a
one-woman theatrical performance using Meta's SAM Audio with text + span
prompting.

The recording contains heavily overlaid sources: actress, narrator (female,
played from PA via MIDI), background music, sine tones, ambience, piano,
and audience reactions. The actress and narrator are both female, so text
prompting alone ("a woman speaking") won't disambiguate. Span anchors are
the critical differentiator:
  (+) marks time ranges where the ACTRESS is clearly speaking
  (-) marks time ranges where she is NOT speaking (narrator-only, music,
      audience, silence)

The two input audio files (RX spectral denoised exports of mov1/8876 and
mov2/8877) are concatenated into a single file before processing. All
timestamps in ANCHORS use absolute seconds in the concatenated timeline.

Usage:
  1. Place both denoised .wav files in bin/
  2. Run: python separate_voice.py
  3. Outputs land in ./output/

Hardware: Tested targeting A100 80GB. The large model + reranking will use
significant VRAM. If OOM, reduce reranking_candidates or use sam-audio-base.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

# ============================================================================
# CONFIG — Edit this section before running
# ============================================================================

# --- INPUT FILES ---
# Two sequential wav files (RX spectral denoised exports from audience recording).
# They will be concatenated into a single audio file for processing.
PART1_FILE = "bin/MVI_8876_spectral_denoise.wav"  # first ~30 min
PART2_FILE = "bin/MVI_8877_spectral_denoise.wav"  # second ~27 min

# If you already have a concatenated file, set this and SKIP_CONCAT=True
CONCAT_FILE = "output/full_denoised.wav"
SKIP_CONCAT = False

# Model choice: "facebook/sam-audio-large" or "facebook/sam-audio-base"
MODEL_ID = "facebook/sam-audio-large"

# Quality vs speed. 8 = best quality, generates 8 candidates and picks best.
RERANKING_CANDIDATES = 8

# Enable span prediction — critical for speech (event-based, not ambient)
PREDICT_SPANS = True

# Output directory
OUTPUT_DIR = Path("./output")

# --- TRIM POINTS (absolute seconds in concatenated timeline) ---
# Trim start: after hold music + audience settling at the top of mov1
# Trim end: before applause/hold music at the end of mov2
# PART1_DURATION_S is detected automatically via ffprobe at runtime.
TRIM_START_S = 40.0       # mov1 00:40 — hold music fades, performance begins
# TRIM_END_S is set dynamically: PART1_DURATION_S + mov2 24:30 (before clapping)
TRIM_END_OFFSET_MOV2 = 24 * 60 + 30  # mov2 timestamp in seconds: 24:30

# --- SPAN ANCHORS ---
# Format: ("+"/"-", start_seconds, end_seconds)
# All times are ABSOLUTE in the concatenated timeline (not relative to trim).
# The script will convert them to trim-relative at runtime.
#
# Positive (+): actress clearly speaking (even with background noise — that's ok,
#   the model uses these as temporal markers, not acoustic templates)
# Negative (-): actress NOT speaking (narrator-only, music-only, silence, audience)
#
# MOV2_OFFSET will be added automatically at runtime. For readability, mov2
# timestamps are written as expressions below.

ANCHORS_MOV1 = [
    # --- POSITIVE: actress clearly speaking ---
    ("+",   84.0,   91.0),  # 01:24-01:31 actress speaking clearly, bg music
    ("+",  102.0,  113.0),  # 01:42-01:53 speaking clearly, some bg music
    ("+",  119.0,  124.0),  # 01:59-02:04 actress calling out
    ("+",  154.0,  161.0),  # 02:34-02:41 actress talking clearly
    ("+",  300.0,  302.0),  # 05:00-05:02 clear voice, bg music barely audible
    ("+",  353.0,  360.0),  # 05:53-06:00 clear voice, music increasing
    ("+",  450.0,  465.0),  # 07:30-07:45 actress speaking loudly
    ("+",  635.0,  638.0),  # 10:35-10:38 clear phrase, NO background music
    ("+",  644.0,  647.0),  # 10:44-10:47 clear phrase, NO background music
    ("+",  995.0, 1005.0),  # 16:35-16:45 very clear speech (waves-on-beach bg)
    ("+", 1092.0, 1105.0),  # 18:12-18:25 clear, interview tone, very soft sine
    ("+", 1388.0, 1400.0),  # 23:08-23:20 no music, varied volumes

    # --- NEGATIVE: actress NOT speaking ---
    ("-",    0.0,   33.0),  # 00:00-00:33 hold music, no actress
    ("-",  820.0,  828.0),  # 13:40-13:48 just narrator
    ("-", 1422.0, 1428.0),  # 23:42-23:48 just background noise, no one speaking
]

# mov2 anchors — offsets computed at runtime from PART1_DURATION_S
# Written as mov2-local seconds for readability; converted in code.
ANCHORS_MOV2_LOCAL = [
    # --- POSITIVE ---
    ("+",  140.0,  149.0),  # 02:20-02:29 no bg music at all, halting speech
    ("+",  314.0,  322.0),  # 05:14-05:22 super clear, piano barely audible
    ("+",  742.0,  759.0),  # 12:22-12:39 very clear and loud
    ("+",  708.0,  727.0),  # 11:48-12:07 very clear over stretched piano

    # --- NEGATIVE ---
    ("-",  341.0,  352.0),  # 05:41-05:52 narrator + piano, actress only moving
    ("-",  926.0,  932.0),  # 15:26-15:32 just narrator over music
    ("-", 1325.0, 1335.0),  # 22:05-22:15 narrator, no music
    ("-", 1355.0, 1365.0),  # 22:35-22:45 just background noise, no music/actress
]


# ============================================================================
# STEP 1: Load and concatenate audio files
# ============================================================================

def concat_audio(part1: str, part2: str, audio_out: str) -> float:
    """Load two wav files, concatenate, save, return part1 duration in seconds."""
    print(f"[1/4] Concatenating audio...")
    print(f"    part1: {part1}")
    print(f"    part2: {part2}")

    w1, sr1 = torchaudio.load(part1)
    w2, sr2 = torchaudio.load(part2)

    # Resample part2 if sample rates differ
    if sr1 != sr2:
        print(f"    Resampling part2 from {sr2}Hz to {sr1}Hz")
        w2 = torchaudio.functional.resample(w2, sr2, sr1)

    # Convert to mono if needed
    if w1.shape[0] > 1:
        w1 = w1.mean(dim=0, keepdim=True)
    if w2.shape[0] > 1:
        w2 = w2.mean(dim=0, keepdim=True)

    part1_duration = w1.shape[1] / sr1
    print(f"    part1 duration: {part1_duration:.1f}s ({part1_duration/60:.1f}m)")
    print(f"    part2 duration: {w2.shape[1]/sr1:.1f}s ({w2.shape[1]/sr1/60:.1f}m)")

    concat = torch.cat([w1, w2], dim=1)
    torchaudio.save(audio_out, concat, sr1)

    total_duration = concat.shape[1] / sr1
    print(f"    -> {audio_out} ({total_duration:.1f}s / {total_duration/60:.1f}m)")
    return part1_duration


# ============================================================================
# STEP 2: Trim to relevant sections
# ============================================================================

def trim_audio(audio_path: str, start_s: float, end_s: float, out_path: str):
    """Trim audio to a time range using torchaudio."""
    waveform, sr = torchaudio.load(audio_path)
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    trimmed = waveform[:, start_sample:end_sample]
    torchaudio.save(out_path, trimmed, sr)
    duration = trimmed.shape[1] / sr
    print(f"    -> Trimmed to {duration:.1f}s, saved to {out_path}")
    return duration


# ============================================================================
# STEP 3: Run SAM Audio separation
# ============================================================================

def run_separation(
    audio_path: str,
    description: str,
    anchors: list,
    output_prefix: str,
    model,
    processor,
    device: torch.device,
    reranking_candidates: int = RERANKING_CANDIDATES,
):
    """
    Run SAM Audio separation with text + span prompting.

    The processor handles encoding the audio + prompt + anchors.
    The model's separate() handles multi-diffusion for long audio internally
    via overlapping windows with soft-mask merging.
    """
    print(f"    Description: '{description}'")
    print(f"    Anchors: {len(anchors)} ({sum(1 for a in anchors if a[0]=='+')}"
          f" positive, {sum(1 for a in anchors if a[0]=='-')} negative)")

    # Format anchors for the processor: list of ["+"/"-", start, end]
    formatted_anchors = [[sign, start, end] for sign, start, end in anchors]

    batch = processor(
        audios=[audio_path],
        descriptions=[description],
        anchors=[formatted_anchors] if formatted_anchors else None,
    ).to(device)

    print(f"    Running separation (reranking_candidates={reranking_candidates})...")
    t0 = time.time()

    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=PREDICT_SPANS,
            reranking_candidates=reranking_candidates,
        )

    elapsed = time.time() - t0
    print(f"    Separation took {elapsed:.1f}s ({elapsed/60:.1f}m)")

    sr = processor.audio_sampling_rate

    target_path = OUTPUT_DIR / f"{output_prefix}_voice.wav"
    residual_path = OUTPUT_DIR / f"{output_prefix}_residual.wav"

    # Result fields are lists of 1D tensors (one per batch item)
    target = result.target[0].cpu().unsqueeze(0)
    residual = result.residual[0].cpu().unsqueeze(0)

    torchaudio.save(str(target_path), target, sr)
    torchaudio.save(str(residual_path), residual, sr)

    print(f"    -> Voice:    {target_path}")
    print(f"    -> Residual: {residual_path}")

    return target_path, residual_path


# ============================================================================
# STEP 4: Quick quality check with SAM Audio Judge (optional)
# ============================================================================

def run_judge(audio_path: str, description: str):
    """
    Use SAM Audio Judge to score the separation quality.
    Returns scores for overall quality, recall, precision, faithfulness.
    This is optional but useful for comparing different prompt strategies.
    """
    try:
        from sam_audio import SAMAudioJudge, SAMAudioJudgeProcessor
        print(f"[optional] Running SAM Audio Judge on {audio_path}...")
        judge = SAMAudioJudge.from_pretrained("facebook/sam-audio-judge")
        judge_processor = SAMAudioJudgeProcessor.from_pretrained("facebook/sam-audio-judge")
        judge = judge.eval().cuda()

        inputs = judge_processor(
            audios=[str(audio_path)],
            descriptions=[description],
        ).to("cuda")

        with torch.inference_mode():
            scores = judge.score(inputs)

        print(f"    Quality: {scores}")
        return scores
    except Exception as e:
        print(f"    Judge not available or failed: {e}")
        print(f"    (This is optional — your ears are the real judge)")
        return None


# ============================================================================
# MAIN
# ============================================================================

def build_anchors(part1_duration_s: float, trim_start_s: float) -> list:
    """
    Combine mov1 and mov2 anchors into a single list with absolute times
    converted to trim-relative times.

    mov2 anchors have part1_duration_s added to get absolute time.
    Then all anchors are shifted by -trim_start_s to be relative to the
    trimmed audio (since SAM Audio sees 0.0 = start of trimmed section).
    """
    all_anchors = []

    # mov1 anchors are already in absolute seconds
    for sign, start, end in ANCHORS_MOV1:
        all_anchors.append((sign, start - trim_start_s, end - trim_start_s))

    # mov2 anchors need the offset added first
    for sign, start, end in ANCHORS_MOV2_LOCAL:
        abs_start = start + part1_duration_s
        abs_end = end + part1_duration_s
        all_anchors.append((sign, abs_start - trim_start_s, abs_end - trim_start_s))

    # Filter out any anchors that ended up with negative times (before trim)
    all_anchors = [(s, st, en) for s, st, en in all_anchors if en > 0]

    return all_anchors


def main():
    parser = argparse.ArgumentParser(description="SAM Audio voice separation")
    parser.add_argument("--test", action="store_true",
                        help="Process only first 5 minutes as a test run")
    parser.add_argument("--reranking", type=int, default=RERANKING_CANDIDATES,
                        help="Override reranking_candidates (1-8)")
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        help="Model ID (e.g. facebook/sam-audio-base for less VRAM)")
    args = parser.parse_args()

    reranking = args.reranking

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Device check ---
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. SAM Audio requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print()

    # --- Step 1: Concatenate audio ---
    if not SKIP_CONCAT:
        for f in [PART1_FILE, PART2_FILE]:
            if not os.path.exists(f):
                print(f"ERROR: Audio file not found: {f}")
                sys.exit(1)
        part1_duration = concat_audio(PART1_FILE, PART2_FILE, CONCAT_FILE)
    else:
        if not os.path.exists(CONCAT_FILE):
            print(f"ERROR: Concatenated file not found: {CONCAT_FILE}")
            sys.exit(1)
        print(f"[1/4] Using existing concatenated audio: {CONCAT_FILE}")
        # Still need part1 duration for anchor offsets
        w1, sr1 = torchaudio.load(PART1_FILE)
        part1_duration = w1.shape[1] / sr1
        del w1

    # --- Step 2: Build anchors and trim ---
    trim_end_s = part1_duration + TRIM_END_OFFSET_MOV2
    anchors = build_anchors(part1_duration, TRIM_START_S)

    print(f"\n[2/4] Trimming audio...")
    print(f"    Trim: {TRIM_START_S:.0f}s - {trim_end_s:.0f}s "
          f"({(trim_end_s - TRIM_START_S)/60:.1f}m)")
    print(f"    Anchors: {len(anchors)} "
          f"({sum(1 for a in anchors if a[0]=='+' )} positive, "
          f"{sum(1 for a in anchors if a[0]=='-')} negative)")

    trim_end = trim_end_s
    if args.test:
        trim_end = min(TRIM_START_S + 300.0, trim_end_s)
        # Also filter anchors to the test window
        test_len = trim_end - TRIM_START_S
        anchors = [(s, st, en) for s, st, en in anchors if st < test_len]
        print(f"    TEST MODE: processing first {test_len:.0f}s only "
              f"({len(anchors)} anchors in range)")

    trimmed_audio = str(OUTPUT_DIR / "performance_trimmed.wav")
    trimmed_duration = trim_audio(CONCAT_FILE, TRIM_START_S, trim_end, trimmed_audio)

    # --- Step 3: Load model ---
    print(f"\n[3/4] Loading SAM Audio model: {args.model}")
    print(f"    First run downloads weights — may take a while...")

    from sam_audio import SAMAudio, SAMAudioProcessor

    model = SAMAudio.from_pretrained(args.model)
    processor = SAMAudioProcessor.from_pretrained(args.model)
    model = model.eval().to(device)

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"    Model loaded. VRAM used: {vram_used:.1f} GB")

    # --- Step 4: Separate actress's voice in chunks ---
    CHUNK_S = 45.0  # seconds per chunk — base model uses ~41GB at 45s on A100 80GB
    OVERLAP_S = 5.0  # overlap for crossfade between chunks
    description = "a live woman's voice performing in a room with natural acoustics"

    n_chunks = int(np.ceil(trimmed_duration / (CHUNK_S - OVERLAP_S)))
    print(f"\n[4/4] Separating actress's voice ({trimmed_duration:.0f}s) "
          f"in {n_chunks} chunks of {CHUNK_S:.0f}s...")

    waveform, sr = torchaudio.load(trimmed_audio)
    voice_chunks = []
    residual_chunks = []

    for i in range(n_chunks):
        chunk_start = i * (CHUNK_S - OVERLAP_S)
        chunk_end = min(chunk_start + CHUNK_S, trimmed_duration)
        if chunk_end - chunk_start < 1.0:
            break

        print(f"\n  --- Chunk {i+1}/{n_chunks}: {chunk_start:.1f}s - {chunk_end:.1f}s ---")

        # Extract chunk waveform
        s_sample = int(chunk_start * sr)
        e_sample = int(chunk_end * sr)
        chunk_wav = waveform[:, s_sample:e_sample]
        chunk_path = str(OUTPUT_DIR / f"_chunk_{i}.wav")
        torchaudio.save(chunk_path, chunk_wav, sr)

        # Filter anchors to this chunk's time range (relative to trimmed audio)
        chunk_anchors = []
        for sign, st, en in anchors:
            # Anchor overlaps this chunk?
            if en > chunk_start and st < chunk_end:
                chunk_anchors.append((sign, st - chunk_start, en - chunk_start))

        # Run separation on this chunk
        target_path, residual_path = run_separation(
            audio_path=chunk_path,
            description=description,
            anchors=chunk_anchors,
            output_prefix=f"_chunk_{i}",
            model=model,
            processor=processor,
            device=device,
            reranking_candidates=reranking,
        )

        # Load results
        voice_chunk, _ = torchaudio.load(str(target_path))
        residual_chunk, _ = torchaudio.load(str(residual_path))
        voice_chunks.append(voice_chunk)
        residual_chunks.append(residual_chunk)

        # Free VRAM between chunks
        torch.cuda.empty_cache()

    # Crossfade and concatenate chunks
    print(f"\n  Crossfading and concatenating {len(voice_chunks)} chunks...")
    overlap_samples = int(OVERLAP_S * sr)

    def crossfade_concat(chunks):
        if len(chunks) == 1:
            return chunks[0]
        result = chunks[0]
        for chunk in chunks[1:]:
            # Length of overlap region (may be shorter if chunk is short)
            ol = min(overlap_samples, result.shape[1], chunk.shape[1])
            if ol > 0:
                fade_out = torch.linspace(1, 0, ol).unsqueeze(0)
                fade_in = torch.linspace(0, 1, ol).unsqueeze(0)
                result[:, -ol:] = result[:, -ol:] * fade_out + chunk[:, :ol] * fade_in
                result = torch.cat([result, chunk[:, ol:]], dim=1)
            else:
                result = torch.cat([result, chunk], dim=1)
        return result

    voice_full = crossfade_concat(voice_chunks)
    residual_full = crossfade_concat(residual_chunks)

    actress_voice = OUTPUT_DIR / "actress_voice.wav"
    actress_residual = OUTPUT_DIR / "actress_residual.wav"
    torchaudio.save(str(actress_voice), voice_full, sr)
    torchaudio.save(str(actress_residual), residual_full, sr)
    print(f"    -> Voice:    {actress_voice} ({voice_full.shape[1]/sr:.1f}s)")
    print(f"    -> Residual: {actress_residual} ({residual_full.shape[1]/sr:.1f}s)")

    # Clean up chunk files
    for f in OUTPUT_DIR.glob("_chunk_*"):
        f.unlink()

    # --- Optional: Judge the output ---
    run_judge(str(actress_voice), description)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DONE. Output files:")
    for f in sorted(OUTPUT_DIR.glob("*.wav")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:40s} {size_mb:8.1f} MB")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Listen to actress_voice.wav — check for hallucinations")
    print("     and whether her enunciation is preserved faithfully.")
    print("  2. Compare actress_residual.wav against the clean backing")
    print("     track recording to verify music/narrator were removed.")
    print("  3. If good, run enhance_voice.py for light cleanup.")
    print("  4. Import into DAW alongside the clean backing track.")
    print()
    print("If quality is poor, try:")
    print("  - Different/more span anchors (most impactful change)")
    print("  - Visual prompting (see sam-audio repo's visual example)")
    print("  - Running Mel-RoFormer via audio-separator for comparison")


if __name__ == "__main__":
    main()
