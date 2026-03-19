#!/usr/bin/env python3
"""
Voice separation for "Instructions On How To Be Alone"
=======================================================
Isolates the performer's voice from a ~40min audience-perspective recording
using Meta's SAM Audio with text + span prompting.

Both speakers (performer and director) are women, so "a woman speaking" alone
won't disambiguate them. Span anchors are the critical differentiator:
  (+) marks time ranges where the PERFORMER is clearly speaking
  (-) marks time ranges where she is definitely NOT speaking
      (audience-only, director outro, music-only passages)

The model was trained on 10s clips. For long audio it uses multi-diffusion:
overlapping windows merged via soft masks at each flow-matching step. The
separate_long() pathway (from the MLX port) or the native multi-diffusion
in the official repo handle this automatically.

Usage:
  1. Edit the CONFIG section below with your file paths and timestamps
  2. Run: python separate_voice.py
  3. Outputs land in ./output/

Hardware: Tested targeting A100 80GB. The large model + reranking will use
significant VRAM. If OOM, reduce reranking_candidates or use sam-audio-base.
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

import torch
import torchaudio

# ============================================================================
# CONFIG — Edit this section before running
# ============================================================================

# Path to the video file (will extract audio) or direct audio file
VIDEO_FILE = "instructions_on_how_to_be_alone.mp4"

# If you already extracted the audio, point here instead and set SKIP_EXTRACT=True
AUDIO_FILE = "full_audio.wav"
SKIP_EXTRACT = False

# Model choice: "facebook/sam-audio-large" or "facebook/sam-audio-base"
# Large (~3B params) is better but heavier. Base (~1B) is the fallback.
MODEL_ID = "facebook/sam-audio-large"

# Quality vs speed. 8 = best quality, generates 8 candidates and picks best.
# Drop to 4 or even 1 if VRAM is tight or you want a faster test run.
RERANKING_CANDIDATES = 8

# Enable span prediction — critical for speech (event-based, not ambient)
PREDICT_SPANS = True

# Output directory
OUTPUT_DIR = Path("./output")

# --- TIMESTAMPS (seconds) ---
# Your colleague needs to fill these in by scrubbing through the recording.
# They don't need to be frame-accurate — within a few seconds is fine.

# Where does the performer's section start and end? (approx)
# Everything outside this range will be trimmed before processing.
PERFORMER_START_S = 90.0     # e.g. 1:30 — after audience settles
PERFORMER_END_S   = 1590.0   # e.g. 26:30 — before director begins

# Where does the director's outro start and end?
DIRECTOR_START_S  = 1600.0   # e.g. 26:40
DIRECTOR_END_S    = 1750.0   # e.g. 29:10

# --- SPAN ANCHORS FOR THE PERFORMER ---
# Pick 3-6 moments where the performer is CLEARLY and SOLO speaking.
# These are positive examples. Pick passages with minimal background music
# if possible — the model uses these to learn what "this specific woman
# speaking" sounds like as opposed to audience coughs, reactions, etc.
#
# Also pick 2-3 negative examples: moments within her section where she
# is NOT speaking (music-only interludes, deliberate silences, audience
# reactions). These teach the model what to exclude.
#
# Format: ("+"/"-", start_seconds, end_seconds)
# Times are RELATIVE TO THE TRIMMED AUDIO (i.e. 0.0 = PERFORMER_START_S)

PERFORMER_ANCHORS = [
    # Positive: she is clearly speaking here
    ("+",  30.0,  40.0),   # ~2:00-2:10 in original — adjust to a clear passage
    ("+", 300.0, 310.0),   # ~6:30-6:40 — another clear passage
    ("+", 600.0, 610.0),   # ~11:30-11:40
    ("+", 900.0, 910.0),   # ~16:30-16:40

    # Negative: she is NOT speaking here (music only, silence, audience)
    ("-",   0.0,   5.0),   # very start of her section, may still be settling
    # Add more negatives if there are known music-only interludes:
    # ("-", 450.0, 460.0),  # e.g. a musical interlude around 9:00
]

# --- SPAN ANCHORS FOR THE DIRECTOR (if you want her isolated too) ---
# Same logic but reversed: positive where director speaks, negative where not.
DIRECTOR_ANCHORS = [
    ("+", 10.0, 20.0),   # 10-20s into the director's section
    # Negative examples are less critical here since it's short
]


# ============================================================================
# STEP 1: Extract audio from video
# ============================================================================

def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video as WAV using ffmpeg."""
    print(f"[1/5] Extracting audio from {video_path}...")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ar", "44100",         # 44.1kHz — SAM Audio's native rate
        "-ac", "1",             # mono (SAM Audio processes mono anyway)
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        sys.exit(1)
    print(f"    -> Saved to {audio_path}")


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

    print(f"    Running separation (reranking_candidates={RERANKING_CANDIDATES})...")
    t0 = time.time()

    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=PREDICT_SPANS,
            reranking_candidates=RERANKING_CANDIDATES,
        )

    elapsed = time.time() - t0
    print(f"    Separation took {elapsed:.1f}s ({elapsed/60:.1f}m)")

    sr = processor.audio_sampling_rate

    target_path = OUTPUT_DIR / f"{output_prefix}_voice.wav"
    residual_path = OUTPUT_DIR / f"{output_prefix}_residual.wav"

    # Handle shape: result.target may be [batch, samples] or [samples]
    target = result.target.cpu()
    residual = result.residual.cpu()
    if target.dim() == 1:
        target = target.unsqueeze(0)
    if residual.dim() == 1:
        residual = residual.unsqueeze(0)

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

def main():
    parser = argparse.ArgumentParser(description="SAM Audio voice separation")
    parser.add_argument("--test", action="store_true",
                        help="Process only first 5 minutes as a test run")
    parser.add_argument("--skip-director", action="store_true",
                        help="Skip director outro separation")
    parser.add_argument("--reranking", type=int, default=RERANKING_CANDIDATES,
                        help="Override reranking_candidates (1-8)")
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        help="Model ID (e.g. facebook/sam-audio-base for less VRAM)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Device check ---
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. SAM Audio requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print()

    # --- Step 1: Extract audio ---
    if not SKIP_EXTRACT:
        if not os.path.exists(VIDEO_FILE):
            print(f"ERROR: Video file not found: {VIDEO_FILE}")
            print("Either place the video file here, or set SKIP_EXTRACT=True")
            print("and point AUDIO_FILE to your pre-extracted audio.")
            sys.exit(1)
        extract_audio(VIDEO_FILE, AUDIO_FILE)
    else:
        if not os.path.exists(AUDIO_FILE):
            print(f"ERROR: Audio file not found: {AUDIO_FILE}")
            sys.exit(1)
        print(f"[1/5] Using pre-extracted audio: {AUDIO_FILE}")

    # --- Step 2: Trim sections ---
    print(f"\n[2/5] Trimming audio to performer and director sections...")

    performer_end = PERFORMER_END_S
    if args.test:
        performer_end = min(PERFORMER_START_S + 300.0, PERFORMER_END_S)
        print(f"    TEST MODE: only processing first 5 minutes of performer")

    performer_audio = str(OUTPUT_DIR / "performer_section.wav")
    performer_duration = trim_audio(AUDIO_FILE, PERFORMER_START_S, performer_end, performer_audio)

    director_audio = None
    if not args.skip_director:
        director_audio = str(OUTPUT_DIR / "director_section.wav")
        trim_audio(AUDIO_FILE, DIRECTOR_START_S, DIRECTOR_END_S, director_audio)

    # --- Step 3: Load model ---
    print(f"\n[3/5] Loading SAM Audio model: {args.model}")
    print(f"    This downloads several models (SAM Audio + PE-AV + Judge).")
    print(f"    First run will take a while...")

    from sam_audio import SAMAudio, SAMAudioProcessor

    model = SAMAudio.from_pretrained(args.model)
    processor = SAMAudioProcessor.from_pretrained(args.model)
    model = model.eval().to(device)

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"    Model loaded. VRAM used: {vram_used:.1f} GB")

    # --- Step 4: Separate performer's voice ---
    print(f"\n[4/5] Separating performer's voice ({performer_duration:.0f}s)...")

    performer_voice, performer_residual = run_separation(
        audio_path=performer_audio,
        description="a woman speaking",
        anchors=PERFORMER_ANCHORS,
        output_prefix="performer",
        model=model,
        processor=processor,
        device=device,
    )

    # --- Step 5: Separate director's voice (if requested) ---
    if director_audio and not args.skip_director:
        print(f"\n[5/5] Separating director's voice...")
        run_separation(
            audio_path=director_audio,
            description="a woman speaking",
            anchors=DIRECTOR_ANCHORS,
            output_prefix="director",
            model=model,
            processor=processor,
            device=device,
        )
    else:
        print(f"\n[5/5] Skipping director separation.")

    # --- Optional: Judge the output ---
    run_judge(performer_voice, "a woman speaking")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DONE. Output files:")
    for f in sorted(OUTPUT_DIR.glob("*.wav")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:40s} {size_mb:8.1f} MB")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Listen to performer_voice.wav — check for hallucinations")
    print("     and whether her enunciation is preserved faithfully.")
    print("  2. If good, consider running through Resemble Enhance for")
    print("     bandwidth restoration and denoising (see enhance_voice.py).")
    print("  3. Import into Ableton alongside the clean music recording.")
    print("  4. Blend ~5-15% of the original untrimmed audio for naturalism.")
    print()
    print("If quality is poor, try:")
    print("  - Different/more span anchors (most impactful change)")
    print("  - Visual prompting (see sam-audio repo's visual example)")
    print("  - Running Mel-RoFormer via audio-separator for comparison")


if __name__ == "__main__":
    main()
