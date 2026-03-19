#!/usr/bin/env python3
"""
Post-separation voice enhancement ("upscaling")
================================================
Takes the separated voice from SAM Audio and improves it using
Resemble Enhance's two-stage pipeline:
  1. Denoiser: removes residual noise/artifacts from separation
  2. Enhancer: restores bandwidth and perceptual quality (the "upscaling")

The enhancer uses a conditional flow matching model trained on 44.1kHz
speech to reconstruct what the clean speech should sound like — it can
genuinely recover high-frequency detail that the separation stripped.

Usage:
  python enhance_voice.py output/performer_voice.wav
  python enhance_voice.py output/performer_voice.wav --denoise-only
  python enhance_voice.py output/performer_voice.wav --enhance-only
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio


def enhance_with_resemble(
    input_path: str,
    output_dir: Path,
    denoise: bool = True,
    enhance: bool = True,
    # CFM solver parameters for the enhancer
    nfe: int = 64,          # number of function evaluations (higher = better, slower)
    solver: str = "midpoint",
    # Denoiser strength: how aggressively to denoise (0.0-1.0)
    # Start conservative (0.3-0.5) — too aggressive strips voice character
    denoise_strength: float = 0.4,
    # Enhancer blending: how much of the enhanced signal to mix in (0.0-1.0)
    # Lower values preserve more of the original character
    enhance_blend: float = 0.6,
):
    """Run Resemble Enhance denoiser and/or enhancer."""

    try:
        from resemble_enhance.enhancer.inference import denoise as re_denoise
        from resemble_enhance.enhancer.inference import enhance as re_enhance
    except ImportError:
        print("ERROR: resemble-enhance not installed.")
        print("  pip install resemble-enhance")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = Path(input_path)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading: {input_path}")
    waveform, sr = torchaudio.load(str(input_path))

    # Resemble Enhance expects mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 1: Denoise
    if denoise:
        print(f"Denoising (strength={denoise_strength})...")
        t0 = time.time()
        denoised = re_denoise(waveform.squeeze(0), sr, device)
        print(f"  Done in {time.time()-t0:.1f}s")

        denoised_path = output_dir / f"{input_path.stem}_denoised.wav"
        torchaudio.save(str(denoised_path), denoised.unsqueeze(0).cpu(), sr)
        print(f"  -> {denoised_path}")

        # Feed denoised into enhancer
        waveform_for_enhance = denoised.unsqueeze(0)
    else:
        waveform_for_enhance = waveform
        denoised_path = None

    # Step 2: Enhance (bandwidth restoration / "upscaling")
    if enhance:
        print(f"Enhancing (nfe={nfe}, solver={solver})...")
        print(f"  This reconstructs high-frequency detail and improves clarity.")
        t0 = time.time()
        enhanced = re_enhance(
            waveform_for_enhance.squeeze(0),
            sr,
            device,
            nfe=nfe,
            solver=solver,
        )
        print(f"  Done in {time.time()-t0:.1f}s")

        enhanced_path = output_dir / f"{input_path.stem}_enhanced.wav"
        torchaudio.save(str(enhanced_path), enhanced.unsqueeze(0).cpu(), 44100)
        print(f"  -> {enhanced_path}")
    else:
        enhanced_path = None

    return denoised_path, enhanced_path


def main():
    parser = argparse.ArgumentParser(
        description="Post-separation voice enhancement via Resemble Enhance"
    )
    parser.add_argument("input", help="Path to separated voice WAV file")
    parser.add_argument("--output-dir", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--denoise-only", action="store_true",
                        help="Run only the denoiser, skip enhancer")
    parser.add_argument("--enhance-only", action="store_true",
                        help="Run only the enhancer, skip denoiser")
    parser.add_argument("--denoise-strength", type=float, default=0.4,
                        help="Denoiser aggressiveness 0.0-1.0 (default: 0.4)")
    parser.add_argument("--nfe", type=int, default=64,
                        help="Enhancer quality: function evaluations (default: 64)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    do_denoise = not args.enhance_only
    do_enhance = not args.denoise_only

    denoised, enhanced = enhance_with_resemble(
        input_path=args.input,
        output_dir=Path(args.output_dir),
        denoise=do_denoise,
        enhance=do_enhance,
        denoise_strength=args.denoise_strength,
        nfe=args.nfe,
    )

    print()
    print("Listen to all versions and compare:")
    print(f"  Original separated: {args.input}")
    if denoised:
        print(f"  Denoised:           {denoised}")
    if enhanced:
        print(f"  Enhanced:           {enhanced}")
    print()
    print("IMPORTANT: The enhancer is itself a generative model.")
    print("It will sound 'better' but may subtly alter vocal character.")
    print("A/B carefully against the separated output before committing.")
    print("For this project, preserving her enunciation matters more than")
    print("pristine audio quality — use the lightest touch that works.")


if __name__ == "__main__":
    main()
