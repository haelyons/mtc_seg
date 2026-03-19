# Voice Separation for "Instructions On How To Be Alone"

## What this does

Isolates the performer's voice from a ~40-minute audience-perspective recording
of a one-woman play, using Meta's SAM Audio (December 2025) with text + span
prompting. Optionally enhances the separated voice with Resemble Enhance.

## The problem

Both the performer and the director are women. SAM Audio's text prompt
`"a woman speaking"` maps directly to how the model was trained (their
training pipeline used a gender classifier → template like `female → "woman
speaking"`). But this prompt alone will capture BOTH women and likely some
audience members too.

**Span anchors are what disambiguate.** They tell the model "the specific
woman I want is speaking HERE (positive) and is NOT speaking HERE (negative)."
The text prompt handles the semantic category; the spans handle identity.

## Files

```
setup.sh            — Run first on Lambda. Installs everything, downloads weights.
separate_voice.py   — Main separation script. Edit CONFIG section, then run.
enhance_voice.py    — Optional post-separation "upscaling" via Resemble Enhance.
```

## Quick start on Lambda

```bash
# 1. SSH in
ssh user@your-lambda-instance

# 2. Clone this repo (or scp files)
git clone <your-repo> && cd <your-repo>

# 3. Upload the recording
scp instructions_on_how_to_be_alone.mp4 user@lambda-ip:~/repo/

# 4. Run setup (installs deps, downloads model weights)
bash setup.sh

# 5. Edit separate_voice.py — fill in YOUR timestamps and anchors
#    (see "How to pick anchors" below)
nano separate_voice.py

# 6. Test on first 5 minutes
python separate_voice.py --test

# 7. Listen to output/performer_voice.wav
#    If it sounds good, run the full thing:
python separate_voice.py

# 8. Optional: enhance
python enhance_voice.py output/performer_voice.wav
```

## How to pick anchors (the most important step)

Open the recording in any audio player with a waveform view (Ableton is fine).
You need to identify:

**Positive anchors (+):** 3-6 passages of ~10 seconds each where the
performer is clearly speaking with minimal overlap from music or audience
noise. Spread them across the full performance — one from the opening, a
couple from the middle, one from near the end. These teach the model what
her voice sounds like throughout the show. If her vocal quality or energy
changes over the performance (which it likely does in a 25-minute piece),
capturing that range in the anchors helps.

**Negative anchors (-):** 2-3 passages where the performer is definitely
NOT speaking. Music-only interludes are ideal. Audience reactions (applause,
laughter) work too. The very start of her section (before she begins
speaking) is a natural negative anchor.

**Times are relative to the trimmed section**, not the original recording.
If PERFORMER_START_S is 90.0 and you want to mark a passage at 3:00 in the
original, the anchor time is 3:00 - 1:30 = 1:30 = 90.0 seconds.

You don't need to be precise — within a few seconds is fine. The model uses
these as hints, not hard boundaries.

## What to listen for

After separation, listen critically for:

1. **Hallucinated details**: SAM Audio is generative (diffusion-based). It
   may synthesize plausible-sounding speech details that weren't in the
   original. Compare against the original recording — does her phrasing
   match exactly?

2. **Amplitude fidelity**: The output amplitude may not match the original
   mix. This is normal for generative separation and easily fixed in Ableton.

3. **Boundary artifacts**: At chunk boundaries (every ~10s), listen for
   subtle discontinuities. The multi-diffusion overlap should handle this,
   but check transitions.

4. **Missing speech**: Does the model drop any quiet passages or whispered
   sections? These are hardest for any separator.

5. **Audience bleed**: Are coughs, rustling, or audience reactions leaking
   into the voice stem? This is where better anchors help most.

## If it's not good enough

In order of impact:

1. **Better anchors** — Pick clearer passages. Add more negatives
   specifically around the problem areas (e.g. if audience laughter at
   minute 15 is bleeding through, add a negative anchor there).

2. **Visual prompting** — Since she's in frame the whole time, visual
   prompting gives an unambiguous spatial anchor. More engineering work
   but potentially much better results. See the sam-audio repo's visual
   prompting example.

3. **Mel-RoFormer instead** — Run `audio-separator` with a Mel-RoFormer
   model. It's discriminative (no hallucinations), much faster, and may
   actually be more faithful for this use case. Worse at "extracting"
   but better at not inventing things.

4. **Combine approaches** — Use Mel-RoFormer for faithful separation,
   then SAM Audio on specific problem passages where you need more
   aggressive extraction.

## The enhancement question

Resemble Enhance and VoiceFixer can genuinely improve separated speech,
but they are ALSO generative models. Each stage in the pipeline
(separation → denoising → enhancement) adds a layer of generation that
moves further from the original recording. For a project where preserving
her enunciation is the whole point, use the lightest processing chain
that achieves acceptable intelligibility. More processing ≠ better.

## Hardware notes

- **A100 80GB**: Comfortable for sam-audio-large with reranking_candidates=8
- **A100 40GB**: Should work for sam-audio-large with reranking_candidates=4
  or sam-audio-base with reranking_candidates=8
- **Processing time**: ~28 minutes for 40 minutes of audio at 0.7x RTF
  (separation only, before reranking overhead)
- The model also loads PE-AV and potentially the Judge model, so total
  VRAM usage is higher than the model weights alone suggest
