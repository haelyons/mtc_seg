#!/bin/bash
# ===========================================================================
# Lambda GPU Setup for SAM Audio
# ===========================================================================
# SSH onto your Lambda instance, then run:
#   bash setup.sh
#
# Prerequisites:
#   - Request access to SAM Audio weights on HuggingFace BEFORE running this:
#     https://huggingface.co/facebook/sam-audio-large
#     (Approval typically takes 10-30 minutes)
#   - Have your HuggingFace token ready
# ===========================================================================

set -e

echo "============================================"
echo "SAM Audio Setup for Lambda GPU"
echo "============================================"
echo ""

# --- Check GPU ---
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- System deps ---
echo "[2/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg > /dev/null 2>&1
echo "    ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
echo ""

# --- Conda environment ---
echo "[3/6] Setting up Python environment..."
# Lambda instances typically have conda pre-installed.
# If not, adjust to use venv instead.
conda create -n samaudio python=3.11 -y -q 2>/dev/null || true
conda activate samaudio 2>/dev/null || source activate samaudio 2>/dev/null || {
    echo "    conda activate failed, trying venv..."
    python3 -m venv samaudio_env
    source samaudio_env/bin/activate
}
echo "    Python: $(python --version)"
echo ""

# --- Install SAM Audio ---
echo "[4/6] Installing SAM Audio and dependencies..."
pip install -q --upgrade pip

# SAM Audio from GitHub (official repo)
pip install -q git+https://github.com/facebookresearch/sam-audio.git

# Additional deps for our pipeline
pip install -q torchaudio

# For the optional enhancement step
pip install -q resemble-enhance

echo "    SAM Audio installed."
echo ""

# --- HuggingFace authentication ---
echo "[5/6] HuggingFace authentication..."
echo "    You need a HF token with access to facebook/sam-audio-large"
echo "    Get one at: https://huggingface.co/settings/tokens"
echo ""

if command -v huggingface-cli &> /dev/null; then
    huggingface-cli login
else
    pip install -q huggingface_hub
    huggingface-cli login
fi
echo ""

# --- Download model weights (pre-cache) ---
echo "[6/6] Pre-downloading model weights..."
echo "    This will download SAM Audio + PE-AV (~15-20GB total)"
echo "    Go get a coffee..."
python -c "
from sam_audio import SAMAudio, SAMAudioProcessor
print('Downloading SAM Audio large...')
SAMAudio.from_pretrained('facebook/sam-audio-large')
SAMAudioProcessor.from_pretrained('facebook/sam-audio-large')
print('Done.')
"
echo ""

echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Upload your video/audio file:"
echo "     scp instructions_on_how_to_be_alone.mp4 user@lambda-ip:~/"
echo ""
echo "  2. Edit separate_voice.py — fill in the CONFIG timestamps:"
echo "     - PERFORMER_START_S / PERFORMER_END_S"
echo "     - DIRECTOR_START_S / DIRECTOR_END_S"
echo "     - PERFORMER_ANCHORS (the span markers)"
echo ""
echo "  3. Test run on first 5 minutes:"
echo "     python separate_voice.py --test"
echo ""
echo "  4. If that sounds good, full run:"
echo "     python separate_voice.py"
echo ""
echo "  5. Optional enhancement:"
echo "     python enhance_voice.py output/performer_voice.wav"
echo "============================================"
