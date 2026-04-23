#!/bin/bash
set -e

echo "=== Setup OmniVoice Test Environment ==="
echo "Server: A100 80GB | CUDA Driver: 12.2 | Cần dùng cu121"

# 1. Tạo conda env
conda create -n omnivoice python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate omnivoice

# 2. Cài PyTorch (PyPI default, tự bundle CUDA runtime)
# Server proxy block download.pytorch.org/whl, dùng bản PyPI thay thế
pip install torch==2.8.0 torchaudio==2.8.0

# 3. Cài OmniVoice + eval tools
pip install omnivoice
pip install "omnivoice[eval]"

# 4. Cài thêm Whisper + jiwer cho WER evaluation
pip install openai-whisper jiwer

# 5. Verify
python -c "
from omnivoice import OmniVoice
import torch
print(f'OmniVoice imported OK')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM total: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'VRAM free:  {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB')
"

echo "=== Setup complete! ==="
echo "Next: add .wav files to ref_audio/, then run: python run_test.py --quick"
