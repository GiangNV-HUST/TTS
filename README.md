# OmniVoice Vietnamese TTS

Vietnamese Text-to-Speech using OmniVoice voice cloning model. Target: natural voice for IVR, virtual assistants, robots.

## Project Structure

```
TTS/
├── omnivoice_test/
│   ├── run_test.py           # Voice cloning test script
│   ├── preprocess_ref.py     # Preprocess ref audio (resample, denoise, normalize)
│   ├── ref_audio/            # Reference audio files
│   ├── ref_text/ref_config.json  # Speaker -> ref_text mapping
│   ├── test_data/test_sentences.jsonl  # Test sentences
│   └── outputs/              # Generated audio
├── download_model.py         # Download model weights
└── download_all_deps.py      # Download dependencies
```

## Usage

```bash
conda activate omnivoice

# Basic run
python run_test.py --ref_audio ref_audio/speaker.wav --steps 32 --no_eval

# With chunking for long text
python run_test.py --ref_audio ref_audio/speaker.wav \
    --steps 48 \
    --chunk \
    --speed 1.0 \
    --sentence-gap-ms 300 \
    --clause-gap-ms 150

# Preprocess ref audio first
python preprocess_ref.py --in-dir ref_audio --max-duration 10
python run_test.py --use-clean --ref_audio ref_audio_clean/speaker.wav
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--steps` | Diffusion steps (higher = better quality, slower) | 8 |
| `--chunk` | Split long text into sentences | False |
| `--speed` | Speech speed (0.9 = slow, 1.1 = fast) | 1.0 |
| `--sentence-gap-ms` | Pause between sentences | 350 |
| `--clause-gap-ms` | Pause after commas | 150 |
| `--no_eval` | Skip UTMOS/WER/SpkSim evaluation | False |

## Text Normalization

- Numbers to words: `0947` → `không chín bốn bảy`
- Abbreviations: `NĐ-CP` → `nờ đê xê pê`
- English words phonetically: `sale` → `xêu`

## Tips

- Ref audio: 5-15 seconds, mono, clear without noise
- Use `--steps 48` or `64` for high quality
- Use `--chunk` for long IVR text
