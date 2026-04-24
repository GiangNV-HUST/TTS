# OmniVoice Vietnamese TTS

Vietnamese Text-to-Speech using OmniVoice voice cloning model. Target: natural voice for IVR, virtual assistants, robots.

## Project Structure

```
TTS/
├── omnivoice_test/
│   ├── run_test.py           # Voice cloning test script
│   ├── extract_best_clip.py  # Extract best clip from long audio for ref
│   ├── preprocess_ref.py     # Preprocess ref audio (resample, denoise, normalize)
│   ├── ref_audio/            # Reference audio files
│   ├── ref_text/ref_config.json  # Speaker -> ref_text mapping
│   ├── test_data/
│   │   ├── test_sentences.jsonl  # Test sentences
│   │   └── ivr_sentences.jsonl   # IVR sentences
│   └── outputs/              # Generated audio
├── download_model.py         # Download model weights
└── download_all_deps.py      # Download dependencies
```

## Usage

```bash
cd omnivoice_test
conda activate omnivoice

# Full command for IVR generation
python run_test.py \
    --ref_audio ref_audio/youtube_cut.wav \
    --test_file test_data/ivr_sentences.jsonl \
    --steps 64 \
    --chunk \
    --speed 1.0 \
    --sentence-gap-ms 300 \
    --clause-gap-ms 150 \
    --comma-replace " , " \
    --end-padding-ms 150 \
    --no_eval

# Basic run
python run_test.py --ref_audio ref_audio/speaker.wav --steps 32 --no_eval

# Extract best clip from long audio
python extract_best_clip.py long_audio.wav -o ref_audio/speaker.wav --min-duration 5 --max-duration 10

# Preprocess ref audio (optional - skip if audio is already clean)
python preprocess_ref.py --in-dir ref_audio --no-denoise --max-duration 10
python run_test.py --use-clean --ref_audio ref_audio_clean/speaker.wav
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ref_audio` | Path to reference audio file | all in ref_audio/ |
| `--test_file` | Path to test sentences JSONL | test_sentences.jsonl |
| `--steps` | Diffusion steps (higher = better quality, slower) | 8, 16, 32 |
| `--chunk` | Split long text into sentences, gen each, concat | False |
| `--speed` | Speech speed (0.9 = slow, 1.1 = fast) | 1.0 |
| `--sentence-gap-ms` | Pause between sentences (when using --chunk) | 350 |
| `--clause-gap-ms` | Pause after commas | 150 |
| `--comma-replace` | Replace commas with this string | " . " |
| `--end-padding-ms` | Silence padding at end to avoid cut-off | 100 |
| `--no_eval` | Skip UTMOS/WER/SpkSim evaluation | False |

## Text Normalization

- Numbers to words: `0947` → `không chín bốn bảy`
- Abbreviations: `NĐ-CP` → `nờ đê xê pê`
- English words phonetically: `sale` → `xêu`
- URLs: `vinaphone.com.vn` → `vinaphone chấm com chấm vi en`
- Phone numbers: `1800.1091` → `một tám không không một không chín một`
- Always end sentences with punctuation (. ! ?) to avoid audio cut-off

## Tips

- Ref audio: 5-15 seconds, mono, clear without noise
- Use `--steps 48` or `64` for high quality
- Use `--chunk` for long IVR text
- Use `--comma-replace " , "` to keep sentence structure (fewer chunks)
- Use `--comma-replace " - "` for medium pause
- Increase `--end-padding-ms` to 200-300 if last word is cut off
- Add ref_text to `ref_config.json` for each speaker to avoid network calls
