# OmniVoice Vietnamese TTS - Project Context

## Overview
Project test và phát triển TTS tiếng Việt sử dụng OmniVoice (voice cloning model). Mục tiêu: tạo giọng đọc tự nhiên cho IVR, trợ lý ảo, robot.

## Cấu trúc thư mục

```
TTS/
├── omnivoice_test/           # Thư mục test chính
│   ├── run_test.py           # Script chạy voice cloning test
│   ├── preprocess_ref.py     # Preprocess ref audio (resample, denoise, normalize)
│   ├── mos_evaluation.py     # Giao diện Gradio đánh giá MOS (không liên quan gen voice)
│   ├── ref_audio/            # Ref audio gốc (chưa xử lý)
│   ├── ref_audio_clean/      # Ref audio đã preprocess
│   ├── ref_text/
│   │   └── ref_config.json   # Mapping speaker -> ref_text
│   ├── test_data/
│   │   └── test_sentences.jsonl  # Các câu test (JSONL format)
│   ├── outputs/              # Audio output theo speaker
│   └── results/              # CSV metrics, JSON summary
├── ref_audio/                # Ref audio upload từ local
└── OmniVoice/                # Model weights (trên server)
```

## Workflow chính

### 1. Thêm ref audio mới
```bash
# Local: Convert audio sang WAV nếu cần
ffmpeg -i "input.m4a" -ar 44100 -ac 1 "output.wav"

# Cắt audio nếu quá dài (Python)
from scipy.io import wavfile
rate, data = wavfile.read("input.wav")
clip = data[:int(7.5 * rate)]  # 7.5 giây đầu
wavfile.write("output.wav", rate, clip)
```

### 2. Cập nhật ref_config.json
```json
{
  "speaker_name": {
    "ref_text": "nội dung transcript của ref audio, chuẩn hóa dấu câu và từ vựng"
  }
}
```

### 3. Chạy test trên server
```bash
cd ~/GiangNVCode/TTS/omnivoice_test
conda activate omnivoice

# Không preprocess (dùng audio gốc)
python run_test.py --ref_audio ref_audio/speaker.wav \
    --steps 48 \
    --chunk \
    --speed 1.0 \
    --sentence-gap-ms 300 \
    --clause-gap-ms 150 \
    --comma-replace " , " \
    --no_eval

# Có preprocess trước
python preprocess_ref.py --in-dir ref_audio --max-duration 10
python run_test.py --use-clean --ref_audio ref_audio_clean/speaker.wav ...
```

## Các tham số quan trọng run_test.py

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--steps` | Diffusion steps (cao = chất lượng tốt, chậm) | 8, 16, 32 |
| `--chunk` | Chia text dài thành câu, gen từng câu rồi ghép | False |
| `--speed` | Tốc độ nói (0.9 = chậm, 1.1 = nhanh) | 1.0 |
| `--sentence-gap-ms` | Pause giữa các câu (khi dùng --chunk) | 350 |
| `--clause-gap-ms` | Pause sau dấu phẩy | 150 |
| `--comma-replace` | Thay dấu phẩy bằng gì (" . " = nhiều chunk, " , " = giữ nguyên) | " . " |
| `--use-clean` | Dùng ref_audio_clean/ thay vì ref_audio/ | False |
| `--no_eval` | Bỏ qua UTMOS/WER/SpkSim eval | False |

## Chuẩn hóa text cho TTS

- Số đọc thành chữ: `0947` → `không chín bốn bảy`
- Viết tắt đọc ra: `NĐ-CP` → `nờ đê xê pê`
- Từ tiếng Anh phiên âm: `sale` → `xêu` (theo cách phát âm thực tế)
- Dấu gạch ngang phân tách số: `0947-414-891` → `không chín bốn bảy - bốn một bốn - tám chín một`

## Server info

- SSH: `hungtt@lab-ai-01` (vnpt-server)
- Path: `~/GiangNVCode/TTS/omnivoice_test`
- Conda env: `omnivoice`
- GPU: CUDA available

## Speakers hiện có

| Speaker | Ref audio | Mô tả |
|---------|-----------|-------|
| female_mb | female_mb.wav | Giọng nữ miền Bắc |
| female_mn | female_mn.wav | Giọng nữ miền Nam |
| hy_nhi | hy_nhi.wav | Giọng nữ Hỷ Nhi (7.5s, từ file quảng cáo sách) |
| ban_ghi_5 | ban_ghi_5.wav | Giọng mới thu âm (6.5s) |

## Tips

- Ref audio nên 5-15 giây, mono, rõ ràng không noise
- `--steps 48` hoặc `64` cho chất lượng cao
- Dùng `--comma-replace " , "` để giữ nguyên cấu trúc câu, ít chunks
- Text IVR dài nên dùng `--chunk` để chia nhỏ
- Hạ pitch sau gen: dùng `librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)`

## Mục tiêu dài hạn

Tích hợp TTS + STT + Chatbot cho robot trợ lý để bàn:
- Phần cứng: Raspberry Pi 5 + mic + speaker
- Kiến trúc: Hybrid (wake word local, xử lý trên server GPU)
- Stack: Porcupine (wake word) → Whisper (STT) → LLM → OmniVoice (TTS)
