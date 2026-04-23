"""
Tải tất cả model dependencies của OmniVoice về local.
Chạy trên máy có internet nhanh, rồi copy lên server.
"""
from huggingface_hub import snapshot_download
import os

SAVE_DIR = "C:/data/models"

# 1. Model chính OmniVoice
print("=== [1/4] Tải OmniVoice model chính ===")
snapshot_download(
    repo_id="k2-fsa/OmniVoice",
    local_dir=os.path.join(SAVE_DIR, "OmniVoice"),
)
print("Done.\n")

# 2. Tải các model phụ mà OmniVoice có thể cần
# (speech tokenizer, vocoder, speaker encoder, whisper, v.v.)
# Kiểm tra trong config.json của OmniVoice để biết chính xác

# Thử tải bằng cách import và chạy thử
print("=== [2/4] Trigger download tất cả sub-models ===")
try:
    from omnivoice import OmniVoice
    import torch

    model = OmniVoice.from_pretrained(
        os.path.join(SAVE_DIR, "OmniVoice"),
        device_map="cpu",
        dtype=torch.float32,
    )
    print("Model loaded. Triggering voice cloning dependencies...")

    # Tạo 1 file audio giả để trigger download whisper/tokenizer
    import torchaudio
    dummy = torch.randn(1, 24000)  # 1s dummy audio
    torchaudio.save("_dummy_ref.wav", dummy, 24000)

    audio = model.generate(
        text="hello",
        ref_audio="_dummy_ref.wav",
    )
    print("Voice cloning dependencies downloaded.\n")
    os.remove("_dummy_ref.wav")

except Exception as e:
    print(f"Warning: {e}")
    print("Sub-models có thể đã được tải vào HuggingFace cache.\n")

# 3. Copy HuggingFace cache
print("=== [3/4] Thông tin HuggingFace cache ===")
hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(hf_cache):
    for d in os.listdir(hf_cache):
        if d.startswith("models--"):
            full = os.path.join(hf_cache, d)
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(full)
                for f in fns
            ) / (1024**3)
            print(f"  {d}: {size:.2f} GB")

print()
print("=== [4/4] Upload lên server ===")
print(f"scp -r {SAVE_DIR}/OmniVoice hungtt@lab-ai-01:~/GiangNVCode/TTS/omnivoice_test/OmniVoice")
print(f"scp -r {hf_cache}/models--* hungtt@lab-ai-01:~/.cache/huggingface/hub/")
print("\nHoặc chỉ copy HF cache nếu sub-models nằm trong cache.")
