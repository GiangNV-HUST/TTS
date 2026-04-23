from huggingface_hub import snapshot_download

print("Đang tải model OmniVoice (~2.45GB)...")
snapshot_download(
    repo_id="k2-fsa/OmniVoice",
    local_dir="C:/data/models/OmniVoice",
)
print("Tải hoàn tất! Model lưu tại: C:/data/models/OmniVoice")
print("Upload lên server: scp -r C:/data/models/OmniVoice hungtt@lab-ai-01:~/GiangNVCode/TTS/omnivoice_test/OmniVoice_model")