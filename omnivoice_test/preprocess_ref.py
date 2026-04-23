"""
Preprocess reference audio for OmniVoice voice cloning.

Pipeline:
  1. Load + convert to mono
  2. Resample to 24kHz (match model output rate)
  3. Trim leading/trailing silence (VAD)
  4. Denoise (optional, nếu có noisereduce)
  5. Loudness normalize về -23 LUFS (optional, nếu có pyloudnorm)
  6. Peak limit tránh clipping
  7. Fade in/out 20ms

Input:  ref_audio/*.wav
Output: ref_audio_clean/*.wav

Usage:
    python preprocess_ref.py
    python preprocess_ref.py --no-denoise           # skip denoise
    python preprocess_ref.py --target-lufs -20      # custom loudness target
    python preprocess_ref.py --max-duration 15      # trim nếu dài hơn 15s
"""

import argparse
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

SCRIPT_DIR = Path(__file__).parent
REF_DIR = SCRIPT_DIR / "ref_audio"
OUT_DIR = SCRIPT_DIR / "ref_audio_clean"
TARGET_SR = 24000


def trim_silence(wav: torch.Tensor, sr: int, threshold_db: float = -40.0) -> torch.Tensor:
    """Trim silence ở đầu/cuối dựa trên energy threshold."""
    # wav: (1, T)
    abs_wav = wav.abs().squeeze(0)
    # Convert threshold dB -> amplitude
    threshold = 10 ** (threshold_db / 20)

    # Frame-based energy (20ms frames)
    frame_len = int(sr * 0.02)
    if abs_wav.shape[0] < frame_len * 2:
        return wav

    # Moving average
    kernel = torch.ones(1, 1, frame_len) / frame_len
    energy = torch.nn.functional.conv1d(
        abs_wav.view(1, 1, -1), kernel, padding=frame_len // 2
    ).squeeze()

    mask = energy > threshold
    if not mask.any():
        return wav

    nonzero = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    start = int(nonzero[0].item())
    end = int(nonzero[-1].item()) + frame_len

    # Pad ra 2 đầu 50ms để không cắt cụt
    pad = int(sr * 0.05)
    start = max(0, start - pad)
    end = min(wav.shape[-1], end + pad)

    return wav[:, start:end]


def apply_fade(wav: torch.Tensor, sr: int, fade_ms: int = 20) -> torch.Tensor:
    """Fade in/out tránh click đầu/cuối."""
    fade_len = int(sr * fade_ms / 1000)
    if wav.shape[-1] < fade_len * 2:
        return wav
    fade_in = torch.linspace(0.0, 1.0, fade_len)
    fade_out = torch.linspace(1.0, 0.0, fade_len)
    wav = wav.clone()
    wav[:, :fade_len] *= fade_in
    wav[:, -fade_len:] *= fade_out
    return wav


def denoise_audio(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, bool]:
    """Denoise dùng noisereduce. Trả về (wav, success)."""
    try:
        import noisereduce as nr
        import numpy as np
        arr = wav.squeeze(0).numpy()
        cleaned = nr.reduce_noise(y=arr, sr=sr, stationary=False, prop_decrease=0.85)
        return torch.from_numpy(cleaned.astype(np.float32)).unsqueeze(0), True
    except ImportError:
        return wav, False


def loudness_normalize(wav: torch.Tensor, sr: int, target_lufs: float) -> tuple[torch.Tensor, bool]:
    """Normalize loudness về target_lufs. Trả về (wav, success)."""
    try:
        import pyloudnorm as pyln
        import numpy as np
        arr = wav.squeeze(0).numpy().astype(np.float64)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(arr)
        if loudness == float("-inf"):
            return wav, False
        normalized = pyln.normalize.loudness(arr, loudness, target_lufs)
        return torch.from_numpy(normalized.astype("float32")).unsqueeze(0), True
    except ImportError:
        return wav, False


def peak_limit(wav: torch.Tensor, max_peak: float = 0.97) -> torch.Tensor:
    """Giảm volume nếu peak > max_peak để tránh clipping."""
    peak = wav.abs().max().item()
    if peak > max_peak:
        wav = wav * (max_peak / peak)
    return wav


def process_file(
    in_path: Path,
    out_path: Path,
    do_denoise: bool,
    target_lufs: float,
    max_duration: float | None,
) -> dict:
    """Process 1 file. Trả về dict thống kê."""
    info = {"file": in_path.name}

    # 1. Load
    wav, sr = torchaudio.load(str(in_path))
    info["orig_sr"] = sr
    info["orig_duration"] = round(wav.shape[-1] / sr, 2)
    info["orig_channels"] = wav.shape[0]

    # 2. Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 3. Resample
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
        sr = TARGET_SR

    # 4. Trim silence
    wav = trim_silence(wav, sr)
    info["after_trim_duration"] = round(wav.shape[-1] / sr, 2)

    # 5. Cắt nếu quá dài
    if max_duration is not None and wav.shape[-1] / sr > max_duration:
        wav = wav[:, : int(max_duration * sr)]
        info["truncated_to"] = max_duration

    # 6. Denoise
    if do_denoise:
        wav, ok = denoise_audio(wav, sr)
        info["denoised"] = ok
    else:
        info["denoised"] = False

    # 7. Loudness normalize
    wav, ok = loudness_normalize(wav, sr, target_lufs)
    info["loudness_normalized"] = ok
    info["target_lufs"] = target_lufs if ok else None

    # 8. Peak limit
    wav = peak_limit(wav)

    # 9. Fade
    wav = apply_fade(wav, sr, fade_ms=20)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav, sr)
    info["final_duration"] = round(wav.shape[-1] / sr, 2)
    info["out"] = str(out_path.relative_to(SCRIPT_DIR))

    return info


def main():
    parser = argparse.ArgumentParser(description="Preprocess reference audio for OmniVoice")
    parser.add_argument("--in-dir", type=str, default=str(REF_DIR), help="Input dir (default: ref_audio/)")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="Output dir (default: ref_audio_clean/)")
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoise step")
    parser.add_argument("--target-lufs", type=float, default=-23.0, help="Target loudness in LUFS (default: -23)")
    parser.add_argument("--max-duration", type=float, default=None, help="Truncate audio longer than this (seconds)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    wav_files = sorted(in_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files in {in_dir}")

    # Kiểm tra optional deps
    try:
        import noisereduce  # noqa
        has_nr = True
    except ImportError:
        has_nr = False
    try:
        import pyloudnorm  # noqa
        has_pyln = True
    except ImportError:
        has_pyln = False

    print(f"Input:  {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Target SR: {TARGET_SR} Hz")
    print(f"Target loudness: {args.target_lufs} LUFS" if has_pyln else "Target loudness: SKIP (install pyloudnorm)")
    print(f"Denoise: {'ON' if (not args.no_denoise and has_nr) else 'OFF' + ('' if has_nr else ' (install noisereduce)')}")
    if args.max_duration:
        print(f"Max duration: {args.max_duration}s")
    print(f"Files: {len(wav_files)}\n")

    if not has_nr or not has_pyln:
        missing = []
        if not has_nr:
            missing.append("noisereduce")
        if not has_pyln:
            missing.append("pyloudnorm")
        print(f"[HINT] Install missing deps for full pipeline:")
        print(f"       pip install {' '.join(missing)}\n")

    results = []
    for f in wav_files:
        out_path = out_dir / f.name
        print(f"Processing: {f.name}")
        info = process_file(
            f, out_path,
            do_denoise=(not args.no_denoise),
            target_lufs=args.target_lufs,
            max_duration=args.max_duration,
        )
        results.append(info)
        print(
            f"  {info['orig_sr']}Hz/{info['orig_channels']}ch/{info['orig_duration']}s"
            f" -> {TARGET_SR}Hz/1ch/{info['final_duration']}s"
            f" | denoise={info['denoised']} | loudnorm={info['loudness_normalized']}"
        )
        print(f"  -> {info['out']}\n")

    print("=" * 60)
    print(f"Done. Processed {len(results)} files -> {out_dir}")
    print("=" * 60)
    print("\nTo use cleaned ref audio in run_test.py:")
    print(f"  python run_test.py --quick  # edit REF_AUDIO_DIR in run_test.py to point to ref_audio_clean/")
    print("Or test 1 file:")
    print(f"  python run_test.py --ref_audio ref_audio_clean/{wav_files[0].name} --quick")


if __name__ == "__main__":
    main()
