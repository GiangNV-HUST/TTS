"""
Extract best clip from long audio for voice cloning reference.

Criteria:
  - Continuous speech (no long pauses)
  - Low noise / high SNR
  - Stable energy (no sudden volume changes)
  - Duration: 5-15 seconds

Usage:
    python extract_best_clip.py input.wav
    python extract_best_clip.py input.wav --output ref_audio/speaker.wav
    python extract_best_clip.py input.wav --min-duration 5 --max-duration 10
"""

import argparse
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

TARGET_SR = 24000


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio, convert to mono, resample to TARGET_SR."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0), TARGET_SR


def compute_energy(wav: torch.Tensor, frame_ms: int = 20, sr: int = TARGET_SR) -> torch.Tensor:
    """Compute frame-level energy."""
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2

    # Pad to make full frames
    pad_len = (frame_len - wav.shape[0] % frame_len) % frame_len
    wav_padded = torch.nn.functional.pad(wav, (0, pad_len))

    # Unfold into frames
    frames = wav_padded.unfold(0, frame_len, hop)
    energy = (frames ** 2).mean(dim=-1)
    return energy


def find_speech_segments(
    wav: torch.Tensor,
    sr: int,
    energy_threshold_db: float = -35.0,
    min_speech_ms: int = 500,
    max_pause_ms: int = 300,
) -> list[tuple[int, int]]:
    """Find continuous speech segments using energy-based VAD."""
    energy = compute_energy(wav, frame_ms=20, sr=sr)

    # Convert threshold
    threshold = 10 ** (energy_threshold_db / 10)

    # Binary mask: speech vs silence
    is_speech = energy > threshold

    # Frame to sample conversion
    frame_ms = 20
    hop_ms = 10

    # Find speech regions
    segments = []
    in_speech = False
    start_frame = 0
    pause_frames = 0
    max_pause_frames = max_pause_ms // hop_ms
    min_speech_frames = min_speech_ms // hop_ms

    for i, s in enumerate(is_speech):
        if s:
            if not in_speech:
                start_frame = i
                in_speech = True
            pause_frames = 0
        else:
            if in_speech:
                pause_frames += 1
                if pause_frames > max_pause_frames:
                    # End of segment
                    end_frame = i - pause_frames
                    if end_frame - start_frame >= min_speech_frames:
                        start_sample = int(start_frame * hop_ms * sr / 1000)
                        end_sample = int(end_frame * hop_ms * sr / 1000)
                        segments.append((start_sample, end_sample))
                    in_speech = False
                    pause_frames = 0

    # Handle last segment
    if in_speech:
        end_frame = len(is_speech) - pause_frames
        if end_frame - start_frame >= min_speech_frames:
            start_sample = int(start_frame * hop_ms * sr / 1000)
            end_sample = min(int(end_frame * hop_ms * sr / 1000), len(wav))
            segments.append((start_sample, end_sample))

    return segments


def estimate_snr(wav: torch.Tensor, sr: int) -> float:
    """Estimate SNR by comparing speech energy to silence energy."""
    energy = compute_energy(wav, frame_ms=20, sr=sr)

    # Sort energy to find noise floor (bottom 10%)
    sorted_energy, _ = torch.sort(energy)
    noise_floor = sorted_energy[:max(1, len(sorted_energy) // 10)].mean()

    # Signal energy (top 50%)
    signal_energy = sorted_energy[len(sorted_energy) // 2:].mean()

    if noise_floor < 1e-10:
        noise_floor = torch.tensor(1e-10)

    snr = 10 * torch.log10(signal_energy / noise_floor)
    return snr.item()


def compute_energy_stability(wav: torch.Tensor, sr: int) -> float:
    """Compute energy stability (lower variance = more stable)."""
    energy = compute_energy(wav, frame_ms=50, sr=sr)
    # Use log energy for stability measure
    log_energy = torch.log(energy + 1e-10)
    # Return negative variance (higher = more stable)
    return -log_energy.var().item()


def score_segment(wav: torch.Tensor, sr: int, min_dur: float, max_dur: float) -> float:
    """Score a segment based on quality metrics."""
    duration = len(wav) / sr

    # Duration penalty
    if duration < min_dur:
        dur_score = duration / min_dur
    elif duration > max_dur:
        dur_score = max_dur / duration
    else:
        # Prefer middle of range
        ideal = (min_dur + max_dur) / 2
        dur_score = 1.0 - abs(duration - ideal) / ideal * 0.2

    # SNR score (normalize to 0-1, assuming SNR 10-40 dB range)
    snr = estimate_snr(wav, sr)
    snr_score = max(0, min(1, (snr - 10) / 30))

    # Stability score (normalize)
    stability = compute_energy_stability(wav, sr)
    stability_score = 1.0 / (1.0 + abs(stability))

    # Combined score
    score = dur_score * 0.3 + snr_score * 0.5 + stability_score * 0.2

    return score, {
        "duration": round(duration, 2),
        "snr_db": round(snr, 1),
        "dur_score": round(dur_score, 3),
        "snr_score": round(snr_score, 3),
        "stability_score": round(stability_score, 3),
        "total_score": round(score, 3),
    }


def extract_best_clip(
    wav: torch.Tensor,
    sr: int,
    min_duration: float = 5.0,
    max_duration: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """Extract the best clip from audio."""

    # Find speech segments
    segments = find_speech_segments(wav, sr)

    if not segments:
        print("[WARN] No speech segments found. Using full audio.")
        segments = [(0, len(wav))]

    print(f"Found {len(segments)} speech segments")

    # Generate candidate clips
    candidates = []

    for start, end in segments:
        seg_duration = (end - start) / sr

        if seg_duration < min_duration:
            # Segment too short, skip
            continue
        elif seg_duration <= max_duration:
            # Segment fits, use as is
            candidates.append((start, end))
        else:
            # Segment too long, create sliding windows
            clip_len = int(max_duration * sr)
            step = int(sr * 1.0)  # 1 second step
            for clip_start in range(start, end - clip_len + 1, step):
                candidates.append((clip_start, clip_start + clip_len))

    if not candidates:
        # Fallback: use longest segment, truncated
        longest = max(segments, key=lambda x: x[1] - x[0])
        clip_len = int(max_duration * sr)
        candidates = [(longest[0], min(longest[0] + clip_len, longest[1]))]

    print(f"Evaluating {len(candidates)} candidate clips...")

    # Score each candidate
    best_score = -float("inf")
    best_clip = None
    best_info = None

    for i, (start, end) in enumerate(candidates):
        clip = wav[start:end]
        score, info = score_segment(clip, sr, min_duration, max_duration)
        info["start_sec"] = round(start / sr, 2)
        info["end_sec"] = round(end / sr, 2)

        if score > best_score:
            best_score = score
            best_clip = clip
            best_info = info

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(candidates)} clips...")

    return best_clip, best_info


def main():
    parser = argparse.ArgumentParser(description="Extract best clip for voice cloning")
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file (default: input_clip.wav)")
    parser.add_argument("--min-duration", type=float, default=7.0, help="Minimum clip duration (default: 7s)")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Maximum clip duration (default: 10s)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_clip.wav"

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Target duration: {args.min_duration}-{args.max_duration}s")
    print()

    # Load audio
    print("Loading audio...")
    wav, sr = load_audio(str(input_path))
    total_duration = len(wav) / sr
    print(f"Total duration: {total_duration:.1f}s")
    print()

    # Extract best clip
    clip, info = extract_best_clip(wav, sr, args.min_duration, args.max_duration)

    print()
    print("=" * 50)
    print("Best clip:")
    print(f"  Time: {info['start_sec']}s - {info['end_sec']}s")
    print(f"  Duration: {info['duration']}s")
    print(f"  SNR: {info['snr_db']} dB")
    print(f"  Score: {info['total_score']}")
    print("=" * 50)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), clip.unsqueeze(0), sr)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
