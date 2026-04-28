"""
OmniVoice Vietnamese TTS - Test chính
Voice cloning + đo metrics (RTF, UTMOS, Speaker Similarity, WER)

Usage:
    python run_test.py                          # Test tất cả ref speakers
    python run_test.py --ref_audio ref_audio/female_01.wav  # Test 1 speaker
    python run_test.py --steps 16               # Chỉ test num_step=16
    python run_test.py --quick                  # Quick test: 3 câu, 1 step config
"""

import argparse
import json
import csv
import time
import os
from pathlib import Path

import torch
import torchaudio
from omnivoice import OmniVoice

# ============================================================
# Config
# ============================================================
SAMPLE_RATE = 24000
SCRIPT_DIR = Path(__file__).parent
TEST_SENTENCES = SCRIPT_DIR / "test_data" / "test_sentences.jsonl"
REF_AUDIO_DIR = SCRIPT_DIR / "ref_audio"
REF_AUDIO_CLEAN_DIR = SCRIPT_DIR / "ref_audio_clean"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "OmniVoice"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RESULTS_DIR = SCRIPT_DIR / "results"

REF_CONFIG = SCRIPT_DIR / "ref_text" / "ref_config.json"

DEFAULT_STEPS = [8, 16, 32]
QUICK_STEPS = [48]
QUICK_SENTENCES = ["bio_01"]  # IDs of sentences to test in quick mode


def load_ref_config(path: Path) -> dict:
    """Load ref_text mapping from ref_config.json."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_test_sentences(path: Path) -> list[dict]:
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))
    return sentences


def find_ref_audios(ref_dir: Path, single_ref: str | None = None) -> list[Path]:
    if single_ref:
        p = Path(single_ref)
        if not p.is_absolute():
            p = SCRIPT_DIR / p
        if not p.exists():
            raise FileNotFoundError(f"Reference audio not found: {p}")
        return [p]

    refs = sorted(ref_dir.glob("*.wav"))
    if not refs:
        raise FileNotFoundError(
            f"No .wav files found in {ref_dir}/\n"
            "Please add reference audio files (e.g., female_01.wav, male_01.wav)"
        )
    return refs


def normalize_commas(text: str, replace_with: str) -> str:
    """Thay dấu phẩy bằng chuỗi khác để model không đọc 'phẩy'.

    Các lựa chọn phổ biến cho replace_with:
    - ' . '   : period — pause rõ rệt nhưng vẫn nhỏ hơn dấu chấm cuối câu (có space trước/sau nên không tạo câu mới quá dài)
    - ' ; '   : semicolon — một số model coi như medium pause
    - ' — '   : em dash — medium pause, tự nhiên cho câu kể
    - '  '    : double space — dựa vào model tự phrasing
    - ' ... ' : ellipsis — pause dài hơn
    """
    # Chỉ thay trong câu, giữ nguyên dấu chấm cuối câu
    return text.replace(",", replace_with).replace(";", replace_with)


def split_into_chunks(text: str) -> list[dict]:
    """Tách text theo câu (. ! ?). Trả về list of dicts."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for sent in sentences:
        sent = sent.strip()
        if sent:
            chunks.append({"text": sent, "is_sentence_end": True})
    return chunks


def crossfade_concat(
    audios: list[torch.Tensor],
    sr: int,
    gap_ms_list: list[int],
    fade_ms: int = 30,
) -> torch.Tensor:
    """Nối audio với crossfade + gap silence khác nhau giữa từng cặp."""
    if not audios:
        return torch.zeros(1, 0)
    if len(audios) == 1:
        return audios[0]

    audios = [a if a.dim() == 2 else a.unsqueeze(0) for a in audios]
    fade_len = int(sr * fade_ms / 1000)

    result = audios[0].clone()
    for i, nxt in enumerate(audios[1:]):
        gap_len = int(sr * gap_ms_list[i] / 1000)
        silence = torch.zeros(1, gap_len)

        if result.shape[-1] >= fade_len:
            fade_out = torch.linspace(1.0, 0.0, fade_len)
            result[:, -fade_len:] = result[:, -fade_len:] * fade_out
        nxt = nxt.clone()
        if nxt.shape[-1] >= fade_len:
            fade_in = torch.linspace(0.0, 1.0, fade_len)
            nxt[:, :fade_len] = nxt[:, :fade_len] * fade_in
        result = torch.cat([result, silence, nxt], dim=-1)
    return result


def generate_chunked(
    model,
    text: str,
    ref_path: str,
    ref_text: str,
    num_step: int,
    speed: float = 1.0,
    sentence_gap_ms: int = 350,
    clause_gap_ms: int = 150,
    comma_replace: str = " . ",
    end_padding_ms: int = 100,
    text_suffix: str = "",
) -> tuple[torch.Tensor, list[str]]:
    """Chia text theo câu, generate từng câu, ghép lại."""
    text = normalize_commas(text, comma_replace)
    chunks = split_into_chunks(text)
    audios = []
    for i, ch in enumerate(chunks, 1):
        text_in = ch["text"]
        # Add suffix to last chunk to prevent cut-off
        if text_suffix and i == len(chunks):
            text_in = text_in + text_suffix
        print(f"    chunk {i}/{len(chunks)}: {text_in[:60]}{'...' if len(text_in) > 60 else ''}")
        kwargs = dict(text=text_in, ref_audio=ref_path, num_step=num_step, speed=speed)
        if ref_text:
            kwargs["ref_text"] = ref_text
        audio = model.generate(**kwargs)
        a = audio[0]
        if a.dim() == 1:
            a = a.unsqueeze(0)
        audios.append(a.cpu())

    # Gap sau chunk i: dài nếu hết câu, ngắn nếu hết vế (dấu phẩy)
    gap_list = [
        sentence_gap_ms if chunks[i]["is_sentence_end"] else clause_gap_ms
        for i in range(len(chunks) - 1)
    ]

    final = crossfade_concat(audios, SAMPLE_RATE, gap_ms_list=gap_list)

    # Add padding silence at the end to avoid cutting off last word
    if end_padding_ms > 0:
        pad_len = int(SAMPLE_RATE * end_padding_ms / 1000)
        final = torch.cat([final, torch.zeros(1, pad_len)], dim=-1)

    return final, [c["text"] for c in chunks]


def warm_up(model, ref_audio_path: str, ref_text: str):
    """Warm-up inference to initialize CUDA kernels."""
    print("Warm-up inference...", flush=True)
    model.generate(
        text="Xin chào",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )
    torch.cuda.synchronize()
    print("Warm-up done.\n")


def run_voice_cloning_test(
    model,
    sentences: list[dict],
    ref_audios: list[Path],
    steps_list: list[int],
    ref_config: dict = None,
    chunk: bool = False,
    speed: float = 1.0,
    sentence_gap_ms: int = 350,
    clause_gap_ms: int = 150,
    comma_replace: str = " . ",
    end_padding_ms: int = 100,
    text_suffix: str = "",
    repeat_short: int = 0,
    short_threshold: int = 2,
    no_trim: bool = False,
    repeat_separator: str = ". ",
) -> list[dict]:
    """Run voice cloning for all sentences x ref_audios x steps."""
    results = []
    total = len(sentences) * len(ref_audios) * len(steps_list)
    idx = 0

    if ref_config is None:
        ref_config = {}

    for ref_path in ref_audios:
        speaker_id = ref_path.stem
        speaker_out_dir = OUTPUT_DIR / speaker_id
        speaker_out_dir.mkdir(parents=True, exist_ok=True)

        # Lấy ref_text từ config
        ref_text = ref_config.get(speaker_id, {}).get("ref_text", "")
        if not ref_text:
            print(f"[WARN] No ref_text for {speaker_id} in ref_config.json. Voice quality may be lower.")

        # Tạo file README trong mỗi thư mục speaker để mô tả các file audio
        readme_lines = [f"# Speaker: {speaker_id}\n", f"Reference audio: {ref_path.name}\n", f"Ref text: {ref_text}\n\n"]

        for sent in sentences:
            for num_step in steps_list:
                idx += 1
                sent_id = sent["id"]
                text = sent["text"]
                group = sent.get("group", "unknown")
                note = sent.get("note", "")

                print(f"[{idx}/{total}] {speaker_id} | {sent_id} | steps={num_step}")
                print(f"  Text: {text[:60]}{'...' if len(text) > 60 else ''}")

                # Reset peak memory
                torch.cuda.reset_peak_memory_stats()

                # Generate
                torch.cuda.synchronize()
                t_start = time.perf_counter()

                # Normalize dấu phẩy trước khi đưa vào model
                text_for_model = normalize_commas(text, comma_replace)

                if chunk:
                    audio_tensor, sub_chunks = generate_chunked(
                        model, text, str(ref_path), ref_text, num_step,
                        speed=speed,
                        sentence_gap_ms=sentence_gap_ms,
                        clause_gap_ms=clause_gap_ms,
                        comma_replace=comma_replace,
                        end_padding_ms=end_padding_ms,
                        text_suffix=text_suffix,
                    )
                else:
                    # Add suffix to prevent cut-off
                    if text_suffix:
                        text_for_model = text_for_model + text_suffix

                    # Repeat short text to prevent cut-off, then trim
                    word_count = len(text_for_model.split())
                    repeat_count = 1
                    if repeat_short > 0 and word_count <= short_threshold:
                        repeat_count = repeat_short
                        text_for_model = repeat_separator.join([text_for_model.rstrip(".!?")] * repeat_count) + "."
                        print(f"  [repeat-short] '{text}' -> '{text_for_model}'")

                    generate_kwargs = dict(
                        text=text_for_model,
                        ref_audio=str(ref_path),
                        num_step=num_step,
                        speed=speed,
                    )
                    if ref_text:
                        generate_kwargs["ref_text"] = ref_text
                    audio = model.generate(**generate_kwargs)
                    audio_tensor = audio[0]
                    sub_chunks = None

                    # Trim to first 1/N if repeated (unless --no-trim)
                    if repeat_count > 1 and not no_trim:
                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        trim_len = audio_tensor.shape[-1] // repeat_count
                        audio_tensor = audio_tensor[:, :trim_len]
                        print(f"  [repeat-short] trimmed to {trim_len / SAMPLE_RATE:.2f}s")

                    # Add padding silence at the end
                    if end_padding_ms > 0:
                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        pad_len = int(SAMPLE_RATE * end_padding_ms / 1000)
                        audio_tensor = torch.cat([audio_tensor, torch.zeros(1, pad_len)], dim=-1)

                torch.cuda.synchronize()
                t_end = time.perf_counter()

                # Metrics
                inference_time = t_end - t_start
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                audio_duration = audio_tensor.shape[-1] / SAMPLE_RATE
                rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")
                gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

                # Save audio — tên file: [stt]_[id]_step[N].wav
                sent_idx = next(i for i, s in enumerate(sentences) if s["id"] == sent_id) + 1
                suffix = "_chunked" if chunk else ""
                out_path = speaker_out_dir / f"{sent_idx:02d}_{sent_id}_step{num_step}{suffix}.wav"
                chunk_info = ""
                if sub_chunks:
                    chunk_info = f" | chunks={len(sub_chunks)}"
                readme_lines.append(
                    f"- **{out_path.name}** | steps={num_step}{chunk_info} | "
                    f"[{group}] {note}\n  Text: \"{text}\"\n"
                )
                torchaudio.save(str(out_path), audio_tensor.cpu(), SAMPLE_RATE)

                result = {
                    "sentence_id": sent_id,
                    "group": group,
                    "text": text,
                    "speaker": speaker_id,
                    "num_step": num_step,
                    "inference_time_s": round(inference_time, 4),
                    "audio_duration_s": round(audio_duration, 4),
                    "rtf": round(rtf, 6),
                    "gpu_mem_mb": round(gpu_mem_mb, 1),
                    "output_path": str(out_path),
                }
                results.append(result)

                print(f"  -> {audio_duration:.2f}s audio | {inference_time:.2f}s infer | RTF={rtf:.4f} | GPU={gpu_mem_mb:.0f}MB")
                print()

        # Lưu README cho speaker
        with open(speaker_out_dir / "README.md", "w", encoding="utf-8") as f:
            f.writelines(readme_lines)

    return results


def compute_eval_metrics(results: list[dict], ref_audios: list[Path]):
    """Compute UTMOS, Speaker Similarity, WER using omnivoice[eval] tools."""
    print("\n" + "=" * 60)
    print("Computing evaluation metrics (UTMOS, Speaker Sim, WER)...")
    print("=" * 60 + "\n")

    # --- UTMOS ---
    try:
        from omnivoice.eval.utmos import UTMOSScore
        utmos_scorer = UTMOSScore(device="cuda:0")
        print("Computing UTMOS scores...")
        for r in results:
            wav, sr = torchaudio.load(r["output_path"])
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            score = utmos_scorer(wav.cuda())
            r["utmos"] = round(score.item(), 4)
            print(f"  {r['sentence_id']} ({r['speaker']}, step={r['num_step']}): UTMOS={r['utmos']:.3f}")
        print()
    except ImportError:
        print("[WARN] UTMOS not available. Install with: pip install omnivoice[eval]\n")
    except Exception as e:
        print(f"[WARN] UTMOS failed: {e}\n")

    # --- Speaker Similarity ---
    try:
        from omnivoice.eval.speaker_similarity import SpeakerSimilarity
        sim_scorer = SpeakerSimilarity(device="cuda:0")
        print("Computing Speaker Similarity...")
        ref_map = {p.stem: str(p) for p in ref_audios}
        for r in results:
            ref_path = ref_map.get(r["speaker"])
            if ref_path:
                score = sim_scorer(ref_path, r["output_path"])
                r["speaker_sim"] = round(score.item() if hasattr(score, "item") else float(score), 4)
                print(f"  {r['sentence_id']} ({r['speaker']}, step={r['num_step']}): Sim={r['speaker_sim']:.3f}")
        print()
    except ImportError:
        print("[WARN] SpeakerSimilarity not available. Install with: pip install omnivoice[eval]\n")
    except Exception as e:
        print(f"[WARN] Speaker similarity failed: {e}\n")

    # --- WER via Whisper ---
    try:
        import whisper
        from jiwer import wer as compute_wer
        print("Computing WER via Whisper transcription...")
        whisper_model = whisper.load_model("large-v3", device="cuda:0")
        for r in results:
            result = whisper_model.transcribe(r["output_path"], language="vi")
            hypothesis = result["text"].strip()
            reference = r["text"].strip()
            w = compute_wer(reference, hypothesis)
            r["wer"] = round(w, 4)
            r["whisper_transcript"] = hypothesis
            print(f"  {r['sentence_id']}: WER={w:.2%}")
            print(f"    REF: {reference[:80]}")
            print(f"    HYP: {hypothesis[:80]}")
        print()
    except ImportError:
        print("[WARN] Whisper/jiwer not available. Install: pip install openai-whisper jiwer\n")
    except Exception as e:
        print(f"[WARN] WER computation failed: {e}\n")

    return results


def save_results(results: list[dict]):
    """Save results to CSV and JSON summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- CSV ---
    csv_path = RESULTS_DIR / "metrics.csv"
    fieldnames = [
        "sentence_id", "group", "speaker", "num_step",
        "inference_time_s", "audio_duration_s", "rtf", "gpu_mem_mb",
        "utmos", "speaker_sim", "wer", "text", "whisper_transcript", "output_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved: {csv_path}")

    # --- JSON summary ---
    summary = {"total_samples": len(results), "by_group": {}, "by_steps": {}, "by_speaker": {}}

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # Group by group
    groups = set(r["group"] for r in results)
    for g in sorted(groups):
        subset = [r for r in results if r["group"] == g]
        summary["by_group"][g] = {
            "count": len(subset),
            "avg_rtf": avg([r["rtf"] for r in subset]),
            "avg_utmos": avg([r.get("utmos") for r in subset]),
            "avg_speaker_sim": avg([r.get("speaker_sim") for r in subset]),
            "avg_wer": avg([r.get("wer") for r in subset]),
        }

    # Group by num_step
    steps = set(r["num_step"] for r in results)
    for s in sorted(steps):
        subset = [r for r in results if r["num_step"] == s]
        summary["by_steps"][str(s)] = {
            "count": len(subset),
            "avg_rtf": avg([r["rtf"] for r in subset]),
            "avg_inference_time": avg([r["inference_time_s"] for r in subset]),
            "avg_utmos": avg([r.get("utmos") for r in subset]),
        }

    # Group by speaker
    speakers = set(r["speaker"] for r in results)
    for sp in sorted(speakers):
        subset = [r for r in results if r["speaker"] == sp]
        summary["by_speaker"][sp] = {
            "count": len(subset),
            "avg_rtf": avg([r["rtf"] for r in subset]),
            "avg_speaker_sim": avg([r.get("speaker_sim") for r in subset]),
            "avg_utmos": avg([r.get("utmos") for r in subset]),
        }

    json_path = RESULTS_DIR / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"JSON summary saved: {json_path}")

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY BY NUM_STEPS")
    print("=" * 70)
    print(f"{'Steps':>6} | {'Count':>5} | {'Avg RTF':>8} | {'Avg Time(s)':>11} | {'Avg UTMOS':>9}")
    print("-" * 70)
    for s in sorted(steps):
        d = summary["by_steps"][str(s)]
        utmos_str = f"{d['avg_utmos']:.3f}" if d["avg_utmos"] else "N/A"
        print(f"{s:>6} | {d['count']:>5} | {d['avg_rtf']:>8.4f} | {d['avg_inference_time']:>11.3f} | {utmos_str:>9}")

    print("\n" + "=" * 70)
    print("SUMMARY BY GROUP")
    print("=" * 70)
    print(f"{'Group':<15} | {'Count':>5} | {'Avg RTF':>8} | {'Avg UTMOS':>9} | {'Avg WER':>8} | {'Avg Sim':>8}")
    print("-" * 70)
    for g in sorted(groups):
        d = summary["by_group"][g]
        utmos_str = f"{d['avg_utmos']:.3f}" if d["avg_utmos"] else "N/A"
        wer_str = f"{d['avg_wer']:.2%}" if d["avg_wer"] is not None else "N/A"
        sim_str = f"{d['avg_speaker_sim']:.3f}" if d["avg_speaker_sim"] else "N/A"
        print(f"{g:<15} | {d['count']:>5} | {d['avg_rtf']:>8.4f} | {utmos_str:>9} | {wer_str:>8} | {sim_str:>8}")

    print()


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Vietnamese TTS Test")
    parser.add_argument("--ref_audio", type=str, default=None, help="Path to single ref audio (default: all in ref_audio/)")
    parser.add_argument("--test_file", type=str, default=None, help="Path to test sentences JSONL (default: test_data/test_sentences.jsonl)")
    parser.add_argument("--use-clean", action="store_true", help="Use ref_audio_clean/ instead of ref_audio/ (run preprocess_ref.py first)")
    parser.add_argument("--chunk", action="store_true", help="Split text by sentence and concat — better for long inputs")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0, slower: 0.85, faster: 1.15)")
    parser.add_argument("--sentence-gap-ms", type=int, default=350, help="Silence gap after . ! ? in ms (default: 350)")
    parser.add_argument("--clause-gap-ms", type=int, default=150, help="Silence gap after , ; in ms (default: 150)")
    parser.add_argument("--end-padding-ms", type=int, default=100, help="Silence padding at end of audio to avoid cut-off (default: 100)")
    parser.add_argument("--text-suffix", type=str, default="", help="Text to append to each sentence to prevent cut-off (e.g., '...' or ' ư')")
    parser.add_argument("--repeat-short", type=int, default=0, help="Repeat short text N times, gen, then trim to 1/N (e.g., 4 for 'bảy' -> 'bảy bảy bảy bảy')")
    parser.add_argument("--short-threshold", type=int, default=2, help="Max word count to consider 'short' for --repeat-short (default: 2)")
    parser.add_argument("--no-trim", action="store_true", help="Don't trim when using --repeat-short (keep full repeated audio)")
    parser.add_argument("--repeat-separator", type=str, default=". ", help="Separator between repeated words (default: '. ' for pause)")
    parser.add_argument(
        "--comma-replace", type=str, default=" . ",
        help="Thay dấu phẩy bằng chuỗi này (default: ' . '). Thử: ' ; ', ' — ', ' ... ', '  '",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=None, help="Diffusion steps to test (default: 8 16 32)")
    parser.add_argument("--quick", action="store_true", help="Quick test: 3 sentences, 1 step config")
    parser.add_argument("--no_eval", action="store_true", help="Skip UTMOS/WER/SpkSim evaluation")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help=f"Model path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    args = parser.parse_args()

    # Steps config
    if args.quick:
        steps_list = QUICK_STEPS
    elif args.steps:
        steps_list = args.steps
    else:
        steps_list = DEFAULT_STEPS

    # Load test sentences
    test_file = Path(args.test_file) if args.test_file else TEST_SENTENCES
    if not test_file.is_absolute():
        test_file = SCRIPT_DIR / test_file
    sentences = load_test_sentences(test_file)
    if args.quick:
        sentences = [s for s in sentences if s["id"] in QUICK_SENTENCES]
    print(f"Loaded {len(sentences)} test sentences")

    # Find ref audios
    ref_dir = REF_AUDIO_CLEAN_DIR if args.use_clean else REF_AUDIO_DIR
    if args.use_clean and not REF_AUDIO_CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"{REF_AUDIO_CLEAN_DIR} not found. Run: python preprocess_ref.py"
        )
    ref_audios = find_ref_audios(ref_dir, args.ref_audio)
    print(f"Ref audio dir: {ref_dir.name}")
    print(f"Reference speakers: {[p.stem for p in ref_audios]}")
    print(f"Diffusion steps: {steps_list}")
    print(f"Total tests: {len(sentences) * len(ref_audios) * len(steps_list)}")
    print()

    # Load model
    print("Loading OmniVoice model...")
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float16,
        local_files_only=True,
    )
    print(f"Model loaded on {args.device}\n")

    # Load ref config (ref_text cho từng speaker)
    ref_config = load_ref_config(REF_CONFIG)
    if ref_config:
        print(f"Loaded ref_text for: {list(ref_config.keys())}")

    # Warm-up
    first_speaker = ref_audios[0].stem
    first_ref_text = ref_config.get(first_speaker, {}).get("ref_text", "")
    warm_up(model, str(ref_audios[0]), first_ref_text)

    # Run voice cloning tests
    results = run_voice_cloning_test(
        model, sentences, ref_audios, steps_list, ref_config,
        chunk=args.chunk, speed=args.speed,
        sentence_gap_ms=args.sentence_gap_ms,
        clause_gap_ms=args.clause_gap_ms,
        comma_replace=args.comma_replace,
        end_padding_ms=args.end_padding_ms,
        text_suffix=args.text_suffix,
        repeat_short=args.repeat_short,
        short_threshold=args.short_threshold,
        no_trim=args.no_trim,
        repeat_separator=args.repeat_separator,
    )

    # Eval metrics
    if not args.no_eval:
        results = compute_eval_metrics(results, ref_audios)

    # Save
    save_results(results)

    print("Done! Check outputs/ for audio files and results/ for metrics.")


if __name__ == "__main__":
    main()
