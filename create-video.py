#!/usr/bin/env python3
"""
Create a demo video from screenshots and optional audio narration.

Supports interleaved .png screenshots and audio in several formats:
  .mp3  — used directly as audio input
  .wav  — used directly as audio input (highest quality)
  .txt  — converted to WAV via Kokoro TTS, then used as audio input

Files are ordered by filename — use numeric prefixes (001-, 002-, etc.).
During audio clips, the most recent screenshot is held on screen.
Each screenshot without audio is displayed for FRAME_DURATION seconds.

When .txt narration files are present, the Kokoro TTS engine is used to
generate WAV audio. Generated files are cached in a generated-wav/
subfolder inside the screenshot directory and reused on subsequent runs
if the .txt source hasn't changed.

Usage: python create-video.py <base-folder> <subfolder-name>

Example folder structure:
  <base-folder>/screenshots/my-demo/
    001-welcome.png
    002-narration.mp3          (or .wav, or .txt)
    003-next-screen.png
    004-explanation.txt         (narration text → Kokoro TTS → WAV)
    005-final.png
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_DURATION = 2  # seconds per screenshot (images without audio)
KOKORO_VOICE = "bm_daniel"
SAMPLE_RATE = 24000  # Kokoro TTS output sample rate

# When True: use all available CPU cores (fastest, but loud fans).
# When False: cap threads at MAX_THREADS for a quieter, cooler run.
HIGH_PERFORMANCE = False
MAX_THREADS = 4     # max CPU threads used when HIGH_PERFORMANCE is False

# ANSI colours
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"  # no colour / reset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_readable_size(path: Path) -> str:
    """Return a human-readable file size string (e.g. '1.2M', '340K')."""
    size = path.stat().st_size
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024:
            if unit == "B":
                return f"{size}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}P"


def run_ffmpeg(args: list[str], *, capture_filter: str | None = None) -> None:
    """Run an ffmpeg command, optionally filtering stdout+stderr.

    When HIGH_PERFORMANCE is False the global ``-threads`` flag is prepended
    so every encode is capped at MAX_THREADS, keeping CPU load (and fan noise)
    low.  When HIGH_PERFORMANCE is True ffmpeg uses all available cores.
    """
    # -threads N must come before any input/output flags to act as a global cap.
    thread_args = [] if HIGH_PERFORMANCE else ["-threads", str(MAX_THREADS)]
    if capture_filter is not None:
        result = subprocess.run(
            ["ffmpeg"] + thread_args + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in result.stdout.splitlines():
            if any(tok in line for tok in capture_filter.split("|")):
                print(line)
    else:
        subprocess.run(
            ["ffmpeg"] + thread_args + args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )


def get_audio_duration(audio_path: Path) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value or value == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Kokoro TTS (in-process)
# ---------------------------------------------------------------------------

# Voice prefix -> language code mapping (mirrors kokoro-generate.py)
_LANG_CODE_MAP = {
    "a": "a",  # American English
    "b": "b",  # British English
    "e": "e",  # Spanish
    "f": "f",  # French
    "h": "h",  # Hindi
    "i": "i",  # Italian
    "j": "j",  # Japanese
    "p": "p",  # Portuguese
    "z": "z",  # Mandarin Chinese
}

# Lazy-loaded pipeline — only initialised when TTS is actually needed.
_kokoro_pipeline = None


def _detect_lang_code(voice: str) -> str:
    """Detect language code from voice name prefix."""
    if voice and len(voice) >= 2:
        first_char = voice[0]
        if first_char in _LANG_CODE_MAP:
            return _LANG_CODE_MAP[first_char]
    return "a"


def _ensure_kokoro_pipeline() -> None:
    """Lazily initialise the Kokoro TTS pipeline."""
    global _kokoro_pipeline  # noqa: PLW0603
    if _kokoro_pipeline is not None:
        return

    # Force fully offline mode — never contact HuggingFace Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"

    if not HIGH_PERFORMANCE:
        # Limit CPU parallelism for PyTorch / OpenMP / MKL so the TTS inference
        # stays within MAX_THREADS and doesn't spin up every core.
        # These env vars must be set before the native extensions are loaded,
        # which is why they live here rather than at module level.
        thread_str = str(MAX_THREADS)
        os.environ.setdefault("OMP_NUM_THREADS", thread_str)
        os.environ.setdefault("MKL_NUM_THREADS", thread_str)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", thread_str)
        try:
            import torch  # type: ignore[import-untyped]
            torch.set_num_threads(MAX_THREADS)
            torch.set_num_interop_threads(MAX_THREADS)
        except Exception:
            pass  # torch not available; Kokoro will still respect the env vars above

    try:
        from kokoro import KPipeline  # type: ignore[import-untyped]
    except ImportError:
        print(
            f"{RED}✗ kokoro package not installed in current environment{NC}",
            file=sys.stderr,
        )
        print(
            "Narration .txt files require the kokoro package.\n"
            "Install with: pip install 'kokoro>=0.9.4' 'misaki[en]' soundfile",
            file=sys.stderr,
        )
        sys.exit(1)

    lang_code = _detect_lang_code(KOKORO_VOICE)
    _kokoro_pipeline = KPipeline(lang_code=lang_code)


def run_kokoro_tts(input_txt: Path, output_wav: Path) -> None:
    """Generate a WAV file from a text file using Kokoro TTS (in-process)."""
    import numpy as np  # imported here so numpy isn't required when no TTS is used
    import soundfile as sf

    _ensure_kokoro_pipeline()

    text = input_txt.read_text(encoding="utf-8").strip()
    if not text:
        print(f"{RED}✗ Narration file is empty: {input_txt.name}{NC}", file=sys.stderr)
        sys.exit(1)

    audio_chunks: list = []
    for _gs, _ps, audio in _kokoro_pipeline(
        text, voice=KOKORO_VOICE, speed=1.0, split_pattern=r"\n+"
    ):
        audio_chunks.append(audio)

    if not audio_chunks:
        print(f"{RED}✗ No audio generated for: {input_txt.name}{NC}", file=sys.stderr)
        sys.exit(1)

    full_audio = np.concatenate(audio_chunks)
    sf.write(str(output_wav), full_audio, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Segment builders (ffmpeg)
# ---------------------------------------------------------------------------


def build_silent_segment(image: Path, segment: Path) -> None:
    """Create a video segment from a still image with silent audio."""
    run_ffmpeg([
        "-y",
        "-loop", "1", "-i", str(image),
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(FRAME_DURATION),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(segment),
    ])


def build_audio_segment(image: Path, audio: Path, segment: Path) -> None:
    """Create a video segment holding *image* on screen for the duration of *audio*."""
    run_ffmpeg([
        "-y",
        "-loop", "1", "-i", str(image),
        "-i", str(audio),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-ar", "44100", "-ac", "2",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(segment),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global MAX_THREADS  # noqa: PLW0603

    # --- Argument parsing ---------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Create a demo video from screenshots and optional audio narration.",
    )
    parser.add_argument("base_folder", help="Base project folder")
    parser.add_argument("subfolder_name", help="Subfolder name under screenshots/")
    parser.add_argument(
        "--no-intro",
        action="store_true",
        default=False,
        help="Skip the intro folder files even if an intro/ sibling folder exists",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=MAX_THREADS,
        metavar="N",
        help=f"Max CPU threads for ffmpeg and Kokoro TTS (default: {MAX_THREADS})",
    )
    args = parser.parse_args()

    if not HIGH_PERFORMANCE:
        # Apply the thread limit globally so run_ffmpeg and _ensure_kokoro_pipeline
        # both pick up the (potentially overridden) value.
        MAX_THREADS = args.threads

        # Lower process scheduling priority so the OS yields to other tasks first,
        # further reducing the chance of sustained full-load fan spin-up.
        try:
            os.nice(10)
        except (AttributeError, PermissionError):
            pass  # os.nice is not available on all platforms

    base_folder = args.base_folder
    subfolder_name = args.subfolder_name

    if not base_folder:
        print("Error: Base folder cannot be empty", file=sys.stderr)
        sys.exit(1)
    if not subfolder_name:
        print("Error: Subfolder name cannot be empty", file=sys.stderr)
        sys.exit(1)

    # --- Path setup ---------------------------------------------------------
    base_path = Path(base_folder)
    screenshot_dir = base_path / "screenshots" / subfolder_name
    intro_dir = screenshot_dir.parent / "intro"
    output_dir = base_path / "test-videos"
    mp4_file = output_dir / f"{subfolder_name}.mp4"
    gif_file = output_dir / f"{subfolder_name}.gif"
    palette_file = output_dir / f"{subfolder_name}-palette.png"
    segment_dir = output_dir / f"{subfolder_name}-segments"
    concat_list = segment_dir / "concat-list.txt"
    generated_wav_dir = screenshot_dir / "generated-wav"
    intro_generated_wav_dir = intro_dir / "generated-wav"

    print(f"{YELLOW}=== Create Video from Screenshots ==={NC}")
    print()

    # --- Dependency checks --------------------------------------------------
    if not shutil.which("ffmpeg"):
        print(f"{RED}✗ ffmpeg is not installed{NC}")
        print("Install with: sudo apt install ffmpeg")
        sys.exit(1)
    if not shutil.which("ffprobe"):
        print(f"{RED}✗ ffprobe is not installed (usually ships with ffmpeg){NC}")
        print("Install with: sudo apt install ffmpeg")
        sys.exit(1)

    # --- Verify screenshot directory ----------------------------------------
    if not screenshot_dir.is_dir():
        print(f"{RED}✗ Directory not found: {screenshot_dir}/{NC}")
        print("Run a demo test with screenshots enabled first.")
        sys.exit(1)

    # --- Collect and sort media files ---------------------------------------
    valid_extensions = {".png", ".mp3", ".wav", ".txt"}
    main_files: list[Path] = sorted(
        f
        for f in screenshot_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in valid_extensions
        and "generated-wav" not in f.parts
    )

    if not main_files:
        print(f"{RED}✗ No .png, .mp3, .wav, or .txt files found in {screenshot_dir}/{NC}")
        sys.exit(1)

    # --- Optionally prepend intro folder files (MP4 only) -------------------
    intro_files: list[Path] = []
    use_intro = not args.no_intro and intro_dir.is_dir()
    if use_intro:
        intro_files = sorted(
            f
            for f in intro_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in valid_extensions
            and "generated-wav" not in f.parts
        )
        if intro_files:
            print(f"Intro folder found: {intro_dir}/")
            print(f"  Prepending {len(intro_files)} intro file(s) to MP4 (skipped for GIF)")
            print()
        else:
            use_intro = False  # folder exists but is empty

    # For MP4 we combine intro + main files
    media_files: list[Path] = intro_files + main_files

    # --- Count by type ------------------------------------------------------
    image_count = sum(1 for f in media_files if f.suffix == ".png")
    audio_count = sum(1 for f in media_files if f.suffix in {".mp3", ".wav"})
    tts_count = sum(1 for f in media_files if f.suffix == ".txt")

    if image_count == 0:
        print(f"{RED}✗ No .png screenshots found in {screenshot_dir}/{NC}")
        print("At least one screenshot is required.")
        sys.exit(1)

    first_file = media_files[0]
    if first_file.suffix != ".png":
        print(f"{RED}✗ First file must be a .png screenshot, got: {first_file.name}{NC}")
        print("Audio clips need a preceding screenshot to display.")
        sys.exit(1)

    # --- Kokoro TTS: convert .txt → .wav ------------------------------------
    if tts_count > 0:
        print(
            f"Converting {tts_count} narration text file(s) to audio "
            f"via Kokoro TTS (voice: {KOKORO_VOICE})..."
        )

        converted_files: list[Path] = []
        for f in media_files:
            if f.suffix == ".txt":
                # Use the appropriate generated-wav/ cache dir for the source file
                if use_intro and f.parent == intro_dir:
                    wav_cache_dir = intro_generated_wav_dir
                else:
                    wav_cache_dir = generated_wav_dir
                wav_cache_dir.mkdir(parents=True, exist_ok=True)

                txt_basename = f.stem
                wav_path = wav_cache_dir / f"{txt_basename}.wav"

                if (
                    wav_path.exists()
                    and wav_path.stat().st_mtime > f.stat().st_mtime
                ):
                    print(
                        f"  [{txt_basename}.txt] {GREEN}cached{NC} → {txt_basename}.wav"
                    )
                else:
                    print(f"  [{txt_basename}.txt] generating WAV ... ", end="", flush=True)
                    run_kokoro_tts(f, wav_path)
                    print(f"{GREEN}✓{NC}")

                converted_files.append(wav_path)
                audio_count += 1
            else:
                converted_files.append(f)

        media_files = converted_files
        print()

    # --- Create output and segment directories ------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    if segment_dir.exists():
        shutil.rmtree(segment_dir)
    segment_dir.mkdir(parents=True, exist_ok=True)

    # Delete previous output files
    mp4_file.unlink(missing_ok=True)
    gif_file.unlink(missing_ok=True)

    # --- Report what we found -----------------------------------------------
    main_image_count = sum(1 for f in main_files if f.suffix == ".png")
    intro_image_count = sum(1 for f in intro_files if f.suffix == ".png")
    if use_intro and intro_files:
        print(
            f"Found {image_count} screenshot(s) ({intro_image_count} intro + "
            f"{main_image_count} main) and {audio_count} audio clip(s)"
        )
    else:
        print(f"Found {image_count} screenshot(s) and {audio_count} audio clip(s)")
    if tts_count > 0:
        print(f"  ({tts_count} narration text file(s) converted via Kokoro TTS)")
    print(f"Image frame duration: {FRAME_DURATION}s")
    if audio_count == 0:
        print(f"Total video length: {image_count * FRAME_DURATION}s")
    print()

    # --- Build video segments -----------------------------------------------
    segment_index = 0
    current_image: Path | None = None
    total_duration: float = 0.0

    with concat_list.open("w", encoding="utf-8") as cl:
        for i, f in enumerate(media_files):
            segment_file = segment_dir / f"segment-{segment_index:04d}.mp4"
            basename = f.name

            if f.suffix == ".png":
                current_image = f
                # If the very next file is audio, skip the silent segment so
                # the audio starts immediately over this image (no 2-second
                # silent hold before the narration begins).
                next_file = media_files[i + 1] if i + 1 < len(media_files) else None
                if next_file is not None and next_file.suffix in {".mp3", ".wav"}:
                    print(f"  [{basename}] image (held for next audio, no silent segment)")
                    continue  # don't build a segment; audio pass will pick up current_image

                print(f"  [{basename}] image, {FRAME_DURATION}s ... ", end="", flush=True)
                build_silent_segment(f, segment_file)
                print(f"{GREEN}✓{NC}")
                total_duration += FRAME_DURATION

            elif f.suffix in {".mp3", ".wav"}:
                audio_duration = get_audio_duration(f)
                if audio_duration is None:
                    print(f"  [{basename}] {RED}✗ could not detect duration, skipping{NC}")
                    continue

                assert current_image is not None
                print(
                    f"  [{basename}] audio, {audio_duration}s "
                    f"(holding {current_image.name}) ... ",
                    end="",
                    flush=True,
                )
                build_audio_segment(current_image, f, segment_file)
                print(f"{GREEN}✓{NC}")
                total_duration += audio_duration

            cl.write(f"file '{segment_file.resolve()}'\n")
            segment_index += 1

    print()
    print(f"Generated {segment_index} segments, total duration: ~{round(total_duration)}s")
    print()

    # --- Concatenate segments into final MP4 --------------------------------
    print("Concatenating into final MP4...")
    run_ffmpeg(
        [
            "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-movflags", "+faststart",
            str(mp4_file),
        ],
        capture_filter="frame=|Duration:|Output|error",
    )

    if not mp4_file.exists():
        print(f"{RED}✗ Failed to create MP4 video{NC}")
        sys.exit(1)

    mp4_size = human_readable_size(mp4_file)
    print(f"{GREEN}✓ MP4 created successfully{NC} ({mp4_size})")
    print()

    # --- Create GIF (images only, no audio support in GIF) ------------------
    # Collect sorted PNG paths for the GIF (glob pattern isn't reliable with
    # pathlib across different filenames, so we feed them via concat demuxer).
    png_files = sorted(screenshot_dir.glob("*.png"))

    print("Generating GIF palette (images only, audio skipped)...")
    run_ffmpeg(
        [
            "-y",
            "-framerate", f"1/{FRAME_DURATION}",
            "-pattern_type", "glob",
            "-i", str(screenshot_dir / "*.png"),
            "-vf", "palettegen",
            str(palette_file),
        ],
        capture_filter="frame=|Duration:|Output|error",
    )

    if not palette_file.exists():
        print(f"{RED}✗ Failed to generate palette{NC}")
        sys.exit(1)

    print("Creating GIF...")
    run_ffmpeg(
        [
            "-y",
            "-framerate", f"1/{FRAME_DURATION}",
            "-pattern_type", "glob",
            "-i", str(screenshot_dir / "*.png"),
            "-i", str(palette_file),
            "-lavfi", "paletteuse",
            str(gif_file),
        ],
        capture_filter="frame=|Duration:|Output|error",
    )

    # Clean up palette file
    palette_file.unlink(missing_ok=True)

    if not gif_file.exists():
        print(f"{RED}✗ Failed to create GIF{NC}")
        sys.exit(1)

    gif_size = human_readable_size(gif_file)
    print(f"{GREEN}✓ GIF created successfully{NC} ({gif_size})")

    # --- Clean up temp segments ---------------------------------------------
    shutil.rmtree(segment_dir)

    # --- Summary ------------------------------------------------------------
    print()
    print(f"{GREEN}=== Done! ==={NC}")
    print(f"  MP4: {mp4_file} ({mp4_size})")
    print(f"  GIF: {gif_file} ({gif_size})")
    if audio_count > 0:
        print(f"  Audio: {audio_count} clip(s) included in MP4 (not in GIF)")
        if tts_count > 0:
            print(
                f"  TTS:   {tts_count} narration(s) generated via Kokoro "
                f"(cached in generated-wav/)"
            )
    print(f"  Subfolder: {subfolder_name}")
    if use_intro and intro_files:
        print(f"  Intro:     {len(intro_files)} file(s) prepended from {intro_dir}/")
    print()
    print(f"To view MP4: mpv {mp4_file}")
    print(f"To view GIF: mpv {gif_file}")


if __name__ == "__main__":
    main()
