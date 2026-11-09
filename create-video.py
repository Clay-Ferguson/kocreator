#!/usr/bin/env python3
"""
Create a demo video from screenshots and optional audio narration.

Supports interleaved .png screenshots and audio in several formats:
  .mp3  — used directly as audio input
  .wav  — used directly as audio input (highest quality)
  .txt  — converted to WAV by the configured TTS plugin, then used as audio input

Files are ordered by filename — use numeric prefixes (001-, 002-, etc.).
During audio clips, the most recent screenshot is held on screen.
Each screenshot without audio is displayed for FRAME_DURATION seconds.

When .txt narration files are present, speech comes from a TTS plugin: a
folder (normally a sibling such as ../kocreator-kokoro) holding a tts.sh
script with a fixed interface — see "Writing a TTS plugin" in README.md.  The
plugin is chosen by kocreator.toml, the KOCREATOR_TTS_PLUGIN environment
variable, or --plugin.  Generated files are cached in a generated-wav/
subfolder inside the screenshot directory, each with a .json manifest and a
.log of the plugin's output beside it, and reused on subsequent runs as long
as the narration text, the seed and the plugin's reported settings are
unchanged.  An optional <name>.seed file next to a narration .txt picks a
different take (seed) for that one clip, on engines that support seeds.

The MP4 is encoded for maximum browser compatibility: H.264 High profile
level 4.0 (8-bit yuv420p), one fixed resolution for the whole file, square
pixels, constant frame rate, BT.709 colour tags, AAC-LC audio, and a faststart
container. Screenshots of differing sizes are letterboxed onto a single canvas
rather than switching resolution mid-stream — see the encoding constants below
and the README for why that matters.

Usage: python create-video.py <base-folder> <subfolder-name> [--plugin DIR]

Example folder structure:
  <base-folder>/screenshots/my-demo/
    001-welcome.png
    002-narration.mp3          (or .wav, or .txt)
    003-next-screen.png
    004-explanation.txt         (narration text → TTS plugin → WAV)
    004-explanation.seed        (optional: re-roll this clip with another seed)
    005-final.png
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_DURATION = 2  # seconds per screenshot (images without audio)

# ---------------------------------------------------------------------------
# Text-to-speech (plugin)
# ---------------------------------------------------------------------------
# Narration is spoken by a TTS plugin folder containing an executable tts.sh
# (see "Writing a TTS plugin" in README.md).  Which folder is used comes from,
# in order of precedence: the --plugin flag, the KOCREATOR_TTS_PLUGIN
# environment variable, or tts_plugin in kocreator.toml.  A relative path is
# resolved against this project's folder, so "../kocreator-kokoro" means the
# sibling folder.  Voice settings live in each plugin's own tts.sh.
KOCREATOR_DIR = Path(__file__).resolve().parent
CONFIG_FILE = KOCREATOR_DIR / "kocreator.toml"
PLUGIN_ENV_VAR = "KOCREATOR_TTS_PLUGIN"
PLUGIN_SCRIPT = "tts.sh"
DONE_MARKER = "KOCREATOR_DONE"  # the plugin prints "KOCREATOR_DONE <n>" after each clip

# ---------------------------------------------------------------------------
# Web-compatibility encoding profile
# ---------------------------------------------------------------------------
# Every segment MUST be encoded with byte-identical stream parameters, because
# the final MP4 is produced by concatenating segments with "-c copy".  An MP4
# stores the video resolution once, in the container header (avcC/tkhd), taken
# from the *first* segment.  If a later segment has a different resolution the
# file still plays under permissive software decoders (ffmpeg, mpv, Chrome on
# Linux) but hardware decoders — Safari, Chrome/Edge on Windows, Firefox on
# most platforms — stop producing frames at the switch.  The classic symptom is
# "audio plays, video is blank".  Screenshots routinely differ in size (an
# intro splash vs. app captures), so all frames are letterboxed onto one fixed
# canvas below.
TARGET_FPS = 30          # constant frame rate for every segment
MAX_WIDTH = 1920         # canvas is capped here so H.264 level 4.0 always holds
MAX_HEIGHT = 1080
H264_PROFILE = "high"    # High 8-bit 4:2:0 — universally decodable in browsers
H264_LEVEL = "4.0"       # 4.0 covers up to 1920x1080@30 on every shipping decoder
PAD_COLOR = "black"      # letterbox/pillarbox fill
# AAC-LC at 44.1 kHz is the universal web audio baseline.  Mono is just as
# universally decodable as stereo, and the narration is mono to begin with
# (every TTS engine we use emits a single channel), so encoding stereo only spent
# bits on a duplicate channel.  64 kbps mono is ample for speech.
AUDIO_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_BITRATE = "64k"
MOV_TIMESCALE = 90000    # standard MP4 video timescale
# Every segment already opens with a keyframe, so each screenshot change is
# seekable regardless.  Extra keyframes *within* a held still image are pure
# waste — at CRF 18 each one costs a full intra-coded screenshot — so the GOP
# is long.  Browsers still seek accurately: they decode forward from the
# preceding keyframe, which is instant across near-empty P-frames.
GOP_SECONDS = 10

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
# TTS via plugin (subprocess)
# ---------------------------------------------------------------------------
# A plugin is any folder holding an executable tts.sh that answers:
#   tts.sh check               exit 0 when ready, else explain on stderr
#   tts.sh settings            print a JSON object of audio-shaping settings
#   tts.sh speak <jobs.tsv>    speak every job (input, output, seed per line) in one
#                              run, printing "KOCREATOR_DONE <n>" after each one
# Speaking is batch-only: a single clip is simply a batch of one.
# See "Writing a TTS plugin" in README.md for the full contract.

TtsSettings = dict[str, object]


def resolve_plugin_dir(cli_value: str | None) -> Path:
    """Return the TTS plugin folder: --plugin, else $KOCREATOR_TTS_PLUGIN, else kocreator.toml."""
    if cli_value:
        raw, source = cli_value, "--plugin"
    elif os.environ.get(PLUGIN_ENV_VAR):
        raw, source = os.environ[PLUGIN_ENV_VAR], PLUGIN_ENV_VAR
    else:
        source = str(CONFIG_FILE)
        try:
            with CONFIG_FILE.open("rb") as fh:
                raw = tomllib.load(fh).get("tts_plugin", "")
        except FileNotFoundError:
            raw = ""
        except tomllib.TOMLDecodeError as exc:
            print(f"{RED}✗ Could not parse {CONFIG_FILE}: {exc}{NC}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(raw, str) or not raw.strip():
            print(
                f"{RED}✗ No TTS plugin configured.{NC} Set tts_plugin in {CONFIG_FILE}, "
                f"export {PLUGIN_ENV_VAR}, or pass --plugin DIR.",
                file=sys.stderr,
            )
            sys.exit(1)

    path = Path(os.path.expanduser(raw.strip()))
    if not path.is_absolute():
        path = KOCREATOR_DIR / path
    path = path.resolve()

    script = path / PLUGIN_SCRIPT
    if not script.is_file() or not os.access(script, os.X_OK):
        print(
            f"{RED}✗ TTS plugin not found: {script}{NC} (from {source})\n"
            "A plugin folder must contain an executable tts.sh. Check out one of\n"
            "kocreator-kokoro, Buffy or kocreator-qwen next to this\n"
            "project and point kocreator.toml at it.",
            file=sys.stderr,
        )
        sys.exit(1)
    return path


def tts_env() -> dict[str, str]:
    """Environment for every plugin subprocess."""
    env = os.environ.copy()
    # uv-env.sh exports UV_PROJECT_ENVIRONMENT pointing at *this* project's venv,
    # and `uv run` sets VIRTUAL_ENV to it.  Inherited by a plugin's own `uv run`,
    # UV_PROJECT_ENVIRONMENT would make uv install the plugin's dependencies into
    # our venv, so both are dropped and each plugin uses its own environment.
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env.pop("VIRTUAL_ENV", None)

    # Always tell the plugin the thread budget, for engines (like torch) that
    # need an explicit call rather than the env vars below.
    env["KOCREATOR_THREADS"] = str(MAX_THREADS if not HIGH_PERFORMANCE else os.cpu_count() or 1)

    if not HIGH_PERFORMANCE:
        # Cap the BLAS/OpenMP thread pools in the child so inference stays
        # within MAX_THREADS and doesn't spin up every core.  (The os.nice()
        # priority set in main() is inherited automatically.)
        thread_str = str(MAX_THREADS)
        env.setdefault("OMP_NUM_THREADS", thread_str)
        env.setdefault("MKL_NUM_THREADS", thread_str)
        env.setdefault("OPENBLAS_NUM_THREADS", thread_str)
    return env


def run_plugin(plugin_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``tts.sh <args>`` with stdout+stderr merged into one captured string."""
    return subprocess.run(
        [str(plugin_dir / PLUGIN_SCRIPT), *args],
        env=tts_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def check_plugin(plugin_dir: Path) -> None:
    """Exit with the plugin's own explanation unless ``tts.sh check`` succeeds."""
    result = run_plugin(plugin_dir, "check")
    if result.returncode != 0:
        print(f"{RED}✗ TTS plugin is not ready: {plugin_dir}{NC}", file=sys.stderr)
        print(result.stdout.rstrip(), file=sys.stderr)
        sys.exit(1)


def plugin_settings(plugin_dir: Path) -> TtsSettings:
    """The JSON object ``tts.sh settings`` prints: everything that shapes its audio."""
    result = subprocess.run(
        [str(plugin_dir / PLUGIN_SCRIPT), "settings"],
        env=tts_env(),
        capture_output=True,
        text=True,
    )
    try:
        settings = json.loads(result.stdout) if result.returncode == 0 else None
    except ValueError:
        settings = None
    if not isinstance(settings, dict):
        print(
            f"{RED}✗ {plugin_dir / PLUGIN_SCRIPT} settings did not print a JSON object "
            f"(exit code {result.returncode}).{NC}",
            file=sys.stderr,
        )
        print((result.stdout + result.stderr).rstrip(), file=sys.stderr)
        sys.exit(1)
    return settings


def clip_seed(input_txt: Path) -> int | None:
    """The seed (take) for one narration: its <name>.seed sidecar, else None.

    A sidecar lets one glitchy clip be re-rolled without touching the others:
    put a different whole number in ``004-narration.seed`` next to
    ``004-narration.txt``.  The seed is part of the cache key, so adding,
    changing or removing the sidecar regenerates just that clip.  Without a
    sidecar no seed is passed and the plugin uses its own default.
    """
    seed_file = input_txt.with_suffix(".seed")
    if not seed_file.is_file():
        return None
    raw = seed_file.read_text(encoding="utf-8").strip()
    try:
        return int(raw)
    except ValueError:
        print(
            f"{RED}✗ {seed_file} must contain a whole number (the seed), "
            f"got: {raw!r}{NC}",
            file=sys.stderr,
        )
        sys.exit(1)


def tts_manifest(text: str, seed: int | None, settings: TtsSettings) -> TtsSettings:
    """The cache key for one narration: its text, its seed, and the plugin's settings."""
    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "seed": seed,
        "settings": settings,
    }


def is_tts_cached(wav_path: Path, manifest_data: TtsSettings) -> bool:
    """True when *wav_path* exists and its .json manifest records exactly *manifest_data*.

    A WAV without a manifest (one from some other tool, or from a run that was
    interrupted) never counts as cached.
    """
    manifest = wav_path.with_suffix(".json")
    if not wav_path.is_file() or not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")) == manifest_data
    except (OSError, ValueError):
        return False


@dataclass
class TtsJob:
    """One narration to speak: its text file, its final WAV, seed and cache manifest."""

    input_txt: Path
    output_wav: Path
    seed: int | None
    manifest_data: TtsSettings

    @property
    def partial_wav(self) -> Path:
        """Where the plugin writes; renamed to output_wav only once the clip is done."""
        return self.output_wav.with_name(f"{self.output_wav.stem}.partial.wav")

    @property
    def label(self) -> str:
        seed_note = f" (seed {self.seed})" if self.seed is not None else ""
        return f"[{self.input_txt.name}]{seed_note}"


def write_jobs_file(jobs: list[TtsJob]) -> Path:
    """Write the batch as a TSV file (input, output, seed) and return its path.

    One line per job, three tab-separated fields: the absolute narration .txt
    path, the absolute .partial.wav path to write, and the seed (empty when
    the clip has no .seed sidecar).  Paths are absolute because plugins run
    their engines from their own folders.
    """
    lines = []
    for job in jobs:
        fields = [
            str(job.input_txt.resolve()),
            str(job.partial_wav.resolve()),
            "" if job.seed is None else str(job.seed),
        ]
        if any(c in field for field in fields for c in "\t\n\r"):
            print(
                f"{RED}✗ Paths may not contain tabs or newlines: {job.input_txt}{NC}",
                file=sys.stderr,
            )
            sys.exit(1)
        lines.append("\t".join(fields) + "\n")

    fd, name = tempfile.mkstemp(prefix="kocreator-tts-", suffix=".tsv")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    return Path(name)


def finish_job(job: TtsJob, log_text: str, started: float) -> None:
    """Move a finished clip into place, then record its log and (last) its manifest."""
    if not job.partial_wav.is_file():
        fail_batch(job, log_text, f"the plugin reported it done but wrote no {job.partial_wav.name}")
    job.partial_wav.replace(job.output_wav)
    job.output_wav.with_suffix(".log").write_text(log_text, encoding="utf-8")
    job.output_wav.with_suffix(".json").write_text(
        json.dumps(job.manifest_data, indent=2) + "\n", encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    duration = get_audio_duration(job.output_wav)
    audio_note = f"audio {duration:.1f}s, " if duration is not None else ""
    print(f"{GREEN}✓{NC} ({audio_note}took {elapsed:.1f}s)")


def fail_batch(job: TtsJob, log_text: str, reason: str) -> NoReturn:
    """Report a failed batch, showing the plugin's output for the clip that failed."""
    print(f"{RED}✗{NC}")
    print(
        f"{RED}✗ TTS plugin failed for {job.input_txt.name}: {reason}. Its output:{NC}",
        file=sys.stderr,
    )
    print(log_text.rstrip(), file=sys.stderr)
    job.partial_wav.unlink(missing_ok=True)
    sys.exit(1)


def run_tts_batch(plugin_dir: Path, jobs: list[TtsJob]) -> None:
    """Speak every job with one ``tts.sh speak <jobs.tsv>`` call.

    The plugin loads its engine once, speaks the jobs in order, and after
    writing each one prints a line ``KOCREATOR_DONE <n>`` (n = 1-based job
    number).  Each marker finishes that clip on the spot: its .partial.wav is
    renamed into place, the plugin output since the previous marker becomes
    its ``<name>.log``, and its manifest is written last.  So an interrupted or
    failed batch keeps every clip finished before the failure, and never
    leaves behind a WAV that looks cached.
    """
    for job in jobs:
        job.output_wav.with_suffix(".json").unlink(missing_ok=True)
        job.output_wav.with_suffix(".log").unlink(missing_ok=True)
        job.partial_wav.unlink(missing_ok=True)

    jobs_file = write_jobs_file(jobs)
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            [str(plugin_dir / PLUGIN_SCRIPT), "speak", str(jobs_file)],
            env=tts_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None

        done = 0                 # jobs finished so far
        log_lines: list[str] = []  # plugin output for the job in progress
        print(f"  {jobs[0].label} generating WAV ... ", end="", flush=True)
        started = time.perf_counter()

        for line in proc.stdout:
            if not line.startswith(DONE_MARKER):
                log_lines.append(line)
                continue
            parts = line.split()
            if len(parts) != 2 or parts[1] != str(done + 1):
                fail_batch(
                    jobs[min(done, len(jobs) - 1)],
                    "".join(log_lines) + line,
                    f"expected '{DONE_MARKER} {done + 1}', got {line.strip()!r}",
                )
            finish_job(jobs[done], "".join(log_lines), started)
            done += 1
            log_lines = []
            if done < len(jobs):
                print(f"  {jobs[done].label} generating WAV ... ", end="", flush=True)
                started = time.perf_counter()

        returncode = proc.wait()
        if done < len(jobs):
            fail_batch(jobs[done], "".join(log_lines), f"exit code {returncode}")
        if returncode != 0:
            # Every clip was delivered, but the plugin still reported failure.
            print(
                f"{RED}✗ TTS plugin exited with code {returncode} after finishing "
                f"all clips. Its trailing output:{NC}",
                file=sys.stderr,
            )
            print("".join(log_lines).rstrip(), file=sys.stderr)
            sys.exit(1)
    finally:
        # On any early exit (a failed clip, Ctrl-C) stop the plugin rather
        # than leave its engine running in the background.
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        jobs_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Canvas sizing and shared encoder settings
# ---------------------------------------------------------------------------


def get_image_size(image_path: Path) -> tuple[int, int] | None:
    """Return (width, height) of an image via ffprobe, or None on failure."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(image_path),
        ],
        capture_output=True,
        text=True,
    )
    parts = result.stdout.strip().split("x")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def compute_canvas_size(images: list[Path]) -> tuple[int, int]:
    """Pick the single output resolution every segment will be rendered onto.

    The most frequently occurring screenshot size wins (ties broken by the
    larger area), so the bulk of the footage stays pixel-exact and only the
    odd-sized frames — intro splashes, dialogs captured at another size — get
    scaled down and letterboxed.  The result is clamped to MAX_WIDTH/MAX_HEIGHT
    and forced to even dimensions, which H.264 4:2:0 chroma subsampling
    requires.
    """
    counts: dict[tuple[int, int], int] = {}
    for image in images:
        size = get_image_size(image)
        if size is not None:
            counts[size] = counts.get(size, 0) + 1

    if not counts:
        print(f"{RED}✗ Could not determine screenshot dimensions{NC}", file=sys.stderr)
        sys.exit(1)

    width, height = max(counts, key=lambda s: (counts[s], s[0] * s[1]))

    # Only ever shrink — upscaling screenshots blurs text for no benefit.
    ratio = min(MAX_WIDTH / width, MAX_HEIGHT / height, 1.0)
    width = int(width * ratio)
    height = int(height * ratio)

    return max(2, width - width % 2), max(2, height - height % 2)


def video_filter(canvas: tuple[int, int]) -> str:
    """Build the filter chain that normalises any image onto the canvas.

    ``scale`` performs the RGB→YUV conversion itself (the following ``format``
    filter constrains its output format), so ``out_color_matrix`` genuinely
    takes effect and the pixel data matches the BT.709 tags set on the output.
    Without that, swscale silently uses BT.601 coefficients and browsers render
    slightly shifted colours.  ``setsar=1`` guarantees square pixels so no
    player second-guesses the aspect ratio.
    """
    width, height = canvas
    return (
        f"scale={width}:{height}"
        ":force_original_aspect_ratio=decrease"
        ":flags=lanczos"
        ":out_color_matrix=bt709"
        ":out_range=tv,"
        "format=yuv420p,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={PAD_COLOR},"
        "setsar=1"
    )


def video_encode_args(canvas: tuple[int, int]) -> list[str]:
    """ffmpeg video arguments shared by every segment.

    These must be identical across segments so the concatenated MP4 has one
    unchanging SPS/PPS — see the note on the encoding constants above.
    """
    return [
        "-vf", video_filter(canvas),
        "-r", str(TARGET_FPS),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-profile:v", H264_PROFILE,
        "-level", H264_LEVEL,
        "-pix_fmt", "yuv420p",
        "-g", str(TARGET_FPS * GOP_SECONDS),
        # Tag the colour space explicitly; untagged streams are guessed at, and
        # different browsers guess differently.
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
    ]


def audio_encode_args() -> list[str]:
    """ffmpeg audio arguments shared by every segment (AAC-LC, the web baseline)."""
    return [
        "-c:a", "aac",
        "-profile:a", "aac_low",
        "-b:a", AUDIO_BITRATE,
        "-ar", str(AUDIO_RATE),
        "-ac", str(AUDIO_CHANNELS),
    ]


def segment_stream_signature(segment: Path) -> str | None:
    """Return a fingerprint of a segment's stream parameters, or None on failure.

    Used to assert that every segment really is concat-compatible before the
    "-c copy" join, so a mismatched segment fails loudly here instead of
    silently producing a file that only plays in permissive players.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,profile,level,"
            "sample_rate,channels,sample_fmt",
            "-of", "csv=p=0",
            str(segment),
        ],
        capture_output=True,
        text=True,
    )
    signature = result.stdout.strip()
    return signature or None


# ---------------------------------------------------------------------------
# Segment builders (ffmpeg)
# ---------------------------------------------------------------------------


def build_silent_segment(image: Path, segment: Path, canvas: tuple[int, int]) -> None:
    """Create a video segment from a still image with silent audio."""
    run_ffmpeg(
        [
            "-y",
            "-loop", "1", "-framerate", str(TARGET_FPS), "-i", str(image),
            "-f", "lavfi", "-i",
            f"anullsrc=r={AUDIO_RATE}:cl={'mono' if AUDIO_CHANNELS == 1 else 'stereo'}",
            "-t", str(FRAME_DURATION),
        ]
        + video_encode_args(canvas)
        + audio_encode_args()
        + [
            "-video_track_timescale", str(MOV_TIMESCALE),
            "-shortest",
            str(segment),
        ]
    )


def build_audio_segment(
    image: Path, audio: Path, segment: Path, canvas: tuple[int, int]
) -> None:
    """Create a video segment holding *image* on screen for the duration of *audio*."""
    run_ffmpeg(
        [
            "-y",
            "-loop", "1", "-framerate", str(TARGET_FPS), "-i", str(image),
            "-i", str(audio),
        ]
        + video_encode_args(canvas)
        + audio_encode_args()
        + [
            "-video_track_timescale", str(MOV_TIMESCALE),
            "-shortest",
            str(segment),
        ]
    )


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
        help=f"Max CPU threads for ffmpeg and the TTS engine (default: {MAX_THREADS})",
    )
    parser.add_argument(
        "--plugin",
        metavar="DIR",
        default=None,
        help=(
            "TTS plugin folder containing tts.sh (overrides $KOCREATOR_TTS_PLUGIN "
            "and tts_plugin in kocreator.toml; relative paths are resolved "
            "against the kocreator folder)"
        ),
    )
    args = parser.parse_args()

    if not HIGH_PERFORMANCE:
        # Apply the thread limit globally so run_ffmpeg and tts_env
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

    # --- TTS plugin: convert .txt → .wav ------------------------------------
    plugin_name = ""
    if tts_count > 0:
        plugin_dir = resolve_plugin_dir(args.plugin)
        plugin_name = plugin_dir.name
        check_plugin(plugin_dir)
        settings = plugin_settings(plugin_dir)
        settings_note = ", ".join(f"{k}: {v}" for k, v in settings.items())
        print(f"TTS plugin: {plugin_dir}")
        print(f"  ({settings_note})")
        print(f"Converting {tts_count} narration text file(s) to audio...")

        converted_files: list[Path] = []
        pending: list[TtsJob] = []
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

                text = f.read_text(encoding="utf-8").strip()
                if not text:
                    print(f"{RED}✗ Narration file is empty: {f.name}{NC}", file=sys.stderr)
                    sys.exit(1)
                seed = clip_seed(f)
                job = TtsJob(f, wav_path, seed, tts_manifest(text, seed, settings))

                if is_tts_cached(wav_path, job.manifest_data):
                    print(f"  {job.label} {GREEN}cached{NC} → {txt_basename}.wav")
                else:
                    pending.append(job)

                converted_files.append(wav_path)
                audio_count += 1
            else:
                converted_files.append(f)

        if pending:
            print(f"Generating {len(pending)} clip(s) in one batch "
                  "(the first one includes loading the engine)...")
            run_tts_batch(plugin_dir, pending)

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
        print(f"  ({tts_count} narration text file(s) converted via {plugin_name})")
    print(f"Image frame duration: {FRAME_DURATION}s")
    if audio_count == 0:
        print(f"Total video length: {image_count * FRAME_DURATION}s")
    print()

    # --- Decide the one output resolution used by every segment -------------
    all_images = [f for f in media_files if f.suffix == ".png"]
    canvas = compute_canvas_size(all_images)
    distinct_sizes = {get_image_size(f) for f in all_images} - {None}
    print(f"Output canvas: {canvas[0]}x{canvas[1]} "
          f"(H.264 {H264_PROFILE} profile, level {H264_LEVEL}, {TARGET_FPS}fps)")
    if len(distinct_sizes) > 1:
        sizes = ", ".join(f"{w}x{h}" for w, h in sorted(distinct_sizes))
        print(f"  {YELLOW}Screenshots vary in size ({sizes}) — off-size frames "
              f"will be scaled down and letterboxed onto the canvas.{NC}")
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
                build_silent_segment(f, segment_file, canvas)
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
                build_audio_segment(current_image, f, segment_file, canvas)
                print(f"{GREEN}✓{NC}")
                total_duration += audio_duration

            cl.write(f"file '{segment_file.resolve()}'\n")
            segment_index += 1

    print()
    print(f"Generated {segment_index} segments, total duration: ~{round(total_duration)}s")
    print()

    # --- Verify every segment is concat-compatible --------------------------
    # "-c copy" writes the first segment's stream parameters into the MP4
    # header and assumes the rest match.  A mismatch yields a file that plays
    # under permissive decoders but goes blank in browsers, so catch it here.
    print("Verifying segment stream parameters match...")
    segments = sorted(segment_dir.glob("segment-*.mp4"))
    signatures: dict[str, list[str]] = {}
    for seg in segments:
        signature = segment_stream_signature(seg)
        if signature is None:
            print(f"{RED}✗ Could not probe segment: {seg.name}{NC}")
            sys.exit(1)
        signatures.setdefault(signature, []).append(seg.name)

    if len(signatures) > 1:
        print(f"{RED}✗ Segments have mismatched stream parameters — the "
              f"concatenated MP4 would not play in most browsers.{NC}")
        for signature, names in signatures.items():
            print(f"    {signature.splitlines()}")
            print(f"      segments: {', '.join(names)}")
        sys.exit(1)
    print(f"{GREEN}✓ All {len(segments)} segments share identical stream parameters{NC}")
    print()

    # --- Concatenate segments into final MP4 --------------------------------
    print("Concatenating into final MP4...")
    run_ffmpeg(
        [
            "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            # +faststart moves the moov atom to the front so browsers can start
            # playing before the whole file has downloaded.
            "-movflags", "+faststart",
            "-video_track_timescale", str(MOV_TIMESCALE),
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
                f"  TTS:   {tts_count} narration(s) generated via {plugin_name} "
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
