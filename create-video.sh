#!/bin/bash

# Create a demo video from screenshots and optional audio narration
# Supports interleaved .png screenshots and audio in several formats:
#   .mp3  — used directly as audio input
#   .wav  — used directly as audio input (highest quality)
#   .txt  — converted to WAV via Kokoro TTS, then used as audio input
#
# Files are ordered by filename — use numeric prefixes (001-, 002-, etc.).
# During audio clips, the most recent screenshot is held on screen.
# Each screenshot without audio is displayed for FRAME_DURATION seconds.
#
# When .txt narration files are present, the Kokoro TTS engine is used to
# generate WAV audio. Generated files are cached in a generated-wav/
# subfolder inside the screenshot directory and reused on subsequent runs
# if the .txt source hasn't changed.  Set KOKORO_PROJECT_DIR below to
# point to your Kokoro project directory.
#
# Usage: ./create-video.sh <base-folder> <subfolder-name>
#
# Example folder structure:
#   <base-folder>/screenshots/my-demo/
#     001-welcome.png
#     002-narration.mp3          (or .wav, or .txt)
#     003-next-screen.png
#     004-explanation.txt         (narration text → Kokoro TTS → WAV)
#     005-final.png

set -e

# Validate arguments
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <base-folder> <subfolder-name>"
    echo "Example: $0 /home/user/myproject create-file-demo"
    exit 1
fi

BASEFOLDER="$1"
SUBFOLDER="$2"

# Check for empty strings
if [ -z "$BASEFOLDER" ]; then
    echo "Error: Base folder cannot be empty"
    exit 1
fi
if [ -z "$SUBFOLDER" ]; then
    echo "Error: Subfolder name cannot be empty"
    exit 1
fi

SCREENSHOT_DIR="$BASEFOLDER/screenshots/$SUBFOLDER"
OUTPUT_DIR="$BASEFOLDER/test-videos"
MP4_FILE="$OUTPUT_DIR/$SUBFOLDER.mp4"
GIF_FILE="$OUTPUT_DIR/$SUBFOLDER.gif"
PALETTE_FILE="$OUTPUT_DIR/$SUBFOLDER-palette.png"
SEGMENT_DIR="$OUTPUT_DIR/$SUBFOLDER-segments"
CONCAT_LIST="$SEGMENT_DIR/concat-list.txt"
FRAME_DURATION=2  # seconds per screenshot (images without audio)

# Path to the Kokoro TTS project directory (only needed when .txt narration files
# are used). Override this variable if your Kokoro project lives somewhere else.
# Uses the official kokoro package (hexgrad): https://github.com/hexgrad/kokoro
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KOKORO_PROJECT_DIR="${SCRIPT_DIR}/../kocreator"
KOKORO_VOICE="bm_daniel"

# Subdirectory for WAV files generated from .txt narration by Kokoro TTS.
# Lives inside the screenshot folder so cached audio stays next to its source.
GENERATED_WAV_DIR="$SCREENSHOT_DIR/generated-wav"

# --- Kokoro TTS helper ---
# Activates the Kokoro venv and runs kokoro-generate.py (official hexgrad kokoro).
# Supports [word](/phonemes/) inline pronunciation overrides via misaki G2P.
# Usage: run_kokoro_tts <input_txt> <output_wav>
run_kokoro_tts() {
    local input_txt="$(realpath "$1")"
    local output_wav="$(realpath --canonicalize-missing "$2")"
    pushd "$KOKORO_PROJECT_DIR" > /dev/null
    source .venv/bin/activate
    python kokoro-generate.py "$input_txt" "$output_wav" --voice "$KOKORO_VOICE"
    deactivate
    popd > /dev/null
}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}=== Create Video from Screenshots ===${NC}"
echo ""

# Check dependencies
if ! command -v ffmpeg &>/dev/null; then
    echo -e "${RED}✗ ffmpeg is not installed${NC}"
    echo "Install with: sudo apt install ffmpeg"
    exit 1
fi
if ! command -v ffprobe &>/dev/null; then
    echo -e "${RED}✗ ffprobe is not installed (usually ships with ffmpeg)${NC}"
    echo "Install with: sudo apt install ffmpeg"
    exit 1
fi

# Check that the directory exists and has media files
if [ ! -d "$SCREENSHOT_DIR" ]; then
    echo -e "${RED}✗ Directory not found: $SCREENSHOT_DIR/${NC}"
    echo "Run a demo test with screenshots enabled first."
    exit 1
fi

# Collect and sort all media files (png + mp3 + wav + txt)
# Exclude the generated-wav/ subfolder so cached WAVs aren't double-counted.
MEDIA_FILES=()
while IFS= read -r -d '' f; do
    MEDIA_FILES+=("$f")
done < <(find "$SCREENSHOT_DIR" -maxdepth 1 \( -name '*.png' -o -name '*.mp3' -o -name '*.wav' -o -name '*.txt' \) -not -path '*/generated-wav/*' -print0 | sort -z)

if [ ${#MEDIA_FILES[@]} -eq 0 ]; then
    echo -e "${RED}✗ No .png, .mp3, .wav, or .txt files found in $SCREENSHOT_DIR/${NC}"
    exit 1
fi

# Count by type
IMAGE_COUNT=0
AUDIO_COUNT=0
TTS_COUNT=0
for f in "${MEDIA_FILES[@]}"; do
    case "$f" in
        *.png) ((IMAGE_COUNT++)) || true ;;
        *.mp3|*.wav) ((AUDIO_COUNT++)) || true ;;
        *.txt) ((TTS_COUNT++)) || true ;;
    esac
done

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ No .png screenshots found in $SCREENSHOT_DIR/${NC}"
    echo "At least one screenshot is required."
    exit 1
fi

# Require the first file to be an image
FIRST_FILE="${MEDIA_FILES[0]}"
if [[ "$FIRST_FILE" != *.png ]]; then
    echo -e "${RED}✗ First file must be a .png screenshot, got: $(basename "$FIRST_FILE")${NC}"
    echo "Audio clips need a preceding screenshot to display."
    exit 1
fi

# --- Kokoro TTS: validate & convert .txt → .wav ---
if [ "$TTS_COUNT" -gt 0 ]; then
    # Verify Kokoro project directory and venv exist
    if [ ! -d "$KOKORO_PROJECT_DIR" ] || [ ! -f "$KOKORO_PROJECT_DIR/.venv/bin/activate" ]; then
        echo -e "${RED}✗ Kokoro TTS not found at: $KOKORO_PROJECT_DIR${NC}"
        echo "Narration .txt files require Kokoro TTS for text-to-speech conversion."
        echo "Expected a Python venv at: $KOKORO_PROJECT_DIR/.venv/"
        exit 1
    fi

    echo "Converting $TTS_COUNT narration text file(s) to audio via Kokoro TTS (voice: $KOKORO_VOICE)..."
    mkdir -p "$GENERATED_WAV_DIR"

    # Build a new array, replacing .txt entries with their generated .wav paths
    CONVERTED_FILES=()
    for f in "${MEDIA_FILES[@]}"; do
        if [[ "$f" == *.txt ]]; then
            TXT_BASENAME="$(basename "$f" .txt)"
            WAV_PATH="$GENERATED_WAV_DIR/${TXT_BASENAME}.wav"

            if [ -f "$WAV_PATH" ] && [ "$WAV_PATH" -nt "$f" ]; then
                echo -e "  [${TXT_BASENAME}.txt] ${GREEN}cached${NC} → ${TXT_BASENAME}.wav"
            else
                echo -n "  [${TXT_BASENAME}.txt] generating WAV ... "
                run_kokoro_tts "$f" "$WAV_PATH"
                echo -e "${GREEN}✓${NC}"
            fi

            CONVERTED_FILES+=("$WAV_PATH")
            ((AUDIO_COUNT++)) || true  # count the converted file as audio
        else
            CONVERTED_FILES+=("$f")
        fi
    done
    MEDIA_FILES=("${CONVERTED_FILES[@]}")
    echo ""
fi

# Create output and segment directories
mkdir -p "$OUTPUT_DIR"
rm -rf "$SEGMENT_DIR"
mkdir -p "$SEGMENT_DIR"

# Delete previous output files
rm -f "$MP4_FILE" "$GIF_FILE"

# Report what we found
echo "Found $IMAGE_COUNT screenshot(s) and $AUDIO_COUNT audio clip(s)"
if [ "$TTS_COUNT" -gt 0 ]; then
    echo "  ($TTS_COUNT narration text file(s) converted via Kokoro TTS)"
fi
echo "Image frame duration: ${FRAME_DURATION}s"
if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "Total video length: $((IMAGE_COUNT * FRAME_DURATION))s"
fi
echo ""

# --- Helper: build an audio segment (image held on screen + audio track) ---
# Usage: build_audio_segment <image> <audio_file> <segment_output>
# Audio is normalized to 44100 Hz stereo so all segments have matching
# parameters for the final concat step (silent image segments also use 44100/stereo).
build_audio_segment() {
    local image="$1"
    local audio="$2"
    local segment="$3"

    ffmpeg -y \
        -loop 1 -i "$image" \
        -i "$audio" \
        -c:v libx264 -preset slow -crf 18 \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
        -ar 44100 -ac 2 \
        -c:a aac -b:a 128k \
        -shortest \
        "$segment" \
        2>/dev/null
}

# --- Build video segments ---
# Each segment has both video and audio tracks so concat works with -c copy.
# Images get silent audio; audio clips get the most recent screenshot held on screen.

SEGMENT_INDEX=0
CURRENT_IMAGE=""
TOTAL_DURATION=0
> "$CONCAT_LIST"  # truncate concat list

for f in "${MEDIA_FILES[@]}"; do
    SEGMENT_FILE="$SEGMENT_DIR/segment-$(printf '%04d' $SEGMENT_INDEX).mp4"
    BASENAME="$(basename "$f")"

    if [[ "$f" == *.png ]]; then
        CURRENT_IMAGE="$f"
        echo -n "  [$BASENAME] image, ${FRAME_DURATION}s ... "
        ffmpeg -y \
            -loop 1 -i "$f" \
            -f lavfi -i anullsrc=r=44100:cl=stereo \
            -t "$FRAME_DURATION" \
            -c:v libx264 -preset slow -crf 18 \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
            -c:a aac -b:a 128k \
            -shortest \
            "$SEGMENT_FILE" \
            2>/dev/null
        echo -e "${GREEN}✓${NC}"
        TOTAL_DURATION=$(echo "$TOTAL_DURATION + $FRAME_DURATION" | bc)

    elif [[ "$f" == *.mp3 ]] || [[ "$f" == *.wav ]]; then
        # Get audio duration via ffprobe
        AUDIO_DURATION=$(ffprobe -v error \
            -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 \
            "$f" 2>/dev/null)

        if [ -z "$AUDIO_DURATION" ] || [ "$AUDIO_DURATION" = "N/A" ]; then
            echo -e "  [$BASENAME] ${RED}✗ could not detect duration, skipping${NC}"
            continue
        fi

        echo -n "  [$BASENAME] audio, ${AUDIO_DURATION}s (holding $(basename "$CURRENT_IMAGE")) ... "
        build_audio_segment "$CURRENT_IMAGE" "$f" "$SEGMENT_FILE"
        echo -e "${GREEN}✓${NC}"
        TOTAL_DURATION=$(echo "$TOTAL_DURATION + $AUDIO_DURATION" | bc)
    fi

    # Add segment to concat list
    echo "file '$(realpath "$SEGMENT_FILE")'" >> "$CONCAT_LIST"
    ((SEGMENT_INDEX++)) || true
done

echo ""
TOTAL_DURATION_INT=$(printf '%.0f' "$TOTAL_DURATION")
echo "Generated $SEGMENT_INDEX segments, total duration: ~${TOTAL_DURATION_INT}s"
echo ""

# --- Concatenate segments into final MP4 ---
echo "Concatenating into final MP4..."
ffmpeg -y \
    -f concat -safe 0 \
    -i "$CONCAT_LIST" \
    -c copy \
    -movflags +faststart \
    "$MP4_FILE" \
    2>&1 | grep -E "(frame=|Duration:|Output|error)" || true

if [ ! -f "$MP4_FILE" ]; then
    echo -e "${RED}✗ Failed to create MP4 video${NC}"
    exit 1
fi

MP4_SIZE=$(du -h "$MP4_FILE" | cut -f1)
echo -e "${GREEN}✓ MP4 created successfully${NC} ($MP4_SIZE)"
echo ""

# --- Create GIF (images only, no audio support in GIF) ---
echo "Generating GIF palette (images only, audio skipped)..."
ffmpeg -y \
    -framerate "1/$FRAME_DURATION" \
    -pattern_type glob \
    -i "$SCREENSHOT_DIR/*.png" \
    -vf "palettegen" \
    "$PALETTE_FILE" \
    2>&1 | grep -E "(frame=|Duration:|Output|error)" || true

if [ ! -f "$PALETTE_FILE" ]; then
    echo -e "${RED}✗ Failed to generate palette${NC}"
    exit 1
fi

echo "Creating GIF..."
ffmpeg -y \
    -framerate "1/$FRAME_DURATION" \
    -pattern_type glob \
    -i "$SCREENSHOT_DIR/*.png" \
    -i "$PALETTE_FILE" \
    -lavfi "paletteuse" \
    "$GIF_FILE" \
    2>&1 | grep -E "(frame=|Duration:|Output|error)" || true

# Clean up palette file
rm -f "$PALETTE_FILE"

if [ ! -f "$GIF_FILE" ]; then
    echo -e "${RED}✗ Failed to create GIF${NC}"
    exit 1
fi

GIF_SIZE=$(du -h "$GIF_FILE" | cut -f1)
echo -e "${GREEN}✓ GIF created successfully${NC} ($GIF_SIZE)"

# --- Clean up temp segments ---
rm -rf "$SEGMENT_DIR"

echo ""
echo -e "${GREEN}=== Done! ===${NC}"
echo "  MP4: $MP4_FILE ($MP4_SIZE)"
echo "  GIF: $GIF_FILE ($GIF_SIZE)"
if [ "$AUDIO_COUNT" -gt 0 ]; then
    echo "  Audio: $AUDIO_COUNT clip(s) included in MP4 (not in GIF)"
    if [ "$TTS_COUNT" -gt 0 ]; then
        echo "  TTS:   $TTS_COUNT narration(s) generated via Kokoro (cached in generated-wav/)"
    fi
fi
echo "  Subfolder: $SUBFOLDER"
echo ""
echo "To view MP4: mpv $MP4_FILE"
echo "To view GIF: mpv $GIF_FILE"
