#!/usr/bin/env python3
"""
kokoro-generate.py — CLI wrapper for the official Kokoro TTS (hexgrad).

Reads a text file and generates a WAV audio file using the official kokoro
Python library with misaki G2P, which supports inline phonetic overrides
using the [word](/phonemes/) syntax.

Usage:
    python kokoro-generate.py input.txt output.wav --voice bm_daniel
    python kokoro-generate.py input.txt output.wav --voice af_heart --speed 1.2

Source: https://github.com/hexgrad/kokoro
"""

import argparse
import os
import sys
import numpy as np
import soundfile as sf

# Force fully offline mode — never contact HuggingFace Hub.
# Model and voice files must already be cached (downloaded during setup.sh).
os.environ['HF_HUB_OFFLINE'] = '1'

from kokoro import KPipeline

# Voice prefix -> language code mapping
# American English voices start with a (af_, am_), British with b (bf_, bm_)
LANG_CODE_MAP = {
    'a': 'a',  # American English
    'b': 'b',  # British English
    'e': 'e',  # Spanish
    'f': 'f',  # French
    'h': 'h',  # Hindi
    'i': 'i',  # Italian
    'j': 'j',  # Japanese
    'p': 'p',  # Portuguese
    'z': 'z',  # Mandarin Chinese
}

SAMPLE_RATE = 24000


def detect_lang_code(voice: str) -> str:
    """Detect language code from voice name prefix."""
    if voice and len(voice) >= 2:
        first_char = voice[0]
        if first_char in LANG_CODE_MAP:
            return LANG_CODE_MAP[first_char]
    return 'a'


def main():
    parser = argparse.ArgumentParser(
        description='Generate speech from text using official Kokoro TTS (hexgrad)')
    parser.add_argument('input', help='Input text file path')
    parser.add_argument('output', help='Output WAV file path')
    parser.add_argument('--voice', default='bm_daniel',
                        help='Voice name (default: bm_daniel)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speech speed (default: 1.0)')
    parser.add_argument('--lang', default=None,
                        help='Language code override (auto-detected from voice if omitted)')
    args = parser.parse_args()

    # Read input text
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not text:
        print(f"Error: Input file is empty: {args.input}", file=sys.stderr)
        sys.exit(1)

    lang_code = args.lang if args.lang else detect_lang_code(args.voice)
    pipeline = KPipeline(lang_code=lang_code)

    # Generate and concatenate audio chunks
    audio_chunks = []
    generator = pipeline(text, voice=args.voice, speed=args.speed, split_pattern=r'\n+')
    for i, (gs, ps, audio) in enumerate(generator):
        audio_chunks.append(audio)

    if not audio_chunks:
        print("Error: No audio generated", file=sys.stderr)
        sys.exit(1)

    full_audio = np.concatenate(audio_chunks)
    sf.write(args.output, full_audio, SAMPLE_RATE)
    print(f"Generated {args.output} ({len(full_audio) / SAMPLE_RATE:.1f}s)")


if __name__ == '__main__':
    main()
