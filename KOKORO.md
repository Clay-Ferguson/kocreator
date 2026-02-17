# Kokoro TTS (Official)

Local text-to-speech using the official [kokoro](https://github.com/hexgrad/kokoro) package (hexgrad, Apache 2.0).

## Files

| File | Purpose |
|------|---------|
| `setup.sh` | One-time setup: creates Python venv, installs `kokoro` + `misaki[en]` from PyPI, downloads model from HuggingFace, runs smoke test |
| `kokoro-generate.py` | CLI wrapper: reads a text file, generates a WAV using the official kokoro Python API |
| `README.md` | This file |

## Setup (first time)

```bash
chmod +x setup.sh
./setup.sh
```

This requires internet access to download the model (~327 MB) and voice files from HuggingFace. These are cached locally in `~/.cache/huggingface/` and only downloaded once.

## Usage

```bash
source .venv/bin/activate
python kokoro-generate.py input.txt output.wav --voice bm_daniel
python kokoro-generate.py input.txt output.wav --voice af_heart --speed 1.2
```

## Offline Mode

`kokoro-generate.py` sets `HF_HUB_OFFLINE=1` internally so it **never contacts the internet** during normal use. The model runs entirely on your local CPU.

This setting only applies to `kokoro-generate.py` — it does not affect `setup.sh`, which needs internet access on first run to download model weights.

## Phonetic Overrides

The official kokoro uses [misaki](https://github.com/hexgrad/misaki) G2P, which supports inline pronunciation overrides:

```
[LaTeX](/lˈeɪtɛk/) renders mathematical formulas.
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model.
```

Phoneme reference: https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md

## Integration

MkBrowser's `create-video-from-screenshots.sh` calls `kokoro-generate.py` via the `run_kokoro_tts()` helper function to convert narration `.txt` files into `.wav` audio for demo videos.
