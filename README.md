# Qwen3-TTS for Mac

Run Qwen3-TTS text-to-speech models locally on your Mac with Apple Silicon.

---

## Wait - There's a Better Version

I've made another repo using MLX (Apple's machine learning framework) that runs faster and uses less memory. If you want the best Mac experience, use that instead:

**[qwen3-tts-apple-silicon](https://github.com/kapi2800/qwen3-tts-apple-silicon)** - Native MLX version, better performance

This repo uses PyTorch with MPS backend. Still works great, but the MLX version is more optimized for Mac.

---

## What This Does

- Text-to-speech with multiple voice options
- Voice cloning from audio samples
- Custom voice design with text prompts
- Optimized for M1/M2/M3/M4 chips
- Simple terminal interface

## Terminal Demo

```
========================================
 Qwen3-TTS Studio (Mac Optimized)
========================================
 High Quality (1.7B)
  1. Custom Voice
  2. Voice Design
  3. Voice Clone

 Fast / Low Res (0.6B)
  4. Custom Voice
  5. Voice Clone

  6. Exit

Select Option: 1

[Loader] Loading CustomVoice (Pro)...
[System] M3/M4 detected. Using bfloat16 for efficiency.
[Loader] Model loaded successfully.

--- CustomVoice (Pro) Mode ---

Available Speakers:
  [English]: Ryan, Aiden
  [Chinese]: Vivian, Serena, Uncle_Fu, Dylan, Eric
  [Japanese]: Ono_Anna
  [Korean]: Sohee

Select Speaker: Ryan
[Selection] Speaker set to: Ryan

Enter text (or 'exit'): Hello, this is a test of the text to speech system.
[Status] Generating...
[IO] Saved to: outputs/CustomVoice_Pro/14-23-45_Hello_this_is_a_tes.wav
```

---

## Quick Start

### 1. Install LLVM

```bash
brew install llvm@20
```

Add to your shell (put in `~/.zshrc`):

```bash
export PATH="/usr/local/opt/llvm@20/bin:$PATH"
export CMAKE_PREFIX_PATH="/usr/local/opt/llvm@20"
```

### 2. Clone This Repo

```bash
git clone https://github.com/kapi2800/qwen3-tts-mac.git
cd qwen3-tts-mac
```

### 3. Clone Qwen3-TTS Library

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
```

### 4. Setup Python Environment

```bash
uv init -p 3.12
uv sync
uv add ./Qwen3-TTS
uv add 'huggingface-hub[cli]'
uv add 'numpy<2'
```

### 5. Download a Model

Pick one based on your Mac's RAM:

**8GB RAM - Use Mini (0.6B)**
```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

**16GB+ RAM - Use Pro (1.7B)**
```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

### 6. Run

```bash
uv run python main.py
```

---

## Model Options

| Model | RAM | Download Command |
|-------|-----|------------------|
| CustomVoice Pro | 8-10GB | `huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| VoiceDesign Pro | 8-10GB | `huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| VoiceClone Pro | 8-10GB | `huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./models/Qwen3-TTS-12Hz-1.7B-Base` |
| CustomVoice Mini | 4-6GB | `huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| VoiceClone Mini | 4-6GB | `huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./models/Qwen3-TTS-12Hz-0.6B-Base` |

---

## Configuration

Edit these values at the top of `main.py`:

```python
MAX_CHAR_LIMIT = 600                  # Max text length per generation
THERMAL_COOLDOWN_SECONDS = 0.3        # Pause between generations (reduce heat)
ENABLE_TORCH_COMPILE = False          # Experimental speedup
```

---

## Tips

- First generation is always slower (model warmup)
- Keep text under 600 characters
- Use Mini models if your Mac runs hot
- Audio files save to `outputs/` folder

---

## Updates

**Latest:**
- Dynamic token optimization for faster short text
- Memory-efficient model loading
- Thermal management between generations
- Auto bfloat16 for M3/M4 chips

---

If you made it this far and this actually worked for you, maybe hit that star button. Or don't. I'm just a readme.