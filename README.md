# Custom-voice TTS

Reads the text from `text.txt` (project root) and speaks it with a cloned voice whose reference lives in the `reference/` folder, using the Qwen3-TTS model.

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (voice-clone variant of Qwen3-TTS; best Russian quality in the series).
- Package: `qwen-tts` (`from qwen_tts import Qwen3TTSModel`).
- GPU required (CUDA). Machine: RTX 3060 Ti 8GB.

## Commands

- Environment: `.venv` (Python 3.13). Activate via `.venv\Scripts\activate`.
- Install deps: `python -m pip install -U qwen-tts` (do NOT install flash-attn; it is unreliable on Windows — model runs with default attention).
- IMPORTANT: plain `pip install qwen-tts` pulls a **CPU-only** torch on Windows. After installing, force the CUDA build: `python -m pip install "torch==2.14.0+cu130" "torchaudio==2.11.0+cu130" --index-url https://download.pytorch.org/whl/cu130`. Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
- Model weights (~4.3GB for 1.7B) are cached in `~/.cache/huggingface/hub/` after the first successful run.
- Run TTS: `python main.py`
- First run downloads model weights (~1-2GB) to the Hugging Face cache.

## File layout rules (important)

- `text.txt` — the text to be synthesized. Read the whole file (stripped), pass as a single string with `language="Auto"`.
- `reference/` — reference voice material. The program reads ONLY top-level **files** (audio + transcript `.txt`) and ignores subdirectories. Users keep a storage folder of voices *inside* `reference/`; it is never scanned.
  - Top-level of `reference/` is expected to contain exactly **one** audio file and exactly **one** `.txt` transcript.
  - Recognized audio extensions: `.wav .mp3 .flac .m4a .ogg .opus .aiff`.
  - If there is not exactly one audio + one text, the program fails with a clear error listing what was found.
- Output: `output.wav` in project root.

## Implementation notes

- Entry point: `main.py`.
- `Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)`.
- Synthesis: `wavs, sr = model.generate_voice_clone(text=..., language="Auto", ref_audio=<path>, ref_text=<transcript>)`; write with `soundfile`.
- The reference-scanning logic is file-only (never recurses into folders inside `reference/`).
- Long `text.txt` may need sentence-level chunking if quality degrades.
