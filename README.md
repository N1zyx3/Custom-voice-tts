# Custom-voice TTS

Reads text from `.txt` files and synthesizes speech using a cloned voice from the `Qwen3-TTS` model.

## Model

- **Model**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (voice-clone variant of Qwen3-TTS; best Russian quality in the series).
- **Package**: `qwen-tts` (`from qwen_tts import Qwen3TTSModel`).
- **Requirements**: GPU with CUDA (CUDA 12.2 recommended). Model weights (~4.3GB) are cached in `~/.cache/huggingface/hub/` after the first run.

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies** (Windows - use CUDA build):
   ```bash
   python -m pip install -U qwen-tts
   # Force CUDA build (required for Windows)
   python -m pip install "torch==2.14.0+cu130" "torchaudio==2.11.0+cu130" --index-url https://download.pytorch.org/whl/cu130
   ```
   > **Note**: On Windows, `pip install qwen-tts` pulls a CPU-only torch. The CUDA build above is required for GPU acceleration.

3. **First run** will download the model weights (~1-2GB) to the Hugging Face cache.

## Usage

Run the TTS synthesis:
```bash
python main.py
```

The script expects the following file layout:

### Text File
- `text.txt` — the text to be synthesized. The entire file is read and stripped, then passed as a single string to the model with `language="Auto"`.

### Reference Directory
- `reference/` — contains **exactly one** audio file and **exactly one** `.txt` transcript at the top level.
  - **Audio formats**: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`, `.aiff`
  - **Transcript**: `.txt` file with the text to be spoken
  - The program only reads top-level files in `reference/`; subdirectories are ignored.

### Output
- `output.wav` — the synthesized speech file in the project root.

## How It Works

1. **Reference Audio Normalization**: The reference audio is converted to a 24kHz mono WAV file using `ffmpeg` to ensure compatibility with the model.
2. **Model Loading**: The `Qwen3TTSModel` is loaded from Hugging Face Hub with:
   - `device_map="cuda:0"` — uses GPU
   - `dtype=torch.bfloat16` — reduced precision for faster inference
3. **Speech Synthesis**: The model's `generate_voice_clone` method creates speech using both the input text and the reference audio/transcript for better quality.
4. **Output**: The synthesized waveform is saved as `output.wav`.

## Troubleshooting

- **ffmpeg not found**: Install ffmpeg and ensure it's in your system PATH.
- **CUDA not available**: Verify with `python -c "import torch; print(torch.cuda.is_available())"`. If not available, follow the CUDA build instructions above.
- **Reference directory issues**: Ensure `reference/` contains exactly one audio file and one `.txt` file at the top level.
- **Empty text file**: The script will exit with an error if `text.txt` is empty.

## License

MIT (check the `qwen-tts` package license for full details).