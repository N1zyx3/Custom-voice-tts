import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parent
TEXT_FILE = ROOT / "text.txt"
REFERENCE_DIR = ROOT / "reference"
OUTPUT_FILE = ROOT / "output.wav"
REF_SAMPLE_RATE = 24000

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aiff"}


def normalize_reference_audio(audio_path: Path) -> Path:
    """Convert the reference audio to a 24kHz mono WAV (ffmpeg), to avoid
    formats that torchaudio/libsndfile cannot decode."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install ffmpeg or provide a .wav reference audio."
        )
    tmp_dir = Path(tempfile.gettempdir()) / "qwen3-tts-ref"
    tmp_dir.mkdir(exist_ok=True)
    out_path = tmp_dir / f"ref-{audio_path.stem}.wav"
    result = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(audio_path),
            "-ar", str(REF_SAMPLE_RATE), "-ac", "1",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg failed to convert {audio_path}: {result.stderr.strip()}")
    return out_path


def read_text_to_speak() -> str:
    if not TEXT_FILE.exists():
        sys.exit(f"ERROR: {TEXT_FILE} not found. Create it and put the text to speak there.")
    text = TEXT_FILE.read_text(encoding="utf-8").strip()
    if not text:
        sys.exit(f"ERROR: {TEXT_FILE} is empty.")
    return text


def find_reference_pair():
    if not REFERENCE_DIR.is_dir():
        sys.exit(f"ERROR: folder {REFERENCE_DIR} not found. Create it and put the reference voice + transcript there.")

    audio_files = []
    text_files = []
    for entry in REFERENCE_DIR.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() in AUDIO_EXTS:
            audio_files.append(entry)
        elif entry.suffix.lower() == ".txt":
            text_files.append(entry)

    if len(audio_files) != 1 or len(text_files) != 1:
        found = [
            f"{entry.name} ({'audio' if entry.suffix.lower() in AUDIO_EXTS else 'text'})"
            for entry in REFERENCE_DIR.iterdir()
            if entry.is_file() and (entry.suffix.lower() in AUDIO_EXTS or entry.suffix.lower() == ".txt")
        ]
        sys.exit(
            "ERROR: reference/ must contain exactly one audio file and one .txt transcript "
            f"(found audio: {len(audio_files)}, text: {len(text_files)}).\n"
            f"Found files: {found or 'none'}"
        )

    ref_audio = audio_files[0]
    ref_text = text_files[0].read_text(encoding="utf-8").strip()
    if not ref_text:
        sys.exit(f"ERROR: transcript file {text_files[0]} is empty.")
    return ref_audio, ref_text


def main():
    text = read_text_to_speak()
    ref_audio, ref_text = find_reference_pair()
    print(f"Text to speak ({len(text)} chars): {text[:200]}...")
    print(f"Reference audio: {ref_audio}")
    print(f"Reference transcript: {ref_text[:200]}...")

    ref_wav = normalize_reference_audio(ref_audio)
    print(f"Normalized reference audio: {ref_wav}")

    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    print("Synthesizing speech...")
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Auto",
        ref_audio=str(ref_wav),
        ref_text=ref_text,
    )

    sf.write(str(OUTPUT_FILE), wavs[0], sr)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()