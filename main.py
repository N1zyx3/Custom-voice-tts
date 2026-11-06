import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

TEXT = Path(input("Write path to output text file (.txt): "))
VOICE_WAV = Path(input("Write path to reference voice file (.wav, .mp3, .flac, .m4a, .ogg, .opus, .aiff): "))
VOICE_TEXT = Path(input("Write path to reference text file (.txt): "))
OUTPUT_FILE = "output.wav"
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
    if not TEXT.exists():
        sys.exit(f"ERROR: {TEXT} not found. Create it and put the text to speak there.")
    text = TEXT.read_text(encoding="utf-8").strip()
    if not text:
        sys.exit(f"ERROR: {TEXT} is empty.")
    return text


def verify_reference_files():
    if VOICE_WAV.suffix.lower() in AUDIO_EXTS:
        ref_audio = VOICE_WAV
    else:
        sys.exit("ERROR: unrecognized audio format.")
    if VOICE_TEXT.suffix.lower() == ".txt":
        ref_text = VOICE_TEXT.read_text(encoding="utf-8").strip()
        if not ref_text:
            sys.exit("ERROR: unrecognized text format or empty file.")
    return ref_audio, ref_text


def main():
    text = read_text_to_speak()
    ref_audio, ref_text = verify_reference_files()
    print(f"Text to speak ({len(text)} chars): {text[:200]}...")
    print(f"Reference audio: {ref_audio}")
    print(f"Reference transcript: {ref_text[:200]}...")

    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ERROR: ffmpeg not found in system!\n"
            "Install ffmpeg and add it into PATH,\n"
            "or put ffmpeg.exe into folder with this executable."
        )
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