"""Audio transcription backends: MLX-Whisper (default) and faster-whisper."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Callable

from core.schemas import TranscriptionSettings


MLX_WHISPER_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
}

# Default cache dir for MLX whisper models
WHISPER_CACHE_DIR = os.path.expanduser("~/.cache/vlm_iframe/whisper_models")


def get_whisper_model_path(model_size: str) -> str:
    """Return local cache path for a whisper model."""
    return os.path.join(WHISPER_CACHE_DIR, f"whisper-{model_size}-mlx")


def is_whisper_model_downloaded(model_size: str) -> bool:
    """Check if a whisper model has been downloaded locally."""
    model_path = get_whisper_model_path(model_size)
    if not os.path.isdir(model_path):
        return False
    # Check for at least a config and weight file
    has_config = os.path.exists(os.path.join(model_path, "config.json"))
    has_weights = any(
        f.endswith((".npz", ".safetensors"))
        for f in os.listdir(model_path)
    ) if os.path.isdir(model_path) else False
    return has_config and has_weights


def download_whisper_model(model_size: str,
                           log: Callable[[str], None] | None = None) -> str:
    """
    Download an MLX whisper model from HuggingFace to local cache.
    Returns the local path.
    """
    _log = log or (lambda m: None)
    repo_id = MLX_WHISPER_MODELS.get(model_size)
    if not repo_id:
        _log(f"[whisper] Unknown model size: {model_size}")
        return ""

    model_path = get_whisper_model_path(model_size)
    os.makedirs(model_path, exist_ok=True)

    _log(f"[whisper] Downloading {repo_id} to {model_path}...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )
        _log(f"[whisper] Download complete: {model_size}")
        return model_path
    except ImportError:
        _log("[whisper] huggingface_hub not installed. Run: pip install huggingface-hub")
        return ""
    except Exception as e:
        _log(f"[whisper] Download failed: {e}")
        return ""


def check_audio_level(video_path: str, threshold_db: float = -40.0,
                      log: Callable[[str], None] | None = None) -> dict:
    """
    Check if the middle 50% of a clip has usable audio.

    Returns dict with:
      - has_audio: bool (True if audio level is above threshold)
      - rms_db: float (RMS level in dBFS)
      - peak_db: float (peak level in dBFS)
      - speech_ratio: float (0-1, fraction of audio with speech — only if Silero VAD available)
    """
    _log = log or (lambda m: None)
    result = {"has_audio": False, "rms_db": -96.0, "peak_db": -96.0, "speech_ratio": 0.0}

    # Get duration first
    probe_cmd = [
        "ffprobe", "-hide_banner", "-loglevel", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ]
    try:
        dur_out = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        duration = float(dur_out.stdout.strip())
    except Exception:
        _log("[audio-check] Could not determine duration")
        return result

    # Analyze the middle 50% to avoid start/end pops
    start_time = duration * 0.25
    analyze_duration = duration * 0.5
    if analyze_duration < 0.5:
        analyze_duration = min(duration, 1.0)
        start_time = 0

    # Select best audio stream (avoid ambisonic on .360 files)
    stream_map = _find_best_audio_stream(video_path)

    # Use ffmpeg astats to get RMS and peak levels
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(analyze_duration),
    ]
    if stream_map:
        cmd += ["-map", stream_map]
    cmd += [
        "-vn", "-af", "astats=metadata=1:reset=0,ametadata=print:key=lavfi.astats.Overall.RMS_level:key=lavfi.astats.Overall.Peak_level",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        _log("[audio-check] Analysis timed out")
        return result

    # Parse RMS and peak from astats output
    rms_values = []
    peak_values = []
    for line in stderr.split("\n"):
        if "RMS_level" in line:
            try:
                val = float(line.split("=")[-1].strip())
                if val > -200:  # filter out -inf
                    rms_values.append(val)
            except (ValueError, IndexError):
                pass
        elif "Peak_level" in line:
            try:
                val = float(line.split("=")[-1].strip())
                if val > -200:
                    peak_values.append(val)
            except (ValueError, IndexError):
                pass

    if not rms_values:
        # Fallback: use volumedetect filter
        cmd2 = [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(analyze_duration),
        ]
        if stream_map:
            cmd2 += ["-map", stream_map]
        cmd2 += [
            "-vn", "-af", "volumedetect",
            "-f", "null", "-",
        ]
        try:
            proc2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
            for line in proc2.stderr.split("\n"):
                if "mean_volume" in line:
                    try:
                        result["rms_db"] = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                    except (ValueError, IndexError):
                        pass
                elif "max_volume" in line:
                    try:
                        result["peak_db"] = float(line.split("max_volume:")[1].split("dB")[0].strip())
                    except (ValueError, IndexError):
                        pass
        except subprocess.TimeoutExpired:
            pass
    else:
        result["rms_db"] = sum(rms_values) / len(rms_values)
        result["peak_db"] = max(peak_values) if peak_values else result["rms_db"]

    result["has_audio"] = result["rms_db"] > threshold_db
    _log(f"[audio-check] RMS: {result['rms_db']:.1f} dB, Peak: {result['peak_db']:.1f} dB, "
         f"threshold: {threshold_db} dB → {'PASS' if result['has_audio'] else 'SKIP'}")

    # Tier 2: Silero VAD speech detection (if audio passes noise floor)
    if result["has_audio"]:
        result["speech_ratio"] = _check_speech_ratio(
            video_path, start_time, analyze_duration, log=log
        )
        if result["speech_ratio"] < 0.05:
            _log(f"[audio-check] Speech ratio {result['speech_ratio']:.1%} — "
                 f"ambient audio only, skipping transcription")
            result["has_audio"] = False

    return result


def _check_speech_ratio(video_path: str, start_time: float, duration: float,
                        log: Callable[[str], None] | None = None) -> float:
    """
    Estimate speech presence using ffmpeg's silencedetect filter.
    Returns ratio of non-silent audio (0-1).
    Falls back to 1.0 on error.
    """
    _log = log or (lambda m: None)

    # Use silencedetect to find silent segments — what's left is likely speech
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-vn", "-af", f"silencedetect=noise=-35dB:d=0.5",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = proc.stderr
    except Exception as e:
        _log(f"[audio-check] Speech detection failed: {e}")
        return 1.0

    # Parse silence durations from output
    import re as _re
    silence_durations = []
    for match in _re.finditer(r"silence_duration:\s*([\d.]+)", stderr):
        silence_durations.append(float(match.group(1)))

    total_silence = sum(silence_durations)
    speech_ratio = max(0.0, 1.0 - (total_silence / duration)) if duration > 0 else 0.0

    _log(f"[audio-check] Speech ratio: {speech_ratio:.0%} "
         f"({len(silence_durations)} silent segments, {total_silence:.1f}s silent of {duration:.1f}s)")
    return speech_ratio


def _parse_vocabulary(vocab_str: str) -> list[str]:
    """Parse comma-separated vocabulary string into a list of words."""
    if not vocab_str.strip():
        return []
    return [w.strip() for w in vocab_str.split(",") if w.strip()]


def _build_initial_prompt(vocab: list[str]) -> str | None:
    """
    Build a Whisper initial_prompt from vocabulary words.
    Whisper uses the initial_prompt as context — words that appear in it
    are more likely to be recognized correctly in the transcription.
    """
    if not vocab:
        return None
    return "The following proper nouns and terms may appear: " + ", ".join(vocab) + "."


def _similarity(a: str, b: str) -> float:
    """Simple character-level similarity ratio (0-1)."""
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0
    # Count matching characters in order (LCS-like)
    matches = sum(1 for ca, cb in zip(a, b) if ca == cb)
    return (2.0 * matches) / (len(a) + len(b))


def _post_process_transcript(text: str, vocab: list[str]) -> str:
    """
    Fix misheard words using the vocabulary list.
    1. Exact case-insensitive replacement (e.g. "asunda" → "Asunda")
    2. Fuzzy matching for single words that are phonetically close
       (e.g. "Naomi" → "Niobe" if similarity > 0.5 and same first letter)
    """
    if not vocab or not text:
        return text

    # Step 1: Exact case-insensitive matches for multi-word terms first
    for correct_word in vocab:
        if not correct_word:
            continue
        if " " in correct_word:
            parts = correct_word.split()
            loose_pattern = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"
            text = re.sub(loose_pattern, correct_word, text, flags=re.IGNORECASE)

    # Step 2: Single-word exact case fix
    for correct_word in vocab:
        if not correct_word or " " in correct_word:
            continue
        pattern = re.compile(r"\b" + re.escape(correct_word) + r"\b", re.IGNORECASE)
        text = pattern.sub(correct_word, text)

    # Step 3: Fuzzy replacement for single-word vocab items
    # Find words in text that look similar to vocab words but aren't exact matches
    single_vocab = [w for w in vocab if w and " " not in w and len(w) >= 3]
    if single_vocab:
        words = text.split()
        for i, word in enumerate(words):
            # Strip punctuation for comparison
            clean = re.sub(r"[^\w]", "", word)
            if len(clean) < 3:
                continue
            for correct in single_vocab:
                # Already correct
                if clean.lower() == correct.lower():
                    break
                # Fuzzy match: similar length, high character similarity,
                # and same first letter (phonetic heuristic)
                len_ratio = len(clean) / len(correct)
                if (0.7 <= len_ratio <= 1.4
                        and clean[0].lower() == correct[0].lower()
                        and _similarity(clean, correct) >= 0.55):
                    # Preserve original punctuation
                    words[i] = word.replace(clean, correct)
                    break
        text = " ".join(words)

    return text


class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, initial_prompt: str | None = None,
                   log: Callable[[str], None] | None = None) -> str:
        ...


def _find_python_with_mlx_whisper() -> str | None:
    """Find a Python interpreter that has mlx-whisper installed.

    Checks (in order):
    1. .venv next to this source file (running from source)
    2. .venv relative to the .app bundle location (PyInstaller)
    3. The Python that launched this process
    4. Common Homebrew / pyenv / system Python paths
    5. ``python3`` on PATH
    """
    import shutil
    import sys

    candidates: list[str] = []

    # 1. Walk up from this source file to find .venv
    here = os.path.dirname(os.path.abspath(__file__))
    for _i in range(5):  # walk up to 5 levels
        venv_py = os.path.join(here, ".venv", "bin", "python")
        if os.path.isfile(venv_py):
            candidates.append(venv_py)
            break
        here = os.path.dirname(here)

    # 2. Walk up from sys.executable (PyInstaller .app):
    #    e.g. dist/AutoBin.app/Contents/MacOS/AutoBin → project root
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    for _i in range(6):
        venv_py = os.path.join(exe_dir, ".venv", "bin", "python")
        if os.path.isfile(venv_py):
            if venv_py not in candidates:
                candidates.append(venv_py)
            break
        exe_dir = os.path.dirname(exe_dir)

    # 3. The Python that launched this process
    if sys.executable and os.path.isfile(sys.executable):
        candidates.append(sys.executable)

    # 4. Common locations
    home = os.path.expanduser("~")
    extra_paths = [
        os.path.join(home, "Documents", "VLM_I_FRAME_EXTRACTOR", ".venv", "bin", "python"),
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
    ]
    # pyenv versions
    pyenv_root = os.path.join(home, ".pyenv", "versions")
    if os.path.isdir(pyenv_root):
        try:
            for ver_dir in sorted(os.listdir(pyenv_root), reverse=True):
                py = os.path.join(pyenv_root, ver_dir, "bin", "python3")
                if os.path.isfile(py):
                    extra_paths.append(py)
                    break
        except OSError:
            pass

    for p in extra_paths:
        if os.path.isfile(p) and p not in candidates:
            candidates.append(p)

    # 5. System python3 on PATH
    sys_py = shutil.which("python3")
    if sys_py and sys_py not in candidates:
        candidates.append(sys_py)

    for py in candidates:
        try:
            r = subprocess.run(
                [py, "-c", "import mlx_whisper; print('ok')"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and "ok" in r.stdout:
                return py
        except Exception:
            continue
    return None


class MLXWhisperBackend(TranscriptionBackend):
    """Uses mlx-whisper for Apple Silicon optimized transcription.

    First tries an in-process import.  If that fails (e.g. when running
    inside a PyInstaller .app where native Metal libs can't be bundled),
    falls back to running mlx-whisper in a subprocess using whichever
    Python on the system has it installed.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size

    def transcribe(self, audio_path: str, initial_prompt: str | None = None,
                   log: Callable[[str], None] | None = None) -> str:
        _log = log or (lambda m: None)

        # --- Attempt 1: in-process import (works when running from source) ---
        try:
            import mlx_whisper
            return self._transcribe_in_process(mlx_whisper, audio_path, initial_prompt, _log)
        except ImportError as exc:
            _log(f"[transcribe] In-process mlx-whisper unavailable ({exc}), trying subprocess...")
        except Exception as exc:
            _log(f"[transcribe] In-process mlx-whisper failed ({exc}), trying subprocess...")

        # --- Attempt 2: subprocess fallback (for .app bundles) ---
        return self._transcribe_subprocess(audio_path, initial_prompt, _log)

    # --------------------------------------------------------------------- #
    def _transcribe_in_process(self, mlx_whisper, audio_path: str,
                                initial_prompt: str | None,
                                _log: Callable[[str], None]) -> str:
        _log(f"[transcribe] MLX-Whisper ({self.model_size}) processing in-process...")
        if initial_prompt:
            _log(f"[transcribe] Vocabulary prompt: {initial_prompt[:80]}...")

        local_path = get_whisper_model_path(self.model_size)
        if is_whisper_model_downloaded(self.model_size):
            model_name = local_path
            _log(f"[transcribe] Using cached model: {local_path}")
        else:
            model_name = MLX_WHISPER_MODELS.get(
                self.model_size,
                f"mlx-community/whisper-{self.model_size}-mlx",
            )
            _log("[transcribe] Downloading model from HuggingFace (first run)...")

        kwargs = {"path_or_hf_repo": model_name}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        result = mlx_whisper.transcribe(audio_path, **kwargs)
        text = result.get("text", "").strip()
        _log(f"[transcribe] Got {len(text)} chars of transcript")
        return text

    # --------------------------------------------------------------------- #
    def _transcribe_subprocess(self, audio_path: str,
                                initial_prompt: str | None,
                                _log: Callable[[str], None]) -> str:
        py = _find_python_with_mlx_whisper()
        if not py:
            _log("[transcribe] ✗ No Python with mlx-whisper found. "
                 "Install it: pip install mlx-whisper")
            return ""
        _log(f"[transcribe] Using subprocess: {py}")

        local_path = get_whisper_model_path(self.model_size)
        if is_whisper_model_downloaded(self.model_size):
            model_name = local_path
        else:
            model_name = MLX_WHISPER_MODELS.get(
                self.model_size,
                f"mlx-community/whisper-{self.model_size}-mlx",
            )

        # Build a small Python script to run transcription and print the text
        script = (
            "import json, sys, mlx_whisper\n"
            f"kwargs = {{'path_or_hf_repo': {model_name!r}}}\n"
        )
        if initial_prompt:
            script += f"kwargs['initial_prompt'] = {initial_prompt!r}\n"
        script += (
            f"result = mlx_whisper.transcribe({audio_path!r}, **kwargs)\n"
            "print(json.dumps({'text': result.get('text', '')}))\n"
        )

        _log(f"[transcribe] MLX-Whisper ({self.model_size}) via subprocess...")
        try:
            proc = subprocess.run(
                [py, "-c", script],
                capture_output=True, text=True,
                timeout=600,  # 10 min max for long clips
            )
            if proc.returncode != 0:
                stderr_short = (proc.stderr or "")[:500]
                _log(f"[transcribe] Subprocess failed (exit={proc.returncode}): {stderr_short}")
                return ""

            import json
            data = json.loads(proc.stdout.strip().split("\n")[-1])
            text = data.get("text", "").strip()
            _log(f"[transcribe] Got {len(text)} chars of transcript")
            return text
        except subprocess.TimeoutExpired:
            _log("[transcribe] Subprocess timed out (>10min)")
            return ""
        except Exception as exc:
            _log(f"[transcribe] Subprocess error: {exc}")
            return ""


class FasterWhisperBackend(TranscriptionBackend):
    """Uses faster-whisper as CPU fallback."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size

    def transcribe(self, audio_path: str, initial_prompt: str | None = None,
                   log: Callable[[str], None] | None = None) -> str:
        _log = log or (lambda m: None)
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            _log("[transcribe] faster-whisper not installed. Run: pip install faster-whisper")
            return ""

        _log(f"[transcribe] faster-whisper ({self.model_size}) processing...")
        model = WhisperModel(self.model_size, compute_type="int8")
        kwargs = {}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segments, _ = model.transcribe(audio_path, **kwargs)
        text = " ".join(seg.text for seg in segments).strip()
        _log(f"[transcribe] Got {len(text)} chars of transcript")
        return text


def get_backend(settings: TranscriptionSettings) -> TranscriptionBackend:
    """Factory for transcription backends."""
    if settings.backend == "mlx-whisper":
        return MLXWhisperBackend(settings.model_size)
    elif settings.backend == "faster-whisper":
        return FasterWhisperBackend(settings.model_size)
    else:
        return MLXWhisperBackend(settings.model_size)


def _find_best_audio_stream(video_path: str) -> str | None:
    """Probe for the first stereo/mono AAC or PCM audio stream (skip ambisonic).

    GoPro .360 files have both a normal AAC track and an ambisonic
    (4-channel) track.  ffmpeg's default stream selection sometimes picks
    the ambisonic one, which the resampler can't handle.  This function
    returns an ffmpeg map specifier like '0:1' for the first usable audio
    stream, or None if only one audio stream exists (let ffmpeg decide).
    """
    probe_cmd = [
        "ffprobe", "-hide_banner", "-loglevel", "error",
        "-show_entries", "stream=index,codec_type,channels,codec_name",
        "-select_streams", "a",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        proc = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        lines = [l.strip() for l in proc.stdout.strip().split("\n") if l.strip()]
    except Exception:
        return None

    if len(lines) <= 1:
        return None  # single audio stream, no need to force-select

    # Output format: index,codec_name,codec_type,channels
    # Prefer the first stream with <= 2 channels (stereo or mono)
    for line in lines:
        parts = line.split(",")
        if len(parts) < 4:
            continue
        idx = parts[0]
        channels_str = parts[-1]  # channels is always the last column
        try:
            channels = int(channels_str)
        except ValueError:
            continue
        if channels <= 2:
            return f"0:{idx}"

    return None  # fallback: let ffmpeg decide


def extract_audio(video_path: str, output_dir: str | None = None,
                  log: Callable[[str], None] | None = None) -> str:
    """Extract audio from video to a temporary WAV file using ffmpeg.

    Automatically selects the first stereo/mono audio stream to avoid
    ambisonic tracks on multi-stream files (e.g. GoPro .360).
    """
    _log = log or (lambda m: None)
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    audio_path = os.path.join(output_dir, "audio.wav")

    stream_map = _find_best_audio_stream(video_path)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", video_path,
    ]
    if stream_map:
        cmd += ["-map", stream_map]
        _log(f"[transcribe] Selecting audio stream {stream_map} (skipping ambisonic)")
    cmd += [
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", audio_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Verify non-empty output
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        _log(f"[transcribe] Audio extraction produced empty file (ffmpeg exit={proc.returncode})")
        if proc.stderr:
            _log(f"[transcribe] ffmpeg: {proc.stderr[:300]}")
        # Retry without stream map as fallback
        if stream_map:
            _log("[transcribe] Retrying without stream selection...")
            cmd_fallback = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning",
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-y", audio_path,
            ]
            subprocess.run(cmd_fallback, capture_output=True, text=True)

    return audio_path


def transcribe_video(video_path: str, settings: TranscriptionSettings,
                     log: Callable[[str], None] | None = None) -> str:
    """Full pipeline: extract audio, transcribe, then post-process with vocabulary."""
    _log = log or (lambda m: None)
    _log("[transcribe] Extracting audio from video...")

    vocab = _parse_vocabulary(settings.vocabulary)
    initial_prompt = _build_initial_prompt(vocab)

    if vocab:
        _log(f"[transcribe] Custom vocabulary: {', '.join(vocab)}")

    with tempfile.TemporaryDirectory(prefix="transcribe_") as tmp:
        audio_path = extract_audio(video_path, tmp, log=_log)
        if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
            _log("[transcribe] Failed to extract audio (missing or empty).")
            return ""

        backend = get_backend(settings)
        text = backend.transcribe(audio_path, initial_prompt=initial_prompt, log=log)

        # Post-process: fix any remaining misheard vocabulary words
        if vocab and text:
            text = _post_process_transcript(text, vocab)
            _log(f"[transcribe] Post-processed with {len(vocab)} vocabulary words")

        return text
