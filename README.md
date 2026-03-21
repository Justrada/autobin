# FrameX

**Local-first video logging pipeline for DaVinci Resolve Studio.**

FrameX extracts contextual I-frames from your video files, transcribes audio, classifies clips using a local Vision LLM, detects multi-camera angles, and exports metadata as CSV for one-click import into DaVinci Resolve.

Everything runs locally on your machine — no cloud APIs required (though OpenAI and Anthropic are supported as optional backends).

![FrameX GUI](assets/framex-screenshot.png)

## What It Does

```
Video Files ──┬── Extract I-Frames (ffmpeg + auto-threshold tuning)
              │
              ├── Check Audio Levels (skip silent clips)
              │
              ├── Transcribe Audio (MLX-Whisper)
              │
              ├── Classify Clip (shot type, camera angle, roll type, etc.)
              │
              ├── Refine with Transcript (subject name, interview detection)
              │
              ├── Generate Keywords (VLM sees frames + transcript summary)
              │
              ├── Detect Multi-Camera Angles (transcript similarity matching)
              │
              └── Export CSV ──→ DaVinci Resolve: File > Import Metadata
```

### Features

- **Smart I-frame extraction** — Auto-tunes similarity threshold to hit a target frames-per-minute, so you get consistent results regardless of video content
- **Audio-aware pipeline** — Checks audio levels before transcription to avoid Whisper hallucination on silent B-roll
- **VLM clip classification** — Shot type, camera angle, camera movement, lighting, location, A-roll/B-roll, talking head detection
- **Transcript-powered refinement** — Identifies speaker names, validates interview vs. background chatter
- **Multi-camera detection** — Matches clips filmed from different angles of the same scene via 5-gram transcript overlap
- **Editable metadata** — Every auto-generated field can be overridden. Toggle between Auto and User values per field
- **DaVinci Resolve CSV export** — Per-clip and combined CSVs with all metadata columns
- **Folder scanning** — Drop a folder, process everything recursively. Subfolder names become tags
- **Custom vocabulary** — Bias Whisper toward project-specific names and terms

## Requirements

| Requirement | Install | Notes |
|---|---|---|
| **Python 3.10+** | [python.org](https://www.python.org/) or `brew install python` | 3.11+ recommended |
| **ffmpeg** | `brew install ffmpeg` | Used for frame extraction and audio processing |
| **Ollama** | [ollama.com](https://ollama.com/) | Local LLM server (default backend) |
| **Qwen 3.5 VL** | `ollama pull qwen3.5:latest` | Vision-language model for clip analysis |

### Hardware

- **Apple Silicon Mac recommended** — MLX-Whisper runs natively on M1/M2/M3/M4 chips
- **16GB+ RAM** — 24GB recommended for the Qwen 3.5 9B model
- Works on Intel Macs and Linux with `faster-whisper` as the transcription backend

## Installation

```bash
# Clone the repo
git clone https://github.com/justinestrada/framex.git
cd framex

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install transcription backend (pick one):
# Apple Silicon (recommended):
pip install mlx-whisper

# CPU fallback:
# pip install faster-whisper
```

### Ollama Setup

```bash
# Install Ollama (if not already installed)
brew install ollama

# Pull the default vision model
ollama pull qwen3.5:latest

# Start the Ollama server (runs in background)
ollama serve
```

## Usage

### GUI (recommended)

```bash
python main.py
```

1. **Add videos** — Click "Add Files" or "Add Folder" to queue videos
2. **Configure settings** — Adjust LLM model, transcription backend, similarity metric, etc.
3. **Start processing** — Click "Start" to run the full pipeline on your queue
4. **Review metadata** — Click any clip to see and edit its metadata in the right panel
5. **Export** — CSVs are auto-generated per clip and as a combined file

### CLI (frame extraction only)

The original CLI still works for standalone frame extraction:

```bash
python extract_iframes.py video.mp4 [-o DIR] [-t THRESHOLD] [-m histogram|ssim|phash]
```

## Settings

Settings are saved to `~/.config/vlm_iframe/settings.json` and persist between sessions.

| Section | Setting | Default | Description |
|---|---|---|---|
| **Ingest** | Metric | `histogram` | Frame similarity metric: histogram, SSIM, or perceptual hash |
| | Target FPM | `4.0` | Target frames per minute (auto-threshold tunes to this) |
| | Offset | `10` | Context frames before/after each I-frame |
| **LLM** | Backend | `ollama` | ollama, openai, or anthropic |
| | Model | `qwen3.5:latest` | Model name for the selected backend |
| | Context Window | `131072` | Token limit for batch planning |
| | VLM Resolution | `480` | Downscale height before sending frames to LLM |
| | Max Images/Batch | `4` | Images per VLM request |
| **Transcription** | Backend | `mlx-whisper` | mlx-whisper or faster-whisper |
| | Model Size | `base` | tiny, base, small, medium, or large |
| | Vocabulary | `""` | Comma-separated custom words for Whisper biasing |
| | Audio Check | `true` | Check audio levels before transcribing |
| | Noise Floor | `-50 dBFS` | Below this = silence (skip transcription) |
| **Export** | Format | `csv` | CSV for DaVinci Resolve metadata import |

## CSV Output

The exported CSV has these columns, ready for DaVinci Resolve's `File > Import Metadata > Media Pool`:

| Column | Source |
|---|---|
| File Name | Video filename |
| Keywords | VLM-generated + folder tags |
| Description | Transcript summary |
| Comments | Title + topics |
| Shot Type | VLM classification |
| Camera Angle | VLM classification |
| Camera Movement | VLM classification |
| Lighting | VLM classification |
| Location | VLM classification |
| Subject | VLM classification, refined by transcript |
| Roll Type | A-ROLL or B-ROLL |
| Subject Name | Extracted from transcript |
| Interview | Yes/No |
| Content Tags | Topical tags from transcript |
| Multicam Group | e.g. MC_001 (matched clips) |

## Project Structure

```
framex/
    main.py                     # GUI entry point
    extract_iframes.py          # CLI entry point (standalone)
    requirements.txt
    pyproject.toml

    core/
        schemas.py              # Pydantic models for settings, LLM output, results
        frames.py               # I-frame extraction + auto-threshold tuning
        transcribe.py           # MLX-Whisper / faster-whisper + audio level check
        llm.py                  # Ollama / OpenAI / Anthropic structured output
        token_budget.py         # Image token estimation for batch planning
        multicam.py             # Multi-camera detection via transcript similarity
        resolve_export.py       # CSV export for DaVinci Resolve

    gui/
        main_window.py          # Main application window
        queue_panel.py          # Video queue (add/remove/reorder)
        settings_panel.py       # All configurable settings
        progress_panel.py       # Pipeline progress display
        metadata_panel.py       # Editable Auto/User metadata fields
        orchestrator.py         # Pipeline sequencing + queue management
        workers.py              # QThread workers for background processing
```

## How Multi-Camera Detection Works

FrameX identifies clips filmed from different angles of the same scene by comparing their transcripts:

1. Normalize transcripts (lowercase, remove stopwords and punctuation)
2. Build 5-gram sets from each transcript
3. Use sliding window alignment to handle different start/end times
4. Score overlap using containment ratio (intersection / min set size)
5. Group connected matches using union-find
6. Threshold: 35% 5-gram overlap with minimum 15 matched n-grams

This works because multi-cam clips capture the same audio from different mics — the words are nearly identical despite minor Whisper transcription differences.

## Contributing

Contributions welcome! Some areas that could use help:

- **FCPXML export** — Generate timeline XML that Resolve/Premiere/FCPX can import
- **Resolve Scripting API** — Live metadata push to a running Resolve instance
- **Progress panel redesign** — Status indicator lights instead of log output
- **Film strip preview** — Show extracted I-frames as a visual strip
- **Pipeline parallelization** — Process multiple clips through different stages simultaneously
- **Linux/Windows testing** — Currently developed and tested on macOS

## License

[MIT](LICENSE)
