# AutoBin

**Local-first video logging pipeline for DaVinci Resolve Studio.**

AutoBin automatically logs your documentary footage — extracting I-frames, transcribing audio, classifying clips using a local Vision LLM, detecting multi-camera angles, and exporting metadata as CSV for one-click import into DaVinci Resolve.

Everything runs locally on your machine — no cloud APIs required (though OpenAI and Anthropic are supported as optional backends).

![AutoBin GUI](assets/autobin-screenshot.png)

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
- **First-run setup wizard** — Automatically detects and installs missing dependencies (ffmpeg, Ollama, Qwen model)

## Quick Start (macOS App)

Download or build `AutoBin.app` and double-click to launch. On first run, the setup wizard will walk you through installing any missing dependencies.

```bash
# Build the app
./build_app.sh

# Launch it
open dist/AutoBin.app

# Or copy to Applications
cp -R dist/AutoBin.app /Applications/
```

## Requirements

| Requirement | Install | Notes |
|---|---|---|
| **Python 3.10+** | [python.org](https://www.python.org/) or `brew install python` | 3.11+ recommended |
| **ffmpeg** | `brew install ffmpeg` | Auto-installed by setup wizard |
| **Ollama** | [ollama.com](https://ollama.com/) | Auto-installed by setup wizard |
| **Qwen 3.5 VL** | `ollama pull qwen3.5:latest` | Auto-pulled by setup wizard |

### Hardware

- **Apple Silicon Mac recommended** — MLX-Whisper runs natively on M1/M2/M3/M4 chips
- **16GB+ RAM** — 24GB recommended for the Qwen 3.5 9B model
- Works on Intel Macs and Linux with `faster-whisper` as the transcription backend

## Installation (from source)

```bash
# Clone the repo
git clone https://github.com/Justrada/autobin.git
cd autobin

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

## Build as macOS App

You can build a standalone `AutoBin.app` that you double-click to launch — no terminal needed:

```bash
# Install PyInstaller (one time)
pip install pyinstaller

# Build the app
./build_app.sh

# Launch it
open dist/AutoBin.app

# Or copy to Applications
cp -R dist/AutoBin.app /Applications/
```

> **Note:** The app bundle is ~1.2 GB because it includes Python, PySide6, OpenCV, and numpy. On first launch, the setup wizard checks for ffmpeg and Ollama and offers one-click install via Homebrew.

## Usage

### GUI (recommended)

```bash
python main.py
```

1. **First run** — The setup wizard checks for ffmpeg, Ollama, and the Qwen model. Install anything missing with one click.
2. **Add videos** — Click "Add Files" or "Add Folder" to queue videos
3. **Configure settings** — Adjust LLM model, transcription backend, similarity metric, etc.
4. **Start processing** — Click "Start" to run the full pipeline on your queue
5. **Review metadata** — Click any clip to see and edit its metadata in the right panel
6. **Export** — CSVs are auto-generated per clip and as a combined file

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
autobin/
    main.py                     # GUI entry point (+ setup wizard trigger)
    extract_iframes.py          # CLI entry point (standalone)
    build_app.sh                # One-command macOS .app build
    AutoBin.spec                # PyInstaller configuration
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
        setup_wizard.py         # First-run dependency checker + installer
        queue_panel.py          # Video queue (add/remove/reorder)
        settings_panel.py       # All configurable settings
        progress_panel.py       # Pipeline progress display
        metadata_panel.py       # Editable Auto/User metadata fields
        orchestrator.py         # Pipeline sequencing + queue management
        workers.py              # QThread workers for background processing
```

## How Multi-Camera Detection Works

AutoBin identifies clips filmed from different angles of the same scene by comparing their transcripts:

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
