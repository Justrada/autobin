# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AutoBin.

Usage:
    pyinstaller AutoBin.spec

Output:
    dist/AutoBin.app
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all PySide6 submodules and data
pyside6_submodules = collect_submodules("PySide6")
pyside6_data = collect_data_files("PySide6")

# Collect pydantic submodules (needed for model validation)
pydantic_submodules = collect_submodules("pydantic")

# Collect skimage data files
skimage_data = collect_data_files("skimage")

# Collect mlx-whisper + mlx (Apple Silicon transcription)
mlx_whisper_submodules = collect_submodules("mlx_whisper")
mlx_submodules = collect_submodules("mlx")
mlx_whisper_data = collect_data_files("mlx_whisper")
mlx_data = collect_data_files("mlx")

# Collect huggingface_hub (for model downloads)
huggingface_submodules = collect_submodules("huggingface_hub")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[
        *pyside6_data,
        *skimage_data,
        *mlx_whisper_data,
        *mlx_data,
    ],
    hiddenimports=[
        *pyside6_submodules,
        *pydantic_submodules,
        *mlx_whisper_submodules,
        *mlx_submodules,
        *huggingface_submodules,
        "core",
        "core.schemas",
        "core.frames",
        "core.llm",
        "core.multicam",
        "core.resolve_export",
        "core.token_budget",
        "core.transcribe",
        "gui",
        "gui.main_window",
        "gui.metadata_panel",
        "gui.orchestrator",
        "gui.progress_panel",
        "gui.queue_panel",
        "gui.settings_panel",
        "gui.workers",
        "gui.filmstrip",
        "gui.setup_wizard",
        "cv2",
        "numpy",
        "requests",
        "skimage",
        "skimage.metrics",
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx_whisper",
        "huggingface_hub",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "test",
        "IPython",
        "jupyter",
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoBin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # No terminal window
    disable_windowed_traceback=False,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AutoBin",
)

app = BUNDLE(
    coll,
    name="AutoBin.app",
    icon="assets/AutoBin.icns",
    bundle_identifier="com.justinestrada.autobin",
    info_plist={
        "CFBundleName": "AutoBin",
        "CFBundleDisplayName": "AutoBin",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        "NSRequiresAquaSystemAppearance": False,
    },
)
