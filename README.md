# GrainMaster

GrainMaster is a local seed-phenotyping workstation for calibrated photographs containing
blue dishes, a metric ruler, and a 24-patch ColorChecker. It combines automatic calibration,
YOLO26 instance segmentation, seed-level morphology and color measurement, manual review,
and CSV export in a browser-based workbench.

## What it does

1. Detects the metric ruler and estimates physical scale.
2. Detects the ColorChecker and applies color correction.
3. Locates blue dishes and crops each dish.
4. Segments individual seeds with a bundled YOLO26s-seg model at 1024 px.
5. Measures seed length, width, area, shape traits, and CIELAB color.
6. Supports manual confirm/exclude review in the web workbench.
7. Exports seed-level and dish-level CSV results.

The original input photograph is not modified.

## Quick start

Requirements:

- Python 3.11 or newer
- Windows, macOS, or Linux
- A modern browser
- CPU inference works by default; GPU support depends on the local PyTorch/Ultralytics setup

Install:

    python -m pip install -e ".[web,model]"

On CPU-only Linux systems, install a CPU-only PyTorch build appropriate for your platform
before the command above. This avoids pulling unnecessary CUDA runtime packages.

Launch:

    python scripts/run_web_ui.py

Then open:

    http://127.0.0.1:8765

An example photograph is bundled in the workbench. You can also import your own photographs.

## Model

The release includes weights/seed_v1_yolo26s_seg.pt.

The default configuration is configs/workbench_yolo26.yaml and uses:

- YOLO26s-seg
- image size: 1024
- confidence threshold: 0.25
- class: seed

## Paper-label recognition

Paper-label recognition is optional and disabled by default.

The Settings panel supports DeepSeek, OpenAI-compatible endpoints, and custom compatible
providers. API credentials are supplied by the user and are stored separately from the normal
settings file. No API key is included in this repository.

Environment variables can also be used:

    GRAINMASTER_LABEL_API_KEY
    DEEPSEEK_API_KEY

## Output and local state

Runtime data is written under artifacts/ and data/processed/web_uploads/.
These paths are ignored by Git. The application keeps review state, settings, generated
previews, and analysis results locally.

## Repository layout

    configs/               Runtime configuration
    docs/                  User/developer documentation
    scripts/               Launch and training utilities
    src/grainmaster/       Core Python package
    tests/                 Automated tests
    web/                   Browser workbench
    weights/               Bundled seed segmentation model

## Development

    python -m pip install -e ".[web,model,dev]"
    python -m pytest -q

## Notes

GrainMaster is research software. Calibration quality, segmentation quality, imaging geometry,
reference-chart condition, glare, and acquisition setup can affect measurements. Review
automatic instances before treating exported measurements as final biological observations.

## License

GrainMaster is released under the MIT License. See `LICENSE`.
