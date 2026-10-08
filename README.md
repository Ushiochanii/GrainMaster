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

## Installation

Requirements:

- Python 3.11 or newer
- Windows, macOS, or Linux
- A modern browser

Clone the repository and enter the project folder:

    git clone https://github.com/Ushiochanii/GrainMaster.git
    cd GrainMaster

Create a virtual environment:

    python -m venv .venv

Activate it.

Windows PowerShell:

    .venv\Scripts\Activate.ps1

macOS / Linux:

    source .venv/bin/activate

Install CPU PyTorch first:

    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Then install GrainMaster:

    python -m pip install -e ".[web,model]"

Installation only needs to be done once.

## Run GrainMaster

Whenever you want to use GrainMaster again, open a terminal in the project folder, activate the
virtual environment, and run:

    python scripts/run_web_ui.py

Then open:

    http://127.0.0.1:8765

GrainMaster uses CPU inference by default. The bundled example image can be used to verify that
the installation is working.

## Update an existing installation

Stop GrainMaster with `Ctrl+C` in the terminal running it. Open a terminal in your existing
`GrainMaster` folder and activate the same virtual environment used for installation:

Windows PowerShell:

    .venv\Scripts\Activate.ps1

macOS / Linux:

    source .venv/bin/activate

If you installed by cloning this repository, update and restart with:

    git pull --ff-only origin main
    python -m pip install -e ".[web,model]"
    python scripts/run_web_ui.py

Reopen `http://127.0.0.1:8765` and hard-refresh the browser (`Ctrl+F5` on Windows/Linux,
or `Cmd+Shift+R` on macOS). Imported photos now have a **Delete** button on their thumbnails.
Existing imported photos are supported; no data migration or model retraining is required.
You do not need to reinstall CPU PyTorch for this update.

The update keeps your imported photos, settings, reviews, and analysis results in
`data/processed/web_uploads/` and `artifacts/`. Back up those folders before updating valuable
research data. Do not delete your installation folder, run `git clean`, or reset local
changes to perform an update. If Git reports local changes or divergent branches, preserve
your changes and resolve that message before proceeding.

If you installed from **Download ZIP**, `git pull` will not work. Download the latest ZIP from
GitHub and extract it into a separate folder. Copy the updated `src/`, `web/`, `scripts/`,
`configs/`, and `pyproject.toml` into your existing installation, replacing matching source
files. Back up any customized configuration files first. Keep the existing `.venv/`,
`artifacts/`, and `data/` folders. Then activate the existing environment and run the two
`python` commands above to reinstall the package and restart.

## Manage imported photos

Click **Delete** in the upper-right corner of an imported photo's thumbnail, then confirm.
Deletion removes that imported photo, its analysis outputs, and its review records from
GrainMaster. It cannot be undone. The original file you selected from your computer is not
deleted. The bundled example and photos provided through `data/raw/` cannot be deleted
through this button.

Wait for analysis or paper-label recognition to finish before deleting a photo. When you
delete the currently open photo, the workbench automatically opens another available photo.

## Access through a public URL

For a reverse proxy or temporary HTTPS tunnel, set `GRAINMASTER_TRUSTED_ORIGINS` to the exact
public origin before starting GrainMaster. Multiple origins can be comma-separated.
For example, on macOS / Linux:

    GRAINMASTER_TRUSTED_ORIGINS=https://grainmaster.example.com python scripts/run_web_ui.py

Windows PowerShell:

    $env:GRAINMASTER_TRUSTED_ORIGINS = "https://grainmaster.example.com"
    python scripts/run_web_ui.py

This allows uploads, analysis, and deletion from the configured browser origin. It does not
create a public URL or provide authentication. Without this variable, the existing local
and Tailscale origin rules still apply. Public instances share their photo library and state
between visitors, including deletion access.

## Model

The release includes `weights/seed_v1_yolo26s_seg.pt`.

The default configuration is `configs/workbench_yolo26.yaml` and uses:

- YOLO26s-seg
- image size: 1024
- confidence threshold: 0.25
- class: seed
- device: CPU by default

## Paper-label recognition

Paper-label recognition is optional and disabled by default.

The Settings panel supports DeepSeek, OpenAI-compatible endpoints, and custom compatible
providers. API credentials are supplied by the user and are stored separately from the normal
settings file. No API key is included in this repository.

Environment variables can also be used:

    GRAINMASTER_LABEL_API_KEY
    DEEPSEEK_API_KEY

## Output and local state

Runtime data is written under `artifacts/` and `data/processed/web_uploads/`.
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
