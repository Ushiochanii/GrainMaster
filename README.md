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

# Installation

If you already use Python regularly, the short version is:

    git clone https://github.com/Ushiochanii/GrainMaster.git
    cd GrainMaster
    python -m venv .venv
    python -m pip install --upgrade pip
    python -m pip install -e ".[web,model]"
    python scripts/run_web_ui.py

Then open http://127.0.0.1:8765 in your browser.

For everyone else, use the step-by-step instructions below.

## Before you start

You need:

- Windows 10/11, macOS, or Linux.
- Python 3.11 or newer. Python 3.11 or 3.12 is recommended.
- About 5 GB of free disk space for Python packages and model dependencies.
- A modern browser such as Chrome, Edge, Firefox, or Safari.
- Internet access for the first installation.

GrainMaster uses CPU inference by default. You do not need an NVIDIA GPU, CUDA, ROCm, or any
other GPU setup just to run the application.

## Windows: step-by-step

### 1. Install Python

Download Python from https://www.python.org/downloads/

During installation, enable the option:

    Add python.exe to PATH

After Python is installed, open **PowerShell** from the Start menu and check:

    python --version

You should see Python 3.11 or newer.

If Windows says that `python` is not recognized, close PowerShell, open it again, and retry.
If it still fails, reinstall Python and make sure "Add python.exe to PATH" is enabled.

### 2. Download GrainMaster

The easiest method is to install Git and clone the repository:

    git clone https://github.com/Ushiochanii/GrainMaster.git
    cd GrainMaster

If you do not want to install Git:

1. Open the GrainMaster GitHub page.
2. Click **Code**.
3. Click **Download ZIP**.
4. Extract the ZIP.
5. Open PowerShell inside the extracted `GrainMaster` folder.

A quick way to open PowerShell in that folder is to click the File Explorer address bar,
type `powershell`, and press Enter.

### 3. Create a private Python environment

Run:

    python -m venv .venv

Activate it:

    .venv\Scripts\Activate.ps1

After activation, the beginning of the PowerShell prompt should contain `(.venv)`.

If PowerShell blocks activation with an execution-policy error, run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then activate the environment again:

    .venv\Scripts\Activate.ps1

This change applies only to the current PowerShell window.

### 4. Update pip

Run:

    python -m pip install --upgrade pip

### 5. Install CPU PyTorch

For a normal CPU-only Windows installation, run:

    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

This prevents pip from downloading unnecessary NVIDIA CUDA packages.

### 6. Install GrainMaster

Run:

    python -m pip install -e ".[web,model]"

The first installation may take several minutes because OpenCV, scientific Python libraries,
Ultralytics, and other dependencies are installed.

### 7. Start GrainMaster

Run:

    python scripts/run_web_ui.py

When the server has started, open:

    http://127.0.0.1:8765

Keep the PowerShell window open while using GrainMaster. Closing it stops the local server.

### 8. Try the bundled example

The workbench includes an example photograph.

Use it first to confirm that:

- the page loads;
- the example image appears;
- analysis can be started;
- measurement results are displayed;
- CSV export works.

After that, import your own photograph.

## macOS: step-by-step

### 1. Install Python

Install Python 3.11 or 3.12 from https://www.python.org/downloads/

Open Terminal and check:

    python3 --version

### 2. Download GrainMaster

    git clone https://github.com/Ushiochanii/GrainMaster.git
    cd GrainMaster

You can also download the repository as a ZIP from GitHub and open Terminal in the extracted
folder.

### 3. Create and activate a virtual environment

    python3 -m venv .venv
    source .venv/bin/activate

### 4. Install GrainMaster

    python -m pip install --upgrade pip
    python -m pip install torch torchvision
    python -m pip install -e ".[web,model]"

Apple Silicon Macs can use the PyTorch support available on the machine, but GrainMaster itself
does not require GPU acceleration.

### 5. Start GrainMaster

    python scripts/run_web_ui.py

Open:

    http://127.0.0.1:8765

## Linux: step-by-step

### 1. Make sure Python and venv are installed

On Ubuntu/Debian, for example:

    sudo apt update
    sudo apt install python3 python3-venv python3-pip git

Then check:

    python3 --version

### 2. Download GrainMaster

    git clone https://github.com/Ushiochanii/GrainMaster.git
    cd GrainMaster

### 3. Create and activate a virtual environment

    python3 -m venv .venv
    source .venv/bin/activate

### 4. Install CPU PyTorch

For a CPU-only machine:

    python -m pip install --upgrade pip
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Installing CPU PyTorch first avoids downloading several gigabytes of unnecessary CUDA packages.

### 5. Install GrainMaster

    python -m pip install -e ".[web,model]"

### 6. Start GrainMaster

    python scripts/run_web_ui.py

Open:

    http://127.0.0.1:8765

# Starting GrainMaster later

You do not need to reinstall GrainMaster every time.

Open a terminal in the GrainMaster folder, activate the environment, and start the server.

Windows:

    .venv\Scripts\Activate.ps1
    python scripts/run_web_ui.py

macOS/Linux:

    source .venv/bin/activate
    python scripts/run_web_ui.py

Then open http://127.0.0.1:8765.

# Updating GrainMaster

If you installed with Git:

    git pull
    python -m pip install -e ".[web,model]"

If you downloaded a ZIP, download the newest ZIP and replace the old program folder. Keep any
important exported CSV files separately.

# First-run settings

Open the Settings panel inside GrainMaster.

Recommended defaults:

- Device: CPU
- Model image size: 1024
- Paper-label recognition: Off

Paper-label recognition is optional. Only enable it if you want automatic transcription of
paper labels and have configured your own compatible API credentials.

# Troubleshooting

## `python` or `python3` is not recognized

Python is either not installed or not available in PATH. Install Python again and enable the
PATH option, then reopen the terminal.

## `No module named ...`

Make sure the virtual environment is activated, then run:

    python -m pip install -e ".[web,model]"

## PowerShell says scripts are disabled

Run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then:

    .venv\Scripts\Activate.ps1

## Port 8765 is already in use

Another GrainMaster instance or another local application may already be using that port.
Close the previous terminal window or process and start GrainMaster again.

## The browser does not open automatically

Open this address manually:

    http://127.0.0.1:8765

## Analysis is slow

CPU mode is the default and is expected to be slower than GPU inference. This is normal,
especially on large photographs. GPU acceleration is optional and depends on your local
PyTorch environment.

## Installation starts downloading huge NVIDIA/CUDA packages

Cancel the installation, activate the virtual environment, install CPU PyTorch first, and then
retry GrainMaster installation.

Windows/Linux CPU command:

    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Then:

    python -m pip install -e ".[web,model]"

# Model

The release includes `weights/seed_v1_yolo26s_seg.pt`.

The default configuration is `configs/workbench_yolo26.yaml` and uses:

- YOLO26s-seg
- image size: 1024
- confidence threshold: 0.25
- class: seed
- device: CPU by default

# Paper-label recognition

Paper-label recognition is optional and disabled by default.

The Settings panel supports DeepSeek, OpenAI-compatible endpoints, and custom compatible
providers. API credentials are supplied by the user and are stored separately from the normal
settings file. No API key is included in this repository.

Environment variables can also be used:

    GRAINMASTER_LABEL_API_KEY
    DEEPSEEK_API_KEY

# Output and local state

Runtime data is written under `artifacts/` and `data/processed/web_uploads/`.
These paths are ignored by Git. The application keeps review state, settings, generated
previews, and analysis results locally.

# Repository layout

    configs/               Runtime configuration
    docs/                  User/developer documentation
    scripts/               Launch and training utilities
    src/grainmaster/       Core Python package
    tests/                 Automated tests
    web/                   Browser workbench
    weights/               Bundled seed segmentation model

# Development

    python -m pip install -e ".[web,model,dev]"
    python -m pytest -q

# Notes

GrainMaster is research software. Calibration quality, segmentation quality, imaging geometry,
reference-chart condition, glare, and acquisition setup can affect measurements. Review
automatic instances before treating exported measurements as final biological observations.

# License

GrainMaster is released under the MIT License. See `LICENSE`.
