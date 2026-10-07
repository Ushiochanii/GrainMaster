"""Read-only seed training monitor, adapted from the wheat-awn dashboard."""
from __future__ import annotations
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
PROGRESS = re.compile(r'^\s*(\d+)/(\d+)\s+([\d.]+)G\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+).*?:\s*\d+%.*?\b(\d+)/(\d+)\b')
ROOT = Path(__file__).resolve().parents[2]


def load_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def read_metrics(path):
    """Reuse the prior dashboard's finite-value, complete-CSV-row parsing."""
    try:
        text = path.read_text(encoding='utf-8-sig')
    except OSError:
        return []
    if not text.endswith('\n'):
        text = text.rsplit('\n', 1)[0] + '\n'
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            parsed = {k.strip(): float(v) for k, v in row.items() if k and v and math.isfinite(float(v))}
            if 'epoch' in parsed:
                rows.append(parsed)
        except (TypeError, ValueError):
            continue
    return rows


def tail_log(path):
    try:
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 64000))
            text = handle.read().decode('utf-8', errors='replace')
        lines = ANSI.sub('', text).replace('\r', '\n').splitlines()
        return [line for line in lines if line.strip()]
    except OSError:
        return []


def supervisor_alive(pid, platform='wsl'):
    # Fixed executable and integer-only PID; do not accept commands from files.
    if not isinstance(pid, int) or pid < 1:
        return None
    if platform == 'windows':
        if os.name != 'nt':
            return None
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False if ctypes.get_last_error() == 87 else None
        try:
            code = wintypes.DWORD()
            return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
        finally:
            kernel.CloseHandle(handle)
    try:
        result = subprocess.run(['wsl', '-u', 'root', '--exec', 'bash', '-c',
                                 f'kill -0 {pid} 2>/dev/null'],capture_output=True,timeout=3)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return None


def training_folder(state):
    name = state.get('training_name', '')
    if re.fullmatch(r'seed_v1_(?:windows|rocm)(?:_smoke)?', name):
        return name
    return 'seed_v1_rocm_smoke' if state.get('stage') == 'smoke_training' else 'seed_v1_rocm'


def snapshot(project=ROOT, *, alive=None, now=None):
    project = Path(project)
    now = time.time() if now is None else now
    state_dir = project/'artifacts/models/seed_v1'
    state = load_json(state_dir/'run_status.json')
    stage = state.get('stage', 'not_started')
    smoke = stage == 'smoke_training'
    folder = project/'artifacts/models/training'/training_folder(state)
    rows = read_metrics(folder/'results.csv')
    lines = tail_log(state_dir/'training.log')
    expected_total = 1 if smoke else 150
    matches = [m for line in lines if (m := PROGRESS.match(line)) and int(m[2]) == expected_total]
    current = matches[-1] if matches else None
    completed = int(rows[-1]['epoch']) if rows else 0
    progress = dict(epoch=int(current[1]) if current else completed, total=expected_total,
                    completed_epochs=completed, batch=int(current[7]) if current else 0,
                    batches=int(current[8]) if current else 0,
                    gpu_memory_gb=float(current[3]) if current else None)
    latest_losses = dict(box=float(current[4]),seg=float(current[5]),cls=float(current[6])) if current else {}
    try:
        log_age = max(0, now - (state_dir/'training.log').stat().st_mtime)
    except OSError:
        log_age = None
    terminal = stage in {'failed','training_complete','not_started'}
    stale = not terminal and log_age is not None and log_age > 120
    effective = 'process_missing' if not terminal and alive is False else stage
    epoch_fraction = progress['batch']/progress['batches'] if progress['batches'] else 0
    done = max(completed, max(0,progress['epoch']-1)+epoch_fraction)
    # Early stopping may finish before the configured maximum; don't fabricate 150 epochs.
    progress['percent_of_maximum'] = min(100, done/expected_total*100)
    metadata = load_json(project/'data/processed/seed_yolo_v1/metadata.json')
    previews = [p.name for p in folder.glob('val_batch*_pred.jpg')]
    return dict(stage=effective, recorded_stage=stage, process_alive=alive,
                progress=progress, latest_losses=latest_losses, rows=rows, lines=lines[-35:],
                log_age_seconds=log_age, log_stale=stale, error=state.get('error'),
                gpu=state.get('gpu') or load_json(state_dir/'environment.json'),
                dataset=metadata.get('summary',{}), previews=sorted(previews)[:3],
                run_name=folder.name, updated=now,
                retry_count=state.get('retry_count',0), retry_limit=state.get('retry_limit',3),
                retry_reason=state.get('reason'),
                recovery_enabled=state.get('stall_timeout_seconds') is not None,
                stall_timeout_seconds=state.get('stall_timeout_seconds',120),
                memory_safe=state.get('memory_safe',False),
                best_checkpoint_available=(folder/'weights/best.pt').exists())


def create_monitor_app(project=ROOT):
    project = Path(project)
    app = FastAPI(title='GrainMaster Training Monitor')
    cache = {'time':0., 'alive':None, 'pid':None}

    @app.get('/api/training')
    def training():
        state = load_json(project/'artifacts/models/seed_v1/run_status.json')
        if time.time()-cache['time'] > 15 or cache['pid'] != state.get('pid'):
            cache.update(time=time.time(),alive=supervisor_alive(state.get('pid'), state.get('platform','wsl')),pid=state.get('pid'))
        return snapshot(project,alive=cache['alive'])

    @app.get('/api/training/preview/{name}')
    def preview(name: str):
        if not re.fullmatch(r'val_batch\d+_pred\.jpg',name):
            raise HTTPException(404)
        state=load_json(project/'artifacts/models/seed_v1/run_status.json')
        run=training_folder(state)
        path=project/'artifacts/models/training'/run/name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get('/')
    def index():
        return FileResponse(project/'web/training.html')

    app.mount('/static',StaticFiles(directory=project/'web'),name='static')
    return app
