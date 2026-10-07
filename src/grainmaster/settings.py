"""Persistent local workbench preferences and secret handling."""
from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

DEFAULT_SETTINGS = {
    "general": {
        "default_view": "review",
        "default_image_layer": "color",
        "remember_last_photo": True,
    },
    "analysis": {
        "device": "cpu",
        "imgsz": 1024,
    },
    "review": {
        "default_filter": "all",
        "auto_advance": False,
        "show_masks": True,
        "show_ids": True,
    },
    "export": {
        "default_mode": "draft",
        "default_kind": "dishes",
        "decimal_precision": 2,
    },
    "integrations": {
        "paper_labels": {
            "enabled": False,
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-flash",
        }
    },
}

_ALLOWED = {
    ("general", "default_view"): {"review", "measurements"},
    ("general", "default_image_layer"): {"color", "original"},
    ("analysis", "device"): {"cpu", "gpu"},
    ("analysis", "imgsz"): {640, 800, 1024, 1280},
    ("review", "default_filter"): {"all", "ambiguous", "pending", "confirmed", "discarded"},
    ("export", "default_mode"): {"draft", "confirmed"},
    ("export", "default_kind"): {"dishes", "seeds"},
    ("export", "decimal_precision"): {2, 3, 4},
    ("integrations", "paper_labels", "provider"): {"deepseek", "openai-compatible", "custom"},
}


def _merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def settings_path(state_root):
    return Path(state_root) / "settings.json"


def secret_path(state_root):
    return Path(state_root) / "credentials" / "paper-label.key"


def load_settings(state_root, base_config=None):
    value = copy.deepcopy(DEFAULT_SETTINGS)
    base_config = base_config or {}
    model = base_config.get("model", {})
    paper = base_config.get("paper_labels", {})
    value["analysis"]["device"] = str(model.get("device", value["analysis"]["device"]))
    value["analysis"]["imgsz"] = int(model.get("imgsz", value["analysis"]["imgsz"]))
    value["integrations"]["paper_labels"]["enabled"] = bool(
        paper.get("enabled", value["integrations"]["paper_labels"]["enabled"]))
    value["integrations"]["paper_labels"]["model"] = str(
        paper.get("model", value["integrations"]["paper_labels"]["model"]))
    path = settings_path(state_root)
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                _merge(value, stored)
        except (OSError, ValueError):
            pass
    return value


def _validate(settings):
    for path, allowed in _ALLOWED.items():
        current = settings
        for key in path:
            current = current[key]
        if current not in allowed:
            raise ValueError(f"Invalid setting: {'.'.join(path)}")
    for path in [
        ("general", "remember_last_photo"),
        ("review", "auto_advance"),
        ("review", "show_masks"),
        ("review", "show_ids"),
        ("integrations", "paper_labels", "enabled"),
    ]:
        current = settings
        for key in path:
            current = current[key]
        if not isinstance(current, bool):
            raise ValueError(f"Invalid setting: {'.'.join(path)}")
    paper = settings["integrations"]["paper_labels"]
    for key, limit in [("base_url", 500), ("model", 160)]:
        value = paper.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"Invalid setting: integrations.paper_labels.{key}")
        paper[key] = value.strip().rstrip("/")
    if not paper["base_url"].startswith(("https://", "http://")):
        raise ValueError("API base URL must start with http:// or https://")
    return settings


def save_settings(state_root, updates, base_config=None):
    settings = load_settings(state_root, base_config)
    _merge(settings, updates)
    _validate(settings)
    path = settings_path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return settings


def load_api_key(state_root):
    for name in ("GRAINMASTER_LABEL_API_KEY", "DEEPSEEK_API_KEY"):
        key = os.environ.get(name, "").strip()
        if key:
            return key
    path = secret_path(state_root)
    if not path.is_file():
        return None
    if os.name == "nt":
        literal = str(path.resolve()).replace("'", "''")
        command = (
            f"$s=ConvertTo-SecureString ((Get-Content -Raw -LiteralPath '{literal}').Trim());"
            "$b=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($s);"
            "try {[Runtime.InteropServices.Marshal]::PtrToStringBSTR($b)} "
            "finally {[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b)}"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=15,
            env={k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def save_api_key(state_root, key):
    path = secret_path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = (key or "").strip()
    if not key:
        path.unlink(missing_ok=True)
        return
    if os.name == "nt":
        literal = str(path.resolve()).replace("'", "''")
        env = dict(os.environ)
        env["GRAINMASTER_SECRET_INPUT"] = key
        command = (
            "$s=ConvertTo-SecureString $env:GRAINMASTER_SECRET_INPUT -AsPlainText -Force;"
            f"$s | ConvertFrom-SecureString | Set-Content -NoNewline -LiteralPath '{literal}'"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=15, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            raise ValueError("Could not securely save the API key.")
    else:
        path.write_text(key, encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass


def public_settings(state_root, base_config=None):
    settings = load_settings(state_root, base_config)
    key = load_api_key(state_root)
    result = copy.deepcopy(settings)
    result["integrations"]["paper_labels"]["api_key_configured"] = bool(key)
    result["integrations"]["paper_labels"]["api_key_hint"] = (
        f"••••{key[-4:]}" if key and len(key) >= 4 else ("Configured" if key else "")
    )
    return result
