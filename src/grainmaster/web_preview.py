"""Cached display images; never used as measurement inputs."""
import hashlib
from pathlib import Path

import cv2
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response
from PIL import Image


def display_image(path: Path, request: Request, cache_root: Path, *, preview=False, versioned=True):
    stat = path.stat()
    identity = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:1600:82"
    version = hashlib.sha256(identity.encode()).hexdigest()
    headers = {"Cache-Control": "private, max-age=3600" if versioned else "private, max-age=0, must-revalidate"}
    if preview:
        with Image.open(path) as source:
            width, height = source.size
        headers.update({"X-Image-Width": str(width), "X-Image-Height": str(height)})
        cached = cache_root / f"{version}.jpg"
        if not cached.exists():
            image = cv2.imread(str(path))
            if image is None:
                raise HTTPException(500, "Could not generate display preview")
            scale = min(1., 1600 / max(width, height))
            if scale < 1:
                image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                raise HTTPException(500, "Could not encode display preview")
            cache_root.mkdir(parents=True, exist_ok=True)
            # Unique partial names keep concurrent requests from exposing partial JPEGs.
            from uuid import uuid4
            temporary = cache_root / f"{version}.{uuid4().hex}.tmp"
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(cached)
        path = cached
    headers["ETag"] = f'"{version}{"-preview" if preview else ""}"'
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    return FileResponse(path, media_type="image/jpeg" if preview else None, headers=headers)
