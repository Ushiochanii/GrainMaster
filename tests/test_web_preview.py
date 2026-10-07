from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from grainmaster.web_preview import display_image


def test_preview_preserves_coordinate_dimensions_and_invalidates_cache(tmp_path):
    source = tmp_path / "source.jpg"
    cv2.imwrite(str(source), np.zeros((1800, 2400, 3), np.uint8))
    app = FastAPI()

    @app.get("/image")
    def image(request: Request, preview: bool = False):
        return display_image(source, request, tmp_path / "cache", preview=preview)

    client = TestClient(app)
    result = client.get("/image?preview=true")
    assert result.headers["x-image-width"] == "2400"
    assert result.headers["x-image-height"] == "1800"
    decoded = cv2.imdecode(np.frombuffer(result.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (1200, 1600)
    assert "max-age=3600" in result.headers["cache-control"]
    etag = result.headers["etag"]
    assert client.get("/image?preview=true", headers={"If-None-Match": etag}).status_code == 304
    assert len(list((tmp_path / "cache").glob("*.jpg"))) == 1
    cv2.imwrite(str(source), np.full((1800, 2400, 3), 200, np.uint8))
    updated = client.get("/image?preview=true", headers={"If-None-Match": etag})
    assert updated.status_code == 200
    assert updated.headers["etag"] != etag
    full = client.get("/image")
    assert cv2.imdecode(np.frombuffer(full.content, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (1800, 2400)
