"""Intermediate results must be available while analysis is still running."""
from threading import Event

import cv2
import numpy as np
from fastapi.testclient import TestClient

from grainmaster import web_server


def test_progress_events_are_incremental_and_assets_available_before_done(tmp_path, monkeypatch):
    raw = tmp_path / 'data/raw'
    raw.mkdir(parents=True)
    cv2.imwrite(str(raw / 'sample.jpg'), np.zeros((30, 30, 3), np.uint8))
    config = tmp_path / 'config.yaml'
    config.write_text('segmentation: {backend: classical}\n')
    ready, release = Event(), Event()

    def pipeline(path, config, output_root, progress_callback):
        folder = output_root / path.stem
        folder.mkdir(parents=True)
        progress_callback(dict(stage='ruler', status='running'))
        progress_callback(dict(stage='ruler', status='done', ruler={'pixels_per_mm': 10.}))
        cv2.imwrite(str(folder / '_progress_color.jpg'), np.zeros((20, 20, 3), np.uint8))
        progress_callback(dict(stage='color', status='done', image_asset='_progress_color.jpg'))
        ready.set()
        release.wait(5)
        progress_callback(dict(stage='export', status='done'))

    monkeypatch.setattr(web_server, 'run_pipeline', pipeline)
    with TestClient(web_server.create_app(tmp_path, config_path=config, state_root=tmp_path / 'state')) as client:
        job_id = client.post('/api/process', json={'image_id': 'sample', 'force': True}).json()['job_id']
        try:
            assert ready.wait(5)
            job = client.get(f'/api/jobs/{job_id}').json()
            assert job['status'] == 'running'
            assert [e['revision'] for e in job['events']] == [1, 2, 3]
            assert client.get(f'/api/jobs/{job_id}?after_revision=2').json()['events'][0]['stage'] == 'color'
            assert client.get(f'/api/jobs/{job_id}?after_revision=3').json()['events'] == []
            assert client.get(f'/api/jobs/{job_id}/asset/_progress_color.jpg').status_code == 200
            assert client.get(f'/api/jobs/{job_id}/asset/calibration.json').status_code == 404
            same = client.post('/api/process', json={'image_id': 'sample', 'force': True}).json()
            assert same['job_id'] == job_id
        finally:
            release.set()
