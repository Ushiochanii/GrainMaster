import io
import time
from threading import Event

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from grainmaster import web_server


@pytest.fixture
def project(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('segmentation: {backend: classical}\n')
    example = tmp_path / 'web/assets/example-photo.jpg'
    example.parent.mkdir(parents=True)
    Image.new('RGB', (30, 30), 'blue').save(example)
    raw = tmp_path / 'data/raw/raw.jpg'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(example.read_bytes())
    return tmp_path, config, tmp_path / 'state', example.read_bytes()


def upload(client, data, name='imported.jpg'):
    response = client.post('/api/upload', files={'file': (name, io.BytesIO(data), 'image/jpeg')})
    assert response.status_code == 200
    return response.json()['image_id']


def test_delete_cleans_photo_results_reviews_and_preview_and_survives_restart(project):
    root, config, state, data = project
    with TestClient(web_server.create_app(root, config, state)) as client:
        image_id = upload(client, data)
        other_id = upload(client, data, 'keep.jpg')
        entries = client.get('/api/images').json()['images']
        assert next(item for item in entries if item['image_id'] == image_id)['is_imported']
        assert client.get(f'/api/images/{image_id}/original?preview=true').status_code == 200
        cache = list((state / 'display_cache').glob('*.jpg'))
        assert len(cache) == 1
        folder = state / 'runs' / image_id
        folder.mkdir(parents=True)
        (folder / 'instances.json').write_text('{}')
        review = state / 'reviews' / f'{image_id}.json'
        review.write_text('{}')
        assert client.delete(f'/api/images/{image_id}').json() == {'deleted': True, 'image_id': image_id}
        assert not folder.exists()
        assert not review.exists()
        assert not cache[0].exists()
        assert not list((root / 'data/processed/web_uploads').glob(f'{image_id}.*'))
        assert client.get(f'/api/images/{image_id}/original').status_code == 404
        assert client.get(f'/api/results/{image_id}').status_code == 404
        assert client.delete(f'/api/images/{image_id}').status_code == 404
        assert client.get(f'/api/images/{other_id}/original').status_code == 200
    with TestClient(web_server.create_app(root, config, state)) as client:
        entries = client.get('/api/images').json()['images']
        assert image_id not in {item['image_id'] for item in entries}
        assert next(item for item in entries if item['image_id'] == other_id)['is_imported']


def test_delete_protects_example_raw_and_untrusted_origins(project):
    root, config, state, data = project
    with TestClient(web_server.create_app(root, config, state)) as client:
        for image_id in ('example', 'raw'):
            assert client.delete(f'/api/images/{image_id}').status_code == 403
            assert client.get(f'/api/images/{image_id}/original').status_code == 200
        image_id = upload(client, data)
        assert client.delete(f'/api/images/{image_id}', headers={'Origin': 'https://evil.example'}).status_code == 403
        assert client.get(f'/api/images/{image_id}/original').status_code == 200
        assert client.delete('/api/images/not-found').status_code == 404


def test_delete_rejects_active_analysis(project, monkeypatch):
    root, config, state, data = project
    started, release = Event(), Event()

    def pipeline(*args, **kwargs):
        started.set()
        release.wait(10)

    monkeypatch.setattr(web_server, 'run_pipeline', pipeline)
    app = web_server.create_app(root, config, state)
    with TestClient(app) as client:
        image_id = upload(client, data)
        job_id = client.post('/api/process', json={'image_id': image_id}).json()['job_id']
        try:
            assert started.wait(5)
            assert client.delete(f'/api/images/{image_id}').status_code == 409
            assert client.get(f'/api/images/{image_id}/original').status_code == 200
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while client.get(f'/api/jobs/{job_id}').json()['status'] in {'queued', 'running'}:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert client.delete(f'/api/images/{image_id}').status_code == 200
        assert client.get(f'/api/jobs/{job_id}').status_code == 404


def test_delete_rejects_active_labels(project, monkeypatch):
    import json
    from grainmaster import paper_labels

    root, config, state, data = project
    config.write_text('segmentation: {backend: classical}\npaper_labels: {enabled: true}\n')
    started, release = Event(), Event()

    def recognize(*args, **kwargs):
        started.set()
        release.wait(10)
        return {'labels': []}

    monkeypatch.setattr(paper_labels, 'recognize_labels', recognize)
    with TestClient(web_server.create_app(root, config, state)) as client:
        image_id = upload(client, data)
        folder = state / 'runs' / image_id
        folder.mkdir()
        for name in ('dishes.json', 'calibration.json'):
            (folder / name).write_text('{}')
        (folder / 'instances.json').write_text(json.dumps({'backend': 'classical'}))
        (folder / 'seed_instances.csv').write_text('seed_id\n')
        assert client.post(f'/api/results/{image_id}/labels').status_code == 200
        try:
            assert started.wait(5)
            assert client.delete(f'/api/images/{image_id}').status_code == 409
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while client.get(f'/api/results/{image_id}/labels').json()['status'] in {'queued', 'running'}:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert client.delete(f'/api/images/{image_id}').status_code == 200
