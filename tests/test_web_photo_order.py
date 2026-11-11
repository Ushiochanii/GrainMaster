import io
import json
import os

from fastapi.testclient import TestClient
from PIL import Image

from grainmaster import web_server


def test_photo_list_keeps_import_order_with_example_first(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('segmentation: {backend: classical}\n')
    example = tmp_path / 'web/assets/example-photo.jpg'
    example.parent.mkdir(parents=True)
    Image.new('RGB', (30, 30), 'blue').save(example)
    data = example.read_bytes()
    order = ['0228.jpg', '0043.jpg', '0213 - コピー.jpg', '0213.jpg']
    state = tmp_path / 'state'
    with TestClient(web_server.create_app(tmp_path, config, state)) as client:
        for name in order:
            response = client.post('/api/upload', files={'file': (name, io.BytesIO(data), 'image/jpeg')})
            assert response.status_code == 200
        names = [item['filename'] for item in client.get('/api/images').json()['images']]
    assert names == ['Example photo.jpg', *order]
    # The order survives a restart, which rebuilds the list from the upload folder.
    with TestClient(web_server.create_app(tmp_path, config, state)) as client:
        names = [item['filename'] for item in client.get('/api/images').json()['images']]
    assert names == ['Example photo.jpg', *order]


def test_uploads_without_recorded_time_fall_back_to_file_time(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('segmentation: {backend: classical}\n')
    uploads = tmp_path / 'data/processed/web_uploads'
    uploads.mkdir(parents=True)
    for image_id, name in [('upload_b', 'second.jpg'), ('upload_a', 'third.jpg'), ('upload_c', 'first.jpg')]:
        path = uploads / f'{image_id}.jpg'
        Image.new('RGB', (30, 30), 'blue').save(path)
        path.with_suffix('.json').write_text(json.dumps({'filename': name}))

    for stamp, image_id in enumerate(['upload_c', 'upload_b', 'upload_a'], start=1):
        os.utime(uploads / f'{image_id}.jpg', (stamp * 1000, stamp * 1000))
    with TestClient(web_server.create_app(tmp_path, config, tmp_path / 'state')) as client:
        names = [item['filename'] for item in client.get('/api/images').json()['images']]
    assert names == ['first.jpg', 'second.jpg', 'third.jpg']
