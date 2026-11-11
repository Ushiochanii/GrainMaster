import io

from fastapi.testclient import TestClient
from PIL import Image

from grainmaster import web_server
from grainmaster.web_server import natural_photo_key


def test_natural_photo_key_orders_by_line_number():
    names = ['0228.jpg', '1000.jpg', '0213 - コピー.jpg', '43.jpg', '0213.jpg', 'IMG_9.jpg', '０１００.jpg', 'IMG_10.jpg']
    assert sorted(names, key=natural_photo_key) == [
        '43.jpg', '０１００.jpg', '0213.jpg', '0213 - コピー.jpg', '0228.jpg', '1000.jpg', 'IMG_9.jpg', 'IMG_10.jpg',
    ]


def test_photo_list_is_sorted_by_filename_number_with_example_first(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('segmentation: {backend: classical}\n')
    example = tmp_path / 'web/assets/example-photo.jpg'
    example.parent.mkdir(parents=True)
    Image.new('RGB', (30, 30), 'blue').save(example)
    data = example.read_bytes()
    with TestClient(web_server.create_app(tmp_path, config, tmp_path / 'state')) as client:
        for name in ['0228.jpg', '0043.jpg', '0213 - コピー.jpg', '0213.jpg']:
            response = client.post('/api/upload', files={'file': (name, io.BytesIO(data), 'image/jpeg')})
            assert response.status_code == 200
        names = [item['filename'] for item in client.get('/api/images').json()['images']]
    assert names == ['Example photo.jpg', '0043.jpg', '0213.jpg', '0213 - コピー.jpg', '0228.jpg']
