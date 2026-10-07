import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


def load_script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raster_polygon_keeps_seed_tip_and_rejects_disconnected_instance():
    pytest.importorskip('pycocotools')
    module=load_script('prepare_roboflow_coco')
    mask=np.zeros((100,100),np.uint8)
    cv2.ellipse(mask,(50,50),(30,10),35,0,360,1,-1)
    points,iou=module.polygon_from_mask(mask)
    assert iou==1
    reconstructed=np.zeros_like(mask)
    cv2.fillPoly(reconstructed,[points.astype(np.int32)],1)
    assert np.array_equal(mask,reconstructed)
    mask=np.zeros((100,100),np.uint8)
    mask[10:20,10:20]=1
    mask[70:80,70:80]=1
    with pytest.raises(ValueError,match='faithfully'):
        module.polygon_from_mask(mask)


def test_group_validator_quarantines_review_but_rejects_photo_leakage(tmp_path):
    module=load_script('train_yolo26')
    dataset=tmp_path/'dataset.yaml'
    dataset.write_text(yaml.safe_dump({'names':{0:'seed'}}))
    records=[dict(source_photo='a.jpg',split='train',instances=3,label_source='human'),
             dict(source_photo='b.jpg',split='val',instances=3,label_source='human'),
             dict(source_photo='a.jpg',split='review',instances=3,label_source='human')]
    metadata=dict(label_source='human',split_unit='whole_photo',records=records)
    path=tmp_path/'metadata.json'
    path.write_text(json.dumps(metadata))
    module.validate_dataset(dataset)
    records[-1]['split']='val'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='leakage'):
        module.validate_dataset(dataset)


def test_synthetic_human_polygons_convert_without_photo_leakage(tmp_path):
    raw = tmp_path / 'raw'
    annotations = tmp_path / 'annotations'
    raw.mkdir()
    annotations.mkdir()
    for index in range(2):
        image = np.full((640, 640, 3), 80, np.uint8)
        cv2.circle(image, (320, 320), 210, (180, 25, 25), -1)
        angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        points = np.column_stack([320 + 30 * np.cos(angle), 320 + 15 * np.sin(angle)])
        cv2.fillPoly(image, [np.round(points).astype(np.int32)], (40, 160, 220))
        cv2.imwrite(str(raw / f'synthetic_{index}.jpg'), image)
        (annotations / f'synthetic_{index}.json').write_text(json.dumps({
            'imagePath': f'synthetic_{index}.jpg',
            'shapes': [{'label': 'seed', 'shape_type': 'polygon', 'points': points.tolist()}]
        }), encoding='utf-8')
    output = tmp_path / 'prepared'
    metadata = load_script('prepare_seed_annotations').prepare(raw, annotations, output)
    assert metadata['label_source'] == 'human'
    assert metadata['split_unit'] == 'whole_photo'
    assert {r['split'] for r in metadata['records']} == {'train', 'val'}
    assert len(metadata['records']) == 2
    assert all(r['instances'] == 1 for r in metadata['records'])
    for row in metadata['records']:
        labels = (output / 'labels' / row['split'] / f"{row['image']}.txt").read_text()
        values = [float(v) for v in labels.split()]
        assert values[0] == 0
        assert len(values) >= 7
        assert all(0 <= v <= 1 for v in values[1:])
    config = yaml.safe_load((output / 'dataset.yaml').read_text())
    assert config['names'] == {0: 'seed'}
    load_script('train_yolo26').validate_dataset(output / 'dataset.yaml')
