"""A real-photograph integration check, without claiming measurement ground truth."""
from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_acceptance_photograph_end_to_end():
    from grainmaster.pipeline import run_pipeline
    photograph = ROOT / 'data/raw/0462.jpg'
    if not photograph.exists():
        pytest.skip('Local acceptance photograph is not distributed in Git')
    summary = run_pipeline(photograph, ROOT / 'configs/prototype.yaml',
                           ROOT / 'artifacts/integration', backend='classical')
    out = ROOT / 'artifacts/integration/0462'
    assert summary['pipeline_status'] == 'ok'
    assert summary['dish_count'] == 4
    calibration = json.loads((out / 'calibration.json').read_text(encoding='utf-8'))
    spatial = calibration['spatial']
    assert spatial['pixels_per_mm'] > 0
    assert spatial['pixels_per_mm'] * spatial['mm_per_pixel'] == pytest.approx(1)
    color = calibration['color']
    assert color['mean_delta_e00_after'] < color['mean_delta_e00_before']
    card = calibration['card_scale']
    assert card['status'] == 'ok'
    assert card['tick_count'] >= 30
    assert calibration['scale_comparison']['status'] == 'disagreement'
    assert calibration['geometry_diagnostic']['applied_to_measurements'] is False
    assert calibration['geometry_diagnostic']['accepted'] is False
    assert (out / 'scale_crosscheck_overlay.jpg').exists()
    with (out / 'seed_instances.csv').open(encoding='utf-8-sig') as stream:
        seeds = list(csv.DictReader(stream))
    with (out / 'dish_summary.csv').open(encoding='utf-8-sig') as stream:
        dishes = list(csv.DictReader(stream))
    assert len(dishes) == 4
    assert len(seeds) == summary['seed_count_total']
    assert sum(int(row['seed_count']) for row in dishes) == len(seeds)
    assert all(int(row['seed_count']) > 0 for row in dishes)
    for row in seeds:
        for field in ['length_mm', 'width_mm', 'area_mm2', 'L', 'a', 'b']:
            assert math.isfinite(float(row[field]))
        assert float(row['length_mm']) >= float(row['width_mm']) > 0
        assert float(row['area_mm2']) > 0
        assert 0 <= float(row['L']) <= 100
        assert 'classical_unvalidated' in row['qc_flag']
    for dish in dishes:
        labels = cv2.imread(str(out / f"masks/dish_{int(dish['dish_id']):02d}.png"),
                            cv2.IMREAD_UNCHANGED)
        assert labels.dtype == np.uint16
        assert len(np.unique(labels)) - 1 == int(dish['seed_count'])
    preview = cv2.imread(str(out / 'preview.jpg'))
    assert preview.shape == cv2.imread(str(photograph)).shape


def test_failed_run_retains_diagnostic(tmp_path):
    from grainmaster.pipeline import run_pipeline, PipelineError
    with pytest.raises(PipelineError) as caught:
        run_pipeline(tmp_path / 'absent.jpg', ROOT / 'configs/prototype.yaml', tmp_path)
    summary = caught.value.summary
    assert summary['pipeline_status'] == 'failed'
    assert 'load_image' in summary['error_message']
    assert (tmp_path / 'absent/run_status.json').exists()
