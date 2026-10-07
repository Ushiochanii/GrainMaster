import cv2
import numpy as np
import pytest

from grainmaster.contracts import SpatialCalibration
from grainmaster.phenotyping import aggregate_dishes, measure_seed


def measure(mask, scale=.1, image=None):
    if image is None:
        image = np.full((*mask.shape, 3), 255, np.uint8)
    spatial = SpatialCalibration(1 / scale, scale, (0, 0, 1, 1), 0, 1)
    return measure_seed(mask, image, spatial, image_id='synthetic', dish_id=1,
                        seed_id=1, seg_confidence=.8)


@pytest.mark.parametrize('angle', [0, 35, 90])
def test_known_ellipse_axes_area_and_scale(angle):
    mask = np.zeros((220, 220), np.uint8)
    cv2.ellipse(mask, (110, 110), (60, 25), angle, 0, 360, 1, -1)
    traits = measure(mask)
    assert traits.length_mm == pytest.approx(12., abs=.15)
    assert traits.width_mm == pytest.approx(5., abs=.15)
    assert traits.area_mm2 == pytest.approx(np.pi * 6 * 2.5, rel=.04)
    assert traits.area_mm2 == pytest.approx(mask.sum() * .01)
    scaled = measure(mask, scale=.2)
    assert scaled.length_mm == pytest.approx(2 * traits.length_mm)
    assert scaled.area_mm2 == pytest.approx(4 * traits.area_mm2)
    assert traits.L == pytest.approx(100., abs=.1)
    assert traits.a == pytest.approx(0., abs=.1)
    assert traits.b == pytest.approx(0., abs=.1)
    assert traits.qc_flag == 'ok'


def test_color_excludes_boundary_contamination():
    mask = np.zeros((50, 50), np.uint8)
    cv2.circle(mask, (25, 25), 15, 1, -1)
    image = np.zeros((50, 50, 3), np.uint8)
    image[:] = (255, 0, 0)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    image[eroded > 0] = (0, 0, 255)
    traits = measure(mask, image=image)
    assert traits.L == pytest.approx(53.24, abs=.2)
    assert traits.a == pytest.approx(80.1, abs=.3)
    assert traits.b == pytest.approx(67.2, abs=.3)


def test_tiny_mask_is_flagged_and_empty_rejected():
    mask = np.zeros((5, 5), np.uint8)
    mask[2, 2] = 1
    traits = measure(mask)
    assert np.isnan(traits.length_mm)
    assert 'ellipse_unavailable' in traits.qc_flag
    assert 'color_erosion_empty' in traits.qc_flag
    with pytest.raises(ValueError, match='empty'):
        measure(np.zeros((5, 5), np.uint8))


def test_summary_preserves_empty_dishes():
    mask = np.zeros((100, 100), np.uint8)
    cv2.ellipse(mask, (50, 50), (20, 10), 20, 0, 360, 1, -1)
    traits = measure(mask)
    summary = aggregate_dishes([traits], [1, 2])
    assert summary[0]['seed_count'] == 1
    assert summary[0]['mean_length_mm'] == traits.length_mm
    assert summary[1] == dict(dish_id=2, seed_count=0, mean_length_mm=None,
                            mean_width_mm=None, mean_area_mm2=None)
