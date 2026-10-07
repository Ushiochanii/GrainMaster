"""Known transformations verify local-to-global mask geometry and metric axes."""
import cv2
import numpy as np
import pytest

from grainmaster.contracts import SpatialCalibration
from grainmaster.phenotyping import measure_seed
from grainmaster.spatial_correction import rectify_instance


@pytest.mark.parametrize('scale_x,scale_y', [(1., 1.), (1.5, 2.)])
def test_rectified_ellipse_geometry(scale_x, scale_y):
    mask = np.zeros((200, 240), np.uint8)
    cv2.ellipse(mask, (120, 100), (60, 30), 0, 0, 360, 1, -1)
    image = np.full((600, 600, 3), 128, np.uint8)
    matrix = np.diag([scale_x, scale_y, 1.])
    warped, crop = rectify_instance(mask, (50, 30), matrix, image)
    spatial = SpatialCalibration(10., .1, (0, 0, 1, 1), 0., 1.)
    traits = measure_seed(warped, crop, spatial, image_id='synthetic', dish_id=1,
                          seed_id=1, seg_confidence=1.)
    assert traits.length_mm == pytest.approx(12 * scale_x, abs=.15)
    assert traits.width_mm == pytest.approx(6 * scale_y, abs=.15)
    assert traits.area_mm2 == pytest.approx(mask.sum() * scale_x * scale_y * .01, rel=.035)
    if scale_x == scale_y == 1:
        assert warped.sum() == mask.sum()
        assert np.all(crop == 128)


def test_empty_mask_is_not_silently_measured():
    with pytest.raises(ValueError, match='empty'):
        rectify_instance(np.zeros((10, 10), np.uint8), (0, 0), np.eye(3),
                         np.zeros((30, 30, 3), np.uint8))
