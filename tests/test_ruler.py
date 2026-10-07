import cv2
import numpy as np
import pytest
from grainmaster.ruler import calibrate_ruler, _fit_ticks


def test_tick_fit_rejects_outliers_and_handles_missing_ticks():
    positions = np.arange(100, 2100, 10., dtype=float)
    positions = np.delete(positions, [20, 60, 120])
    positions[45] += 3
    step, residual, regularity, keep = _fit_ticks(positions)
    assert step == pytest.approx(10, abs=.02)
    assert residual < .2
    assert keep.sum() >= 190
    assert regularity > .9


def test_synthetic_metric_ruler_location():
    image = np.full((1600, 1000, 3), 65, np.uint8)
    cv2.rectangle(image, (320, 100), (410, 1500), (235,235,235), -1)
    for y in range(120,1480,8):
        cv2.line(image, (395,y), (410,y), (20,20,20), 1)
    for y in list(range(120,500,6)) + list(range(510,1480,13)):
        cv2.line(image, (320,y), (334,y), (20,20,20), 1)
    calibration = calibrate_ruler(image)
    # Fractional inch side changes subdivisions; metric side has uniform 1 mm ticks.
    assert calibration.pixels_per_mm == pytest.approx(8, abs=.1)
    assert calibration.mm_per_pixel == pytest.approx(.125, abs=.002)
    assert calibration.metadata['tick_count'] > 150


def test_no_ruler_is_explicit_failure():
    with pytest.raises(ValueError):
        calibrate_ruler(np.full((500,700,3),60,np.uint8))
