import cv2
import numpy as np
import pytest
from grainmaster.contracts import ColorCalibration
from grainmaster.card_scale import detect_card_scale, _fit_tick_positions


def fixture_card(angle=0, ticks=True):
    image = np.full((1000, 1000, 3), 35, np.uint8)
    grid = np.array([[100 + 120 * c, 220 + 120 * r] for r in range(4) for c in range(6)], np.float32)
    if ticks:
        for i in range(41):
            if i in (7, 18):
                continue
            y = 180 + i * 10
            cv2.line(image, (798 if i % 10 else 787, y), (821, y), (235, 235, 235), 3)
    transform = cv2.getRotationMatrix2D((500, 500), angle, 1)
    image = cv2.warpAffine(image, transform, (1000, 1000), borderValue=(35, 35, 35))
    grid = cv2.transform(grid[None], transform)[0]
    color = ColorCalibration(np.zeros((4, 3)), (0, 0, 1, 1), 0, 0,
                             {'patch_centers': grid.tolist(), 'patch_size': 100})
    return image, color


@pytest.mark.parametrize('angle', [0, 25, 90, 180])
def test_original_metric_scale_rotations_and_missing_ticks(angle):
    image, color = fixture_card(angle)
    result = detect_card_scale(image, color)
    assert result['pixels_per_mm'] == pytest.approx(10, abs=.06)
    assert result['tick_count'] >= 35
    assert result['residual'] < .5
    assert result['metadata']['physical_spacing_assumed']
    assert np.linalg.norm(result['direction']) == pytest.approx(1)


def test_absent_ticks_fail():
    image, color = fixture_card(ticks=False)
    with pytest.raises(ValueError, match='no reliable'):
        detect_card_scale(image, color)


def test_robust_lattice_rejects_outlier():
    positions = np.sort(np.r_[np.arange(40) * 10., 154.])
    slope, residual, keep, indices = _fit_tick_positions(positions)
    assert slope == pytest.approx(10, abs=.01)
    assert not keep[16]
    assert residual < .01
