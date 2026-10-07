import cv2
import numpy as np
import pytest
from grainmaster.reference_metrics import measure_reference_outline


def rectangle(width=200, height=100):
    return np.float32([[0, 0], [width, 0], [width, height], [0, height]])


def dense_rectangle():
    mask = np.zeros((140, 240), np.uint8)
    cv2.rectangle(mask, (20, 20), (220, 120), 255, -1)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    return contour.reshape(-1, 2), rectangle() + 20


def test_rectangle_units_and_four_sides():
    contour, corners = dense_rectangle()
    measured = measure_reference_outline({'contour': contour, 'quadrilateral': corners}, 10)
    assert measured['long_side_mm'] == pytest.approx(20)
    assert measured['short_side_mm'] == pytest.approx(10)
    assert measured['area_mm2'] == pytest.approx(200)
    assert measured['perimeter_mm'] == pytest.approx(60)
    assert measured['side_lengths_mm'] == pytest.approx([20, 10, 20, 10])
    assert max(measured['opposite_edge_angle_deg']) < .001
    assert max(x['rms_deviation_mm'] for x in measured['edge_straightness']) < .001


def test_uniform_resize_keeps_physical_geometry():
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    contour = np.column_stack([100 * np.cos(theta), 50 * np.sin(theta)])
    a = measure_reference_outline({'contour': contour}, 10)
    b = measure_reference_outline({'contour': contour * 3}, 30)
    for key in ['long_side_mm', 'short_side_mm', 'area_mm2', 'perimeter_mm']:
        assert a[key] == pytest.approx(b[key], rel=1e-5)
    assert a['area_mm2'] == pytest.approx(np.pi * 10 * 5, rel=.001)


def test_rounded_boundary_area_and_perimeter_differ_from_rectangle():
    mask = np.zeros((140, 240), np.uint8)
    cv2.rectangle(mask, (40, 20), (200, 120), 255, -1)
    cv2.rectangle(mask, (20, 40), (220, 100), 255, -1)
    for center in [(40, 40), (200, 40), (200, 100), (40, 100)]:
        cv2.circle(mask, center, 20, 255, -1)
    # Hole doesn't affect the requested outer-boundary area.
    cv2.circle(mask, (100, 70), 10, 0, -1)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    measured = measure_reference_outline({'contour': contour}, 10)
    assert 195 < measured['area_mm2'] < 200
    assert 55 < measured['perimeter_mm'] < 60
    assert measured['raw_perimeter_px'] > measured['perimeter_px']


def test_curved_edge_is_not_confused_with_projective_straight_edge():
    top_x = np.linspace(0, 200, 201)
    top = np.column_stack([top_x, 8 * (1 - ((top_x - 100) / 100) ** 2)])
    right = np.column_stack([np.full(101, 200), np.linspace(0, 100, 101)])
    bottom = np.column_stack([top_x[::-1], np.full(201, 100)])
    left = np.column_stack([np.zeros(101), np.linspace(100, 0, 101)])
    measured = measure_reference_outline({'contour': np.vstack([top, right, bottom, left]),
                                          'quadrilateral': rectangle()}, 10)
    edge = measured['edge_straightness'][0]
    assert edge['quadratic_bow_mm'] > .3
    assert edge['rms_deviation_mm'] > .1
    contour, corners = dense_rectangle()
    transform = np.float32([[1, .12, 0], [.05, 1, 0], [.0003, .0005, 1]])
    warped = cv2.perspectiveTransform(contour.astype(np.float32).reshape(-1, 1, 2), transform)
    warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), transform)
    projective = measure_reference_outline({'contour': warped, 'quadrilateral': warped_corners}, 10)
    assert max(e['rms_deviation_mm'] for e in projective['edge_straightness']) < .001
    assert max(projective['opposite_edge_angle_deg']) > 1


@pytest.mark.parametrize('ppm', [0, -1, np.nan, np.inf])
def test_invalid_scale_is_rejected(ppm):
    with pytest.raises(ValueError):
        measure_reference_outline({'contour': rectangle()}, ppm)
