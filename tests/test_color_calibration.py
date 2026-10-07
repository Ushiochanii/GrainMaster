import cv2
import numpy as np
import pytest
from grainmaster.color_calibration import (reference_colors, calibrate_color,
                                          apply_color_calibration, _decode, _encode)


def _synthetic_chart(rotate=False):
    reference, _ = reference_colors()
    observed = _encode(_decode(reference) * np.array([.75, .85, .65]) + .015)
    chart = np.zeros((350, 520, 3), np.uint8)
    for index, rgb in enumerate(observed):
        row, col = divmod(index, 6)
        cv2.rectangle(chart, (20+col*80, 20+row*80), (80+col*80, 80+row*80),
                      tuple(int(v) for v in np.rint(rgb[::-1]*255)), -1)
    if rotate:
        chart = chart[::-1, ::-1]
    canvas = np.full((650, 900, 3), 140, np.uint8)
    canvas[160:510, 170:690] = chart
    return canvas


@pytest.mark.parametrize('rotate', [False, True])
def test_detect_chart_and_remove_known_color_cast(rotate):
    image = _synthetic_chart(rotate)
    result = calibrate_color(image)
    assert result.metadata['geometry_inliers'] >= 18
    assert len(result.metadata['patch_centers']) == 24
    assert result.mean_delta_e00_after < 1
    assert result.mean_delta_e00_after < result.mean_delta_e00_before
    corrected = apply_color_calibration(image, result)
    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint8
    assert result.metadata['orientation'] == ('180' if rotate else 'identity')


def test_no_chart_fails_explicitly():
    with pytest.raises(ValueError, match='ColorChecker'):
        calibrate_color(np.full((600, 800, 3), 128, np.uint8))


def test_reference_conversion_has_standard_patch_order():
    colors, names = reference_colors()
    assert colors.shape == (24, 3)
    assert names[0] == 'dark skin'
    assert names[18].startswith('white')
    assert np.all((colors >= 0) & (colors <= 1))
    assert colors[18].mean() > colors[23].mean()


def test_color_transform_is_identical_at_different_positions_and_chunks():
    from grainmaster.contracts import ColorCalibration
    image = np.zeros((300, 40, 3), np.uint8)
    positions = [(0, 0), (127, 11), (128, 21), (299, 39)]
    for y, x in positions:
        image[y, x] = (85, 125, 190)
    matrix = np.array([[.8, .03, .02], [.02, .9, .04], [.05, .01, .7], [.01, -.01, .02]])
    calibration = ColorCalibration(matrix, (0, 0, 1, 1), 0., 0.)
    output = apply_color_calibration(image, calibration)
    values = [output[y, x].tolist() for y, x in positions]
    assert all(value == values[0] for value in values)
    assert values[0] != [85, 125, 190]


def test_reference_version_is_explicit_and_must_have_24_patches():
    rgb, names = reference_colors('ColorChecker24 - After November 2014')
    assert rgb.shape == (24, 3)
    assert len(names) == 24
    with pytest.raises(ValueError, match='24-patch'):
        reference_colors('ColorCheckerSG - After November 2014')


def test_root_polynomial_calibration_and_global_application():
    from grainmaster.color_models import transform_linear
    image = _synthetic_chart()
    result = calibrate_color(image, model='root_polynomial2')
    assert result.matrix.shape == (6, 3)
    assert result.metadata['model'] == 'root_polynomial2'
    assert np.isfinite(result.metadata['leave_one_patch_out_mean_delta_e00'])
    corrected = apply_color_calibration(image, result)
    expected = np.rint(_encode(transform_linear(_decode(image[:, :, ::-1]/255.),
                         result.matrix, 'root_polynomial2'))[:, :, ::-1]*255).astype(np.uint8)
    np.testing.assert_array_equal(corrected, expected)
    assert result.mean_delta_e00_after < result.mean_delta_e00_before
