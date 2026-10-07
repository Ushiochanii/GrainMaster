from types import SimpleNamespace
import cv2
import numpy as np
from grainmaster.geometry import assess_planar_rectification, warp_geometry


def fixture(perspective=True, wrong_pitch=False, flip=False):
    transform = np.array([[1., .02, 0], [.01, 1., 0], [.0002 if perspective else 0, .0001 if perspective else 0, 1.]])
    def project(p):
        return cv2.perspectiveTransform(np.array(p, float)[None], transform)[0]
    grid = np.array([[100+x*30, 80+y*30] for y in range(4) for x in range(6)])
    if wrong_pitch:
        grid[:, 1] = 80+(grid[:, 1]-80)*1.25
    centers = project(grid).reshape(4, 6, 2)
    if flip:
        centers = centers[::-1, ::-1]
    color = SimpleNamespace(metadata={'patch_centers':centers.reshape(-1, 2).tolist()})
    card = project([[300, 80+y*10] for y in range(12)])
    main = project([[30, 40+y*10] for y in range(30)])
    spatial = SimpleNamespace(pixels_per_mm=10., metadata={'tick_points':main.tolist()})
    scale = dict(status='ok', tick_points=card.tolist(), tick_indices=list(range(12)))
    return np.zeros((450, 500, 3), np.uint8), color, spatial, scale


def test_known_homography_held_out_ruler_improves():
    result = assess_planar_rectification(*fixture())
    assert result['accepted'], result
    assert result['relative_scale_disagreement_after'] < 1e-5
    assert result['long_ruler_after']['robust_cv'] < 1e-5
    assert warp_geometry(fixture()[0], result).shape == fixture()[0].shape


def test_flipped_reference_order_same_geometry():
    normal = assess_planar_rectification(*fixture())
    flipped = assess_planar_rectification(*fixture(flip=True))
    np.testing.assert_allclose(normal['matrix'], flipped['matrix'], atol=1e-6)


def test_no_improvement_remains_diagnostic():
    result = assess_planar_rectification(*fixture(perspective=False))
    assert not result['accepted']
    assert any('No meaningful' in s for s in result['failure_reasons'])


def test_inconsistent_grid_pitch_rejected_by_held_out_ruler():
    image, color, spatial, scale = fixture()
    scale['tick_indices'] = (np.arange(12)*.8).tolist()
    result = assess_planar_rectification(image, color, spatial, scale)
    assert not result['accepted']
    assert result['relative_scale_disagreement_after'] > .03


def test_missing_scale_rejected():
    image, color, spatial, _ = fixture()
    result = assess_planar_rectification(image, color, spatial, {})
    assert not result['accepted']
    assert result['failure_reasons']

