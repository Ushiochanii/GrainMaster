import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from grainmaster.rounded_reference import fit_rounded_card, _rounded_contour


@pytest.mark.parametrize('projective', [False, True])
def test_noisy_rounded_rectangle(projective):
    width, height, radius = 1000., 600., 30.
    corners = np.float32([[0,0],[width,0],[width,height],[0,height]])
    target = (np.float32([[120,80],[1160,110],[1110,730],[150,680]]) if projective
              else np.float32([[120,80],[1100,279],[981,867],[1,668]]))
    h = cv2.getPerspectiveTransform(corners, target)
    expected = cv2.perspectiveTransform(_rounded_contour(width,height,radius)[:,None],h)[:,0]
    noise = np.random.default_rng(42).normal(0, 1.5, expected.shape)
    fitted = fit_rounded_card(dict(contour=expected+noise, quadrilateral=target))
    assert abs(cv2.contourArea(fitted['contour'])/cv2.contourArea(expected)-1)<.003
    scale = np.mean(np.linalg.norm(np.roll(target,-1,axis=0)-target,axis=1)/[width,height,width,height])
    assert abs(fitted['fit']['radius_rectified_px']/(radius*scale)-1)<.1
    assert fitted['fit']['residual_p95_px']<4
    assert fitted['metadata']['smoothing_only']
    assert not fitted['metadata']['global_geometry_corrected']


def test_invalid_contour():
    with pytest.raises(ValueError):fit_rounded_card(dict(contour=[[0,0],[1,1]]))


def test_real_card_fit():
    path=Path('artifacts/prototype_p0/reference_shapes/0462.json')
    if not path.exists():pytest.skip('Local outline audit unavailable')
    outline=json.loads(path.read_text())['outlines']['card']
    result=fit_rounded_card(outline)
    assert abs(result['fit']['area_change_percent'])<2
    assert result['fit']['residual_p95_px']<8
    assert 2<result['fit']['radius_rectified_px']<40
