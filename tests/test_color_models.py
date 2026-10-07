import numpy as np
import pytest
from grainmaster.color_models import MODELS, features, fit_model, transform_linear


@pytest.mark.parametrize('model', MODELS)
def test_identity_colours(model):
    rgb = np.random.default_rng(7).uniform(.02, .9, (24, 3))
    fit = fit_model(rgb, rgb, model)
    assert np.allclose(transform_linear(rgb, fit, model), rgb, atol=1e-10)


@pytest.mark.parametrize('model', ['linear3', 'root_polynomial2'])
def test_exposure_scaling_and_black(model):
    rgb = np.random.default_rng(2).uniform(.01, .4, (24, 3))
    assert np.allclose(features(2*rgb, model), 2*features(rgb, model))
    fit = fit_model(rgb, rgb, model)
    assert np.allclose(transform_linear(np.zeros((1, 3)), fit, model), 0)


def test_linear_recovers_known_channel_mixing():
    rgb = np.random.default_rng(1).uniform(.01, .7, (24, 3))
    matrix = np.array([[.9, .1, 0], [0, 1.1, .1], [.1, 0, .8]])
    fit = fit_model(rgb, rgb@matrix, 'linear3')
    assert np.max(abs(transform_linear(rgb, fit, 'linear3')-rgb@matrix)) < .002
