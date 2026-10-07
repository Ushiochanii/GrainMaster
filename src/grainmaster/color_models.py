"""Small comparison models for chart-based JPEG colour calibration.

RP2 follows Finlayson et al. (2015), DOI 10.1109/TIP.2015.2405336.
Input is assumed linear sRGB decoded from JPEG, not raw sensor RGB.
Regularization is fixed in advance, not selected on the held-out patches.
"""
import numpy as np

MODELS = ('affine_baseline', 'linear3', 'root_polynomial2')
RIDGE = 1e-3


def features(rgb, model):
    rgb = np.asarray(rgb, dtype=float)
    if model == 'affine_baseline':
        return np.concatenate([rgb, np.ones(rgb.shape[:-1]+(1,))], axis=-1)
    if model == 'linear3':
        return rgb
    if model == 'root_polynomial2':
        r, g, b = np.moveaxis(np.maximum(rgb, 0), -1, 0)
        return np.stack([r, g, b, np.sqrt(r*g), np.sqrt(r*b), np.sqrt(g*b)], axis=-1)
    raise ValueError(f'Unknown colour model: {model}')


def fit_model(observed_linear, reference_linear, model):
    X = features(observed_linear, model)
    if model == 'affine_baseline':
        return np.linalg.lstsq(X, reference_linear, rcond=None)[0]
    prior = np.zeros((X.shape[1], 3))
    prior[:3] = np.eye(3)
    penalty = RIDGE*np.trace(X.T@X)/X.shape[1]
    return np.linalg.solve(X.T@X+penalty*np.eye(X.shape[1]),
                           X.T@reference_linear+penalty*prior)


def transform_linear(rgb, coefficients, model):
    return features(rgb, model)@coefficients
