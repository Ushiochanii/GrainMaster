"""Budget exhaustion gets one bounded continuation, not silent acceptance."""
from types import SimpleNamespace

import numpy as np
import pytest

from grainmaster import spatial_preview


def observations():
    ticks = np.column_stack([np.full(30, 100.), np.arange(30) * 10. + 50.])
    card = np.array([[200, 200], [500, 200], [500, 400], [200, 400]], float)
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    rim = np.column_stack([np.cos(angles), np.sin(angles)]) * 70 + [600, 600]
    return ticks, np.arange(30), card, [rim, rim + [150, 0]], (1000, 1000, 3)


@pytest.mark.parametrize('first_status,expected_calls', [(0, 2), (-1, 1)])
def test_only_budget_exhaustion_retries(monkeypatch, first_status, expected_calls):
    calls = []
    real_solver = spatial_preview.least_squares

    def solver(fun, initial, **options):
        calls.append((np.array(initial), options))
        if len(calls) == 1:
            return SimpleNamespace(status=first_status, success=False, x=np.zeros(4),
                                   nfev=180, message='simulated stop', cost=1.)
        return real_solver(fun, initial, **options)

    monkeypatch.setattr(spatial_preview, 'least_squares', solver)
    report = spatial_preview.fit_joint_candidate(*observations())
    assert len(calls) == expected_calls
    assert report['optimizer_retried'] == (first_status == 0)
    assert report['optimizer_success'] == (first_status == 0)
    assert report['applied_to_measurements'] is False
    if first_status == 0:
        assert calls[1][1]['max_nfev'] == 720
        np.testing.assert_array_equal(calls[0][1]['bounds'], calls[1][1]['bounds'])
        np.testing.assert_array_equal(calls[1][0], np.zeros(4))
        assert report['optimizer_nfev'] > 180
