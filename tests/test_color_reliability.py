import numpy as np
from grainmaster.color_reliability import background_tiles, describe_tiles, between_dishes


def test_uniform_background_excludes_seed_and_rim():
    image = np.full((240, 240, 3), [180, 80, 30], dtype=np.uint8)
    labels = np.zeros((240, 240), dtype=np.uint16)
    labels[90:130, 90:130] = 1
    image[labels > 0] = [30, 100, 200]
    tiles, valid = background_tiles(image, labels, (120, 120), 110)
    assert not valid[100, 100] and not valid[0, 0]
    assert np.allclose([t['rgb'] for t in tiles], [30/255, 80/255, 180/255])
    assert describe_tiles([t['rgb'] for t in tiles])['within_dish_tile_delta_e00_p90'] < 1e-8


def test_relative_range_has_explicit_denominator():
    result = between_dishes([dict(L=48, a=0, b=0, Y=.19), dict(L=52, a=0, b=0, Y=.21)])
    assert np.isclose(result['background_L_range_percent_of_mean'], 8)
    assert np.isclose(result['background_Y_range_percent_of_mean'], 10)
    assert result['background_pair_delta_e00_max'] > 0


def test_identical_surfaces_have_no_between_dish_difference():
    row = describe_tiles([[.1, .3, .7]]*8)
    result = between_dishes([row, row])
    assert result['background_pair_delta_e00_max'] == 0
    assert result['background_L_range_percent_of_mean'] == 0
