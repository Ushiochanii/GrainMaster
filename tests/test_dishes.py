from pathlib import Path

import cv2
import numpy as np
import pytest

from grainmaster.dishes import crop_dish, detect_dishes, check_dish_crop
from grainmaster.contracts import DishROI


def test_blue_circles_rows_and_crop_scale():
    image = np.full((700, 1000, 3), 90, dtype=np.uint8)
    centers = [(200, 200), (490, 170), (780, 205), (230, 510)]
    for center in centers:
        cv2.circle(image, center, 90, (200, 30, 20), -1)
        cv2.ellipse(image, center, (15, 5), 40, 0, 360, (100, 180, 220), -1)
    # A blue ColorChecker patch must not become a dish.
    cv2.rectangle(image, (850, 550), (875, 575), (200, 30, 20), -1)
    dishes = detect_dishes(image)
    assert len(dishes) == 4
    assert [d.dish_id for d in dishes] == [1, 2, 3, 4]
    for dish, center in zip(dishes, centers):
        assert np.allclose(dish.center, center, atol=1)
        assert dish.radius == pytest.approx(90, abs=2)
        x, y, w, h = dish.bbox
        crop = crop_dish(image, dish)
        assert crop.shape == (h, w, 3)
        assert np.array_equal(crop, image[y:y + h, x:x + w])


def test_no_blue_and_invalid_image():
    assert detect_dishes(np.zeros((300, 400, 3), np.uint8)) == []
    with pytest.raises(ValueError):
        detect_dishes(np.zeros((300, 400), np.uint8))


def test_crop_qc_flags_clipping_and_source_boundary():
    dish = DishROI(1, (100., 100.), 40., (50, 50, 100, 100),
                   ((100., 100.), (80., 80.), 0.), {'envelope_refined': True})
    assert check_dish_crop(dish, (300, 300, 3))['status'] == 'ok'
    dish.bbox = (80, 50, 70, 100)
    assert 'fitted_outline_clipped' in check_dish_crop(dish, (300, 300, 3))['flags']
    dish.bbox = (0, 0, 150, 150)
    assert 'source_image_edge' in check_dish_crop(dish, (300, 300, 3))['flags']
    dish.ellipse = ((100., 100.), (80., 55.), 0.)
    assert 'elongated_shape_review' in check_dish_crop(dish, (300, 300, 3))['flags']


@pytest.mark.parametrize("image_id, expected", [("0462", 4), ("0504", 5), ("0507", 5)])
def test_real_dishes_including_crowded_seed_broken_rims(image_id, expected):
    # 0504 needs wider hole closing; 0507 has seeds opening the blue rim and
    # neighboring dishes that join under wide closing. Keep both regressions.
    path = Path(__file__).resolve().parents[1] / f"data/raw/{image_id}.jpg"
    if not path.exists():
        pytest.skip("Local acceptance photo is not distributed with source")
    dishes = detect_dishes(cv2.imread(str(path)))
    assert len(dishes) == expected
    for index, dish in enumerate(dishes):
        assert all(np.linalg.norm(np.subtract(dish.center, other.center)) > dish.radius
                   for other in dishes[index + 1:])


@pytest.mark.parametrize('image_id,dish_id', [('0484', 4), ('0501', 1),
                                           ('0501', 2), ('0503', 5), ('0509', 1)])
def test_reported_truncated_crops_recover_full_blue_envelope(image_id, dish_id):
    path = Path(__file__).resolve().parents[1] / f'data/raw/{image_id}.jpg'
    if not path.exists():
        pytest.skip('Local source photo unavailable')
    image = cv2.imread(str(path))
    dishes = detect_dishes(image)
    dish = next(d for d in dishes if d.dish_id == dish_id)
    assert dish.metadata.get('envelope_refined')
    assert dish.metadata['crop_qc']['status'] == 'ok'
    # Include the complete ellipse with margin, rather than a seed-separated
    # fragment. In these photos the recovered surface is nearly circular.
    assert min(dish.ellipse[1]) / max(dish.ellipse[1]) > .85
    (cx, cy), (aa, bb), degrees = dish.ellipse
    theta = np.deg2rad(degrees)
    ex = np.hypot(aa / 2 * np.cos(theta), bb / 2 * np.sin(theta))
    ey = np.hypot(aa / 2 * np.sin(theta), bb / 2 * np.cos(theta))
    x, y, w, h = dish.bbox
    assert x < cx - ex and x + w > cx + ex
    assert y < cy - ey and y + h > cy + ey
    # Almost all blue pixels nearest this dish within a broad local circle
    # must be in its crop. This independently catches the former clipped arc.
    blue = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
                       (95, 75, 40), (140, 255, 255))
    yy, xx = np.nonzero(blue)
    points = np.column_stack((xx, yy))
    distance = np.linalg.norm(points[:, None] - np.asarray([d.center for d in dishes]), axis=2)
    local = points[(distance.argmin(axis=1) == dish_id - 1) &
                   (distance[:, dish_id - 1] < max(aa, bb) * .65)]
    included = ((local[:, 0] >= x) & (local[:, 0] < x + w) &
                (local[:, 1] >= y) & (local[:, 1] < y + h))
    assert included.mean() > .995
