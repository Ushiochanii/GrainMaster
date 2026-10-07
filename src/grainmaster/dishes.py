"""Locate blue media in full photos, with ROI coordinates in original pixels."""
import cv2
import numpy as np

from .contracts import DishROI


def check_dish_crop(dish: DishROI, image_shape: tuple) -> dict:
    """Check fitted-outline coverage; this is not proof of physical completeness."""
    height, width = image_shape[:2]
    (cx, cy), (axis_a, axis_b), angle = dish.ellipse
    theta = np.deg2rad(angle)
    ex = float(np.hypot(axis_a / 2 * np.cos(theta), axis_b / 2 * np.sin(theta)))
    ey = float(np.hypot(axis_a / 2 * np.sin(theta), axis_b / 2 * np.cos(theta)))
    x, y, w, h = dish.bbox
    outline_contained = bool(x <= cx - ex and y <= cy - ey and
                             x + w >= cx + ex and y + h >= cy + ey)
    touches_image_edge = bool(x == 0 or y == 0 or x + w == width or y + h == height)
    axis_ratio = float(min(axis_a, axis_b) / max(axis_a, axis_b))
    flags = []
    if not outline_contained:
        flags.append('fitted_outline_clipped')
    if touches_image_edge:
        flags.append('source_image_edge')
    # An ellipse is expected under oblique viewing; this only flags a strong
    # departure for review, never forces a circle or discards a detection.
    if axis_ratio < .85:
        flags.append('elongated_shape_review')
    if not dish.metadata.get('envelope_refined'):
        flags.append('envelope_not_refined')
    return dict(status='review' if flags else 'ok', flags=flags,
                fitted_outline_contained=outline_contained,
                touches_image_edge=touches_image_edge, axis_ratio=axis_ratio,
                basis='fitted ellipse; source boundary; shape plausibility',
                guarantees_physical_completeness=False)


def detect_dishes(image_bgr: np.ndarray) -> list[DishROI]:
    """Find near-round blue components; IDs follow rows, then left to right.

    Detection is performed at a bounded resolution. Thresholds use image area,
    not a known dish count or fixed coordinates. The small blue checker patches
    are rejected by area and shape. All returned coordinates are full resolution.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")
    height, width = image_bgr.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale) if scale < 1 else image_bgr
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 75, 40), (140, 255, 255))
    # A larger closing fills seed gaps in crowded dishes, but can join adjacent
    # rims. Combining both scales retains isolated detections from either pass.
    contours = []
    for size in (7, 13, 19):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidates, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(candidates)
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < small.shape[0] * small.shape[1] * 0.003 or len(contour) < 5:
            continue
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        if min(axis_a, axis_b) / max(axis_a, axis_b) < 0.65:
            continue
        fill_ratio = area / (np.pi * axis_a * axis_b / 4)
        if fill_ratio < 0.65 or fill_ratio > 1.20:
            continue
        center = (float(cx / scale), float(cy / scale))
        axes = (float(axis_a / scale), float(axis_b / scale))
        if any(np.linalg.norm(np.subtract(center, other.center)) < other.radius
               for other in found):
            continue
        ellipse = (center, axes, float(angle))
        # Include a small rim margin, while preserving the fitted media ellipse.
        x, y, w, h = cv2.boundingRect(contour)
        margin = max(2, int(round(max(axis_a, axis_b) * 0.025 / scale)))
        left = max(0, int(np.floor(x / scale)) - margin)
        top = max(0, int(np.floor(y / scale)) - margin)
        right = min(width, int(np.ceil((x + w) / scale)) + margin)
        bottom = min(height, int(np.ceil((y + h) / scale)) + margin)
        found.append(DishROI(
            dish_id=0, center=center, radius=float(sum(axes) / 4),
            bbox=(left, top, right - left, bottom - top), ellipse=ellipse,
            metadata={"blue_fill_ratio": float(fill_ratio),
                      "crop_transform": {"offset_x": left, "offset_y": top, "scale": 1.0}},
        ))
    # Seed clusters can split the blue surface into fragments. Recover the
    # complete local blue envelope before assigning IDs or exporting crops.
    # Other dishes in this photo supply only a search-size prior, not a mask.
    if found:
        typical_diameter = float(np.median([max(d.ellipse[1]) for d in found]))
        hsv_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        blue_full = cv2.inRange(hsv_full, (95, 75, 40), (140, 255, 255))
        centers = np.asarray([d.center for d in found])
        for index, dish in enumerate(found):
            cx, cy = dish.center
            search_radius = max(typical_diameter, max(dish.ellipse[1])) * .80
            left, top = max(0, int(cx - search_radius)), max(0, int(cy - search_radius))
            right, bottom = min(width, int(cx + search_radius) + 1), min(height, int(cy + search_radius) + 1)
            yy, xx = np.nonzero(blue_full[top:bottom, left:right])
            points = np.column_stack((xx + left, yy + top)).astype(np.float32)
            distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            points = points[(distances.argmin(axis=1) == index) &
                            (distances[:, index] < search_radius)]
            if len(points) >= 5:
                hull = cv2.convexHull(points)
                envelope = np.zeros((bottom - top, right - left), np.uint8)
                cv2.fillConvexPoly(envelope, np.round(hull[:, 0] - [left, top]).astype(np.int32), 255)
                outlines, _ = cv2.findContours(envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour = max(outlines, key=cv2.contourArea)
                contour = contour + np.array([[[left, top]]])
                fitted = cv2.fitEllipse(contour)
                if (min(fitted[1]) / max(fitted[1]) >= .65 and
                        max(fitted[1]) <= typical_diameter * 1.40):
                    dish.ellipse = fitted
                    dish.center = tuple(map(float, fitted[0]))
                    dish.radius = float(sum(fitted[1]) / 4)
                    dish.metadata['envelope_refined'] = True
                    x, y, w, h = cv2.boundingRect(contour)
                    # Keep both the blue envelope and full rotated ellipse,
                    # including a 6% diameter margin for transparent dish rims.
                    theta = np.deg2rad(fitted[2])
                    a, b = np.asarray(fitted[1]) / 2
                    ex = float(np.hypot(a * np.cos(theta), b * np.sin(theta)))
                    ey = float(np.hypot(a * np.sin(theta), b * np.cos(theta)))
                    fx, fy = fitted[0]
                    margin = max(3, int(np.ceil(max(fitted[1]) * .06)))
                    l = max(0, int(np.floor(min(x, fx - ex))) - margin)
                    t = max(0, int(np.floor(min(y, fy - ey))) - margin)
                    r = min(width, int(np.ceil(max(x + w, fx + ex))) + margin)
                    bot = min(height, int(np.ceil(max(y + h, fy + ey))) + margin)
                    dish.bbox = (l, t, r - l, bot - t)
                    dish.metadata['crop_transform'] = {'offset_x': l, 'offset_y': t, 'scale': 1.0}
                    dish.metadata['crop_margin_px'] = margin
    rows: list[list[DishROI]] = []
    for dish in sorted(found, key=lambda d: (d.center[1], d.center[0])):
        if rows and abs(dish.center[1] - np.mean([d.center[1] for d in rows[-1]])) < (
            np.median([d.radius for d in rows[-1]]) * 0.9
        ):
            rows[-1].append(dish)
        else:
            rows.append([dish])
    ordered = [dish for row in rows for dish in sorted(row, key=lambda d: d.center[0])]
    for index, dish in enumerate(ordered, 1):
        dish.dish_id = index
        dish.metadata['crop_qc'] = check_dish_crop(dish, image_bgr.shape)
    return ordered


def crop_dish(image_bgr: np.ndarray, dish: DishROI) -> np.ndarray:
    """Return an independent crop without resampling (pixel scale is unchanged)."""
    x, y, width, height = dish.bbox
    return image_bgr[y:y + height, x:x + width].copy()


def draw_dish_overlay(image_bgr: np.ndarray, dishes: list[DishROI]) -> np.ndarray:
    """Draw fitted media outlines and stable IDs on a full-resolution image."""
    result = image_bgr.copy()
    thickness = max(2, round(max(image_bgr.shape[:2]) / 1000))
    for dish in dishes:
        if dish.ellipse:
            cv2.ellipse(result, dish.ellipse, (0, 255, 0), thickness)
        else:
            cv2.circle(result, tuple(map(round, dish.center)), round(dish.radius),
                       (0, 255, 0), thickness)
        x, y, _, _ = dish.bbox
        cv2.putText(result, f"Dish {dish.dish_id}", (x, max(30, y - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, thickness * 0.4),
                    (0, 255, 0), thickness, cv2.LINE_AA)
    return result
