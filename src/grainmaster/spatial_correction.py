"""Planar rectification using long-ruler ticks and observed object shapes.

This is a conditional planar model, not a calibrated lens-distortion model.
Segmentation stays in original photo coordinates; only measurement is rectified.
"""
from dataclasses import asdict, replace

import cv2
import numpy as np

from .reference_outline import detect_reference_outlines
from .reference_rectification import ruler_samples, project
from .spatial_preview import detect_dish_rim, fit_joint_candidate


def shape_descriptors(mask, length, width):
    """Contour-based shape ratios in the same plane as the reported geometry."""
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    area = sum(cv2.contourArea(c) for c in contours)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    hull = cv2.convexHull(np.concatenate(contours)) if contours else None
    hull_area = cv2.contourArea(hull) if hull is not None else 0
    return dict(aspect_ratio=float(length / width) if width > 0 else None,
                circularity=float(4 * np.pi * area / perimeter**2) if perimeter > 0 else None,
                solidity=float(area / hull_area) if hull_area > 0 else None)


def build_spatial_correction(image, color, spatial, dishes):
    shapes = detect_reference_outlines(image, spatial, color)
    rims = [detect_dish_rim(image, asdict(dish)) for dish in dishes]
    ticks, indices = ruler_samples(spatial)
    report = fit_joint_candidate(ticks, indices,
                                 np.asarray(shapes['card']['quadrilateral']),
                                 rims, image.shape)
    matrix = np.asarray(report['matrix'], dtype=float)
    scale = report['after']['pixels_per_mm']
    if not report['optimizer_success']:
        raise ValueError('Spatial correction optimizer did not converge: '
                         f"{report.get('optimizer_message', 'unknown solver failure')} "
                         f"(evaluations={report.get('optimizer_nfev', 'unknown')})")
    if (not np.isfinite(matrix).all() or not np.isfinite(scale) or scale <= 0):
        raise ValueError('Spatial correction fit did not produce a valid transform')
    report.update(applied_to_measurements=True, status='ok' if report['accepted'] else 'qc_warning',
                  rim_point_counts=[len(r) for r in rims],
                  segmentation_coordinates='original_photo',
                  measurement_coordinates='rectified_plane',
                  warnings=[] if report['accepted'] else ['Held-out shape/tick QC did not improve consistently.'])
    overlay = image.copy()
    for dish, points in zip(dishes, rims):
        for x, y in points.astype(int):
            cv2.circle(overlay, (x, y), 3, (0, 255, 255), -1)
        x, y = np.asarray(dish.center).astype(int)
        cv2.putText(overlay, f'D1{dish.dish_id} rim', (x-100, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    rectified_spatial = replace(spatial, pixels_per_mm=float(scale),
                               mm_per_pixel=float(1 / scale),
                               metadata={**spatial.metadata, 'measurement_plane': 'rectified'})
    return report, rectified_spatial, overlay



def rectify_instance_crop(mask, crop_origin, matrix, corrected_crop, output_size):
    """Warp a corrected native dish crop and mask into the same tight rectified crop."""
    mask = np.asarray(mask, dtype=np.uint8)
    if not mask.any():
        raise ValueError('Cannot rectify an empty seed instance')
    ys, xs = np.nonzero(mask)
    corners = np.array([[xs.min(), ys.min()], [xs.max()+1, ys.min()],
                        [xs.max()+1, ys.max()+1], [xs.min(), ys.max()+1]], float)
    corners += np.asarray(crop_origin)
    transformed = project(corners, np.asarray(matrix))
    x0, y0 = np.floor(transformed.min(0)-2).astype(int)
    x1, y1 = np.ceil(transformed.max(0)+2).astype(int)
    width, height = map(int, output_size)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Rectified instance lies outside the output canvas')
    source = np.array([[1., 0, crop_origin[0]], [0, 1., crop_origin[1]], [0, 0, 1.]])
    destination = np.array([[1., 0, -x0], [0, 1., -y0], [0, 0, 1.]])
    transform = destination @ matrix @ source
    warped_mask = cv2.warpPerspective(
        mask, transform, (int(x1-x0), int(y1-y0)), flags=cv2.INTER_NEAREST
    )
    if not warped_mask.any():
        raise ValueError('Spatial transform erased a seed instance')
    warped_image = cv2.warpPerspective(
        corrected_crop, transform, (int(x1-x0), int(y1-y0)), flags=cv2.INTER_LINEAR
    )
    return warped_mask.astype(bool), warped_image


def rectify_instance(mask, crop_origin, matrix, rectified_image):
    """Warp a native crop mask to a tight rectified crop, preserving binary edges."""
    mask = np.asarray(mask, dtype=np.uint8)
    if not mask.any():
        raise ValueError('Cannot rectify an empty seed instance')
    ys, xs = np.nonzero(mask)
    corners = np.array([[xs.min(), ys.min()], [xs.max()+1, ys.min()],
                        [xs.max()+1, ys.max()+1], [xs.min(), ys.max()+1]], float)
    corners += np.asarray(crop_origin)
    transformed = project(corners, np.asarray(matrix))
    x0, y0 = np.floor(transformed.min(0)-2).astype(int)
    x1, y1 = np.ceil(transformed.max(0)+2).astype(int)
    height, width = rectified_image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Rectified instance lies outside the output canvas')
    source = np.array([[1., 0, crop_origin[0]], [0, 1., crop_origin[1]], [0, 0, 1.]])
    destination = np.array([[1., 0, -x0], [0, 1., -y0], [0, 0, 1.]])
    warped = cv2.warpPerspective(mask, destination @ matrix @ source,
                                 (int(x1-x0), int(y1-y0)), flags=cv2.INTER_NEAREST)
    if not warped.any():
        raise ValueError('Spatial transform erased a seed instance')
    return warped.astype(bool), rectified_image[y0:y1, x0:x1]
