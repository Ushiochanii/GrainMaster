"""Secondary card-edge millimetre scale, measured in original image pixels.

Fine printed ticks are assumed 1 mm, based on the visible `mm` label on this
card. This is a local cross-check, not a camera/lens or full-plane calibration.
"""
import cv2
import numpy as np
from .contracts import ColorCalibration


def _fit_tick_positions(positions):
    positions = np.sort(np.asarray(positions, dtype=float))
    if len(positions) < 12:
        raise ValueError('card scale: fewer than 12 tick candidates')
    gaps = np.diff(positions)
    spacing = float(np.median(gaps))
    if spacing < 3:
        raise ValueError('card scale: tick spacing too small')
    indices = np.rint((positions - positions[0]) / spacing)
    keep = np.ones(len(positions), dtype=bool)
    for _ in range(5):
        slope, intercept = np.polyfit(indices[keep], positions[keep], 1)
        errors = positions - (slope * indices + intercept)
        keep = abs(errors) < max(1.5, slope * .18)
        if keep.sum() < 12:
            raise ValueError('card scale: inconsistent tick lattice')
        indices = np.rint((positions - intercept) / slope)
    slope, intercept = np.polyfit(indices[keep], positions[keep], 1)
    residual = float(np.sqrt(np.mean((positions[keep] - slope * indices[keep] - intercept) ** 2)))
    if keep.sum() / len(positions) < .7 or residual > slope * .16:
        raise ValueError('card scale: poor tick lattice fit')
    return float(slope), residual, keep, indices


def detect_card_scale(image_bgr, color: ColorCalibration):
    grid = np.asarray(color.metadata['patch_centers'], np.float32).reshape(4, 6, 2)
    lattice = np.array([[c, r] for r in range(4) for c in range(6)], np.float32)
    homography, _ = cv2.findHomography(lattice, grid.reshape(-1, 2), 0)
    step = float(np.median(np.linalg.norm(np.diff(grid, axis=0), axis=2)))
    candidates = []
    for side, limits in [('column_0_outer', (-1.25, -.45)), ('column_5_outer', (5.45, 6.25))]:
        xs = np.arange(*limits, 1 / step)
        ys = np.arange(-.35, 3.10, 1 / step)
        xx, yy = np.meshgrid(xs, ys)
        local = np.dstack([xx, yy]).astype(np.float32)
        mapped = cv2.perspectiveTransform(local.reshape(1, -1, 2), homography).reshape(*xx.shape, 2)
        strip = cv2.remap(image_bgr, mapped[:, :, 0], mapped[:, :, 1], cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 165)).astype(np.uint8)
        _, _, stats, centers = cv2.connectedComponentsWithStats(white, 8)
        ticks = []
        for stat, center in zip(stats[1:], centers[1:]):
            _, top, w, h, area = stat
            if top <= 1 or top + h >= len(ys) - 1:
                continue
            if .075 * step < w < .42 * step and 1 <= h < .065 * step and w / h > 2.0 and area / (w * h) > .4:
                ticks.append(center)
        if len(ticks) < 12:
            continue
        ticks = np.asarray(ticks)
        # Order the white stroke components along the scale.
        order = np.argsort(ticks[:, 1])
        ticks = ticks[order]
        y_positions = ticks[:, 1]
        # Evaluate spacing in the original photo, never report warped-strip units.
        fixed_x = float(np.median(ticks[:, 0]))
        query = np.column_stack([np.full(len(ticks), xs[0] + fixed_x / step), ys[0] + y_positions / step]).astype(np.float32)
        points = cv2.perspectiveTransform(query[None], homography)[0]
        direction = points[-1] - points[0]
        direction /= np.linalg.norm(direction)
        projected = points @ direction
        try:
            ppm, residual, keep, indices = _fit_tick_positions(projected)
        except ValueError:
            continue
        confidence = float(keep.mean() * max(0, 1 - residual / (ppm * .2)))
        good = points[keep]
        result = dict(status='ok', tick_spacing_mm=1.0, pixels_per_mm=ppm, mm_per_pixel=1 / ppm, confidence=confidence,
                      residual=residual, tick_count=int(keep.sum()), tick_points=good.tolist(),
                      tick_indices=indices[keep].astype(int).tolist(), center=good.mean(0).tolist(),
                      direction=direction.tolist(), metadata=dict(side=side,
                      fine_tick_spacing_mm=1.0, physical_spacing_assumed=True,
                      unit_evidence='Visible mm label on physical card; not OCR verified for each image',
                      method='white elongated tick components; robust lattice fit in original pixel coordinates',
                      scope='local secondary reference only; no global distortion correction',
                      candidate_tick_count=len(ticks),
                      confidence_note='Heuristic lattice consistency, not probability or absolute metric accuracy'))
        candidates.append(result)
    if not candidates:
        raise ValueError('card scale: no reliable white millimetre tick strip beside chart')
    return max(candidates, key=lambda r: r['tick_count'] * r['confidence'])


def draw_card_scale_overlay(image_bgr, result):
    out = image_bgr.copy()
    for point in result['tick_points']:
        cv2.circle(out, tuple(np.rint(point).astype(int)), 4, (0, 0, 255), 2)
    points = np.rint(result['tick_points']).astype(int)
    cv2.line(out, tuple(points[0]), tuple(points[-1]), (0, 255, 0), 2)
    center = np.rint(result['center']).astype(int)
    cv2.putText(out, f"Card scale {result['pixels_per_mm']:.4f} px/mm ({result['tick_count']} ticks)",
                tuple(center + [-600, -350]), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
    return out


