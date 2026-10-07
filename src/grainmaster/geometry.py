"""Conditional planar homography diagnostic, never a lens calibration.

Square, equally spaced chart patches and coplanarity are explicit assumptions.
The long ruler is held out of fitting and is used only to reject extrapolation.
"""
from pathlib import Path
import json
import cv2
import numpy as np


def _project(points, matrix):
    points = np.asarray(points, float).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _spacing(points, indices):
    points = np.asarray(points, float)
    steps = np.diff(np.asarray(indices, float))
    spacing = np.linalg.norm(np.diff(points, axis=0), axis=1) / steps
    spacing = spacing[np.isfinite(spacing) & (steps > 0)]
    if len(spacing) < 5:
        raise ValueError('At least six ordered ticks required')
    median = float(np.median(spacing))
    return dict(median_pixels_per_mm=median,
                robust_cv=float(1.4826 * np.median(abs(spacing-median)) / median),
                p10_pixels_per_mm=float(np.percentile(spacing, 10)),
                p90_pixels_per_mm=float(np.percentile(spacing, 90)),
                tick_count=len(points))


def assess_planar_rectification(image_bgr, color, spatial, card_scale):
    """Return JSON-compatible candidate and held-out checks; fitting uses card only.

    The card's fine ticks are assumed 1 mm. Even accepted results are conditional
    planar correction, not estimated camera lens distortion coefficients.
    """
    result = dict(status='diagnostic_only', accepted=False,
                  assumptions=['ColorChecker patch centers form a square equal-pitch 6x4 lattice',
                               'card fine ticks represent 1 mm',
                               'card, long ruler and seed measurement surfaces are coplanar'],
                  limitations=['No camera intrinsics or lens distortion coefficients are estimated',
                               'Chart-local fit extrapolates to full image; no independent horizontal scale',
                               'Tick units and physical chart pitch need independent confirmation',
                               'Patch centers already come from chart detector homography; chart fit RMS is not independent QC',
                               'Long ruler points were reconstructed from ruler ROI; spacing has pixel quantization'],
                  failure_reasons=[])
    try:
        points = np.asarray(color.metadata['patch_centers'], float).reshape(4, 6, 2)
        if np.mean(np.diff(points, axis=1)[..., 0]) < 0:
            points = points[:, ::-1]
        if np.mean(np.diff(points, axis=0)[..., 1]) < 0:
            points = points[::-1]
        pitch = float(np.median(np.r_[np.linalg.norm(np.diff(points, axis=1), axis=2).ravel(),
                                     np.linalg.norm(np.diff(points, axis=0), axis=2).ravel()]))
        lattice = np.array([[x, y] for y in range(4) for x in range(6)], float)
        destination = lattice * pitch + points[0, 0]
        matrix, _ = cv2.findHomography(points.reshape(-1, 2), destination, 0)
        if matrix is None:
            raise ValueError('Chart homography unavailable')
        card_points = np.asarray(card_scale['tick_points'], float)
        card_indices = np.asarray(card_scale['tick_indices'], float)
        unit = float(card_scale.get('tick_spacing_mm', 1.))
        card_indices = card_indices * unit
        card_before = _spacing(card_points, card_indices)
        card_candidate = _spacing(_project(card_points, matrix), card_indices)
        target = float(spatial.pixels_per_mm)
        factor = target / card_candidate['median_pixels_per_mm']
        origin = points[0, 0]
        scaling = np.array([[factor, 0, origin[0]*(1-factor)],
                            [0, factor, origin[1]*(1-factor)], [0, 0, 1.]])
        matrix = scaling @ matrix
        matrix = matrix / matrix[2, 2]
        main_points = np.asarray(spatial.metadata['tick_points'], float)
        gaps = np.linalg.norm(np.diff(main_points, axis=0), axis=1)
        # Existing long-ruler API retains ticks but does not retain lattice indices.
        main_indices = np.r_[0, np.cumsum(np.maximum(1, np.rint(gaps/np.median(gaps))))]
        before = _spacing(main_points, main_indices)
        after_points = _project(main_points, matrix)
        after = _spacing(after_points, main_indices)
        before_disagreement = abs(before['median_pixels_per_mm']/card_before['median_pixels_per_mm']-1)
        after_disagreement = abs(after['median_pixels_per_mm']/target-1)
        height, width = image_bgr.shape[:2]
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], float)
        denominators = np.column_stack([corners, np.ones(4)]) @ matrix[2]
        warped_corners = _project(corners, matrix)
        bounds_valid = bool(np.all(np.isfinite(warped_corners)) and np.all(denominators > .1)
                            and np.max(abs(warped_corners)) < 4 * max(height, width)
                            and .25 < factor < 4)
        result.update(matrix=matrix.tolist(), output_size=[width, height],
                      nominal_pixels_per_mm=target, card_before=card_before,
                      card_after=_spacing(_project(card_points, matrix), card_indices),
                      long_ruler_before=before, long_ruler_after=after,
                      relative_scale_disagreement_before=before_disagreement,
                      relative_scale_disagreement_after=after_disagreement,
                      chart_fit_rms_pixels=float(np.sqrt(np.mean(np.sum(
                          (_project(points.reshape(-1, 2), matrix)-_project(destination, scaling))**2, axis=1)))),
                      bounds_valid=bounds_valid, transformed_image_corners=warped_corners.tolist(),
                      long_ruler_transformed_tick_points=after_points.tolist(),
                      long_ruler_tick_indices_inferred=True)
        reasons = result['failure_reasons']
        if not bounds_valid:
            reasons.append('Unsafe full-image homography bounds or denominator')
        if card_scale.get('status') not in ('ok', 'success', 'accepted'):
            reasons.append('Auxiliary card scale is not accepted')
        if after_disagreement > .03:
            reasons.append('Held-out long-ruler scale disagrees with card by more than 3 percent')
        if after['robust_cv'] > max(.01, before['robust_cv'] * 1.15):
            reasons.append('Held-out long-ruler spacing variation worsens')
        if not (after_disagreement < before_disagreement - .005 or
                after['robust_cv'] < before['robust_cv'] * .9):
            reasons.append('No meaningful held-out improvement')
        if not reasons:
            result.update(status='accepted_conditional_planar', accepted=True)
    except (ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        result['failure_reasons'].append(str(exc))
    return result


def warp_geometry(image_bgr, result):
    """Render candidate for inspection; caller must gate scientific use on accepted."""
    if 'matrix' not in result or not result.get('bounds_valid', False):
        raise ValueError('No safe homography candidate available')
    return cv2.warpPerspective(image_bgr, np.asarray(result['matrix'], float),
                               tuple(result['output_size']))


def save_geometry_diagnostics(image_bgr, result, output_dir, image_id):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/f'{image_id}.json').open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    if result.get('bounds_valid'):
        candidate = warp_geometry(image_bgr, result)
        label = 'CONDITIONAL PLANAR' if result['accepted'] else 'CANDIDATE NOT VALIDATED'
        cv2.putText(candidate, label, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        for x, y in result['long_ruler_transformed_tick_points']:
            cv2.circle(candidate, (round(x), round(y)), 4, (0, 255, 255), -1)
        filename = f'{image_id}_candidate_' + ('conditional.jpg' if result['accepted'] else 'not_validated.jpg')
        cv2.imwrite(str(output_dir/filename), candidate)
