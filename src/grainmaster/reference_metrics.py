"""Measurements of detected reference *outer boundaries*, without nominal sizes.

Area is the continuous polygon area enclosed by the contour (internal holes are
deliberately not subtracted). Perimeter uses Douglas-Peucker simplification at
0.1 mm by default to suppress pixel staircase inflation; raw perimeter is kept.
Min-area rectangle dimensions are oriented extents, not fitted physical edges.
A four-corner polygon, when supplied, is measured separately from the outline.
All mm outputs assume one isotropic scale: they do not themselves undo perspective.
"""
import cv2
import numpy as np


def _angle_difference(a, b):
    return float(abs((a - b + 90) % 180 - 90))


def _edge_diagnostics(contour, corners, ppm):
    """Fit each edge's central 80%, excluding rounded corner transitions.

    Points are assigned to their nearest corner-to-corner segment, then robust
    line fits describe orientation. Deviations retain all assigned points, so
    curvature is not silently discarded. quadratic_bow is a diagnostic fitted
    departure from a straight line, not a lens-distortion estimate.
    """
    starts = corners
    ends = np.roll(corners, -1, axis=0)
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    projections = np.stack([
        (contour - a) @ v / (length * length)
        for a, v, length in zip(starts, vectors, lengths)
    ], axis=1)
    nearest = np.stack([
        np.linalg.norm(contour - (a + np.clip(t, 0, 1)[:, None] * v), axis=1)
        for a, v, t in zip(starts, vectors, projections.T)
    ], axis=1).argmin(axis=1)
    diagnostics = []
    for index in range(4):
        t = projections[:, index]
        points = contour[(nearest == index) & (t >= .1) & (t <= .9)]
        result = {'edge_index': index, 'point_count': len(points)}
        if len(points) < 5:
            result['status'] = 'insufficient_points'
            diagnostics.append(result)
            continue
        vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER,
                                    0, .01, .01).ravel()
        direction = np.array([vx, vy], dtype=float)
        offset = points - np.array([x0, y0])
        along = offset @ direction
        normal = offset @ np.array([-vy, vx])
        rms = float(np.sqrt(np.mean(normal ** 2)))
        maximum = float(np.max(np.abs(normal)))
        p95 = float(np.percentile(np.abs(normal), 95))
        # Normalize to [-1,1] to keep the quadratic fit numerically conditioned.
        span = float(np.ptp(along))
        normalized = (along - np.mean(along)) / max(span / 2, 1e-6)
        bow = float(abs(np.polyfit(normalized, normal, 2)[0]))
        result.update(status='ok', angle_deg=float(np.degrees(np.arctan2(vy, vx))),
                      rms_deviation_px=rms, p95_deviation_px=p95,
                      max_deviation_px=maximum, quadratic_bow_px=bow,
                      rms_deviation_mm=rms / ppm, p95_deviation_mm=p95 / ppm,
                      max_deviation_mm=maximum / ppm, quadratic_bow_mm=bow / ppm)
        diagnostics.append(result)
    opposite = []
    for a, b in ((0, 2), (1, 3)):
        if diagnostics[a]['status'] == diagnostics[b]['status'] == 'ok':
            opposite.append(_angle_difference(diagnostics[a]['angle_deg'],
                                               diagnostics[b]['angle_deg']))
        else:
            opposite.append(None)
    return diagnostics, opposite


def measure_reference_outline(outline: dict, pixels_per_mm: float,
                              tolerance_mm: float = .1) -> dict:
    """Return JSON-ready geometry from Nx2 original-image contour coordinates.

    Optional quadrilateral must contain four distinct corners in cyclic order.
    A sparse four-point contour supports lengths and area but cannot independently
    assess straightness; retain dense boundary points in the detector for that.
    """
    ppm = float(pixels_per_mm)
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError('pixels_per_mm must be finite and positive')
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError('tolerance_mm must be finite and positive')
    points = np.asarray(outline['contour'], dtype=np.float32).reshape(-1, 2)
    if len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError('Contour needs at least three finite points')
    contour = points.reshape(-1, 1, 2)
    area = float(abs(cv2.contourArea(contour)))
    if area <= 0:
        raise ValueError('Contour must enclose positive area')
    short, long = sorted(cv2.minAreaRect(contour)[1])
    epsilon = float(tolerance_mm * ppm)
    smooth = cv2.approxPolyDP(contour, epsilon, True)
    perimeter = float(cv2.arcLength(smooth, True))
    raw = float(cv2.arcLength(contour, True))
    result = dict(long_side_px=float(long), short_side_px=float(short),
                  perimeter_px=perimeter, area_px2=area,
                  long_side_mm=float(long / ppm), short_side_mm=float(short / ppm),
                  perimeter_mm=perimeter / ppm, area_mm2=area / ppm ** 2,
                  aspect_ratio=float(long / short), pixels_per_mm=ppm,
                  raw_perimeter_px=raw, raw_perimeter_mm=raw / ppm,
                  perimeter_tolerance_mm=float(tolerance_mm),
                  perimeter_tolerance_px=epsilon,
                  area_definition='outer_contour_enclosed_area_holes_ignored',
                  side_definition='minimum_area_rectangle_oriented_extents')
    if outline.get('quadrilateral') is not None:
        corners = np.asarray(outline['quadrilateral'], dtype=np.float32).reshape(4, 2)
        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if not np.all(np.isfinite(corners)) or np.min(sides) <= 0:
            raise ValueError('Quadrilateral corners must be finite and distinct')
        result['side_lengths_px'] = sides.astype(float).tolist()
        result['side_lengths_mm'] = (sides / ppm).astype(float).tolist()
        result['quadrilateral_area_px2'] = float(abs(cv2.contourArea(corners)))
        result['quadrilateral_area_mm2'] = result['quadrilateral_area_px2'] / ppm ** 2
        result['edge_straightness'], result['opposite_edge_angle_deg'] = (
            _edge_diagnostics(points.astype(float), corners.astype(float), ppm))
    else:
        result.update(side_lengths_px=None, side_lengths_mm=None,
                      edge_straightness=[], opposite_edge_angle_deg=[])
    return result
