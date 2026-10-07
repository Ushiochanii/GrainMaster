"""Smooth a measured card silhouette, preserving its observed perspective.

This fits an assumed rounded rectangle; it does not estimate camera calibration
or impose a manufacturer's physical size. The original outline remains evidence.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def _rounded_contour(width, height, radius, samples=120):
    """Cyclic, dense straight segments and quarter-circle corners."""
    centers = [(width-radius, radius), (width-radius, height-radius),
               (radius, height-radius), (radius, radius)]
    starts = [-np.pi/2, 0, np.pi/2, np.pi]
    arcs = [np.array(c) + radius*np.column_stack((np.cos(a), np.sin(a)))
            for c, s in zip(centers, starts)
            for a in [np.linspace(s, s+np.pi/2, samples)]]
    parts = []
    for i, arc in enumerate(arcs):
        parts.append(arc)
        parts.append(np.linspace(arc[-1], arcs[(i+1) % 4][0], samples)[1:-1])
    return np.vstack(parts).astype(np.float32)


def _robust_quad(points, initial):
    lines = []
    lengths = np.linalg.norm(np.roll(initial, -1, axis=0)-initial, axis=1)
    for p, q in zip(initial, np.roll(initial, -1, axis=0)):
        v = q-p
        t = (points-p)@v/(v@v)
        delta = points-p
        normal = np.abs(v[0]*delta[:, 1]-v[1]*delta[:, 0])/np.linalg.norm(v)
        keep = (t > .18) & (t < .82) & (normal < .06*min(lengths))
        if keep.sum() < 8:
            raise ValueError('Insufficient central side support for rounded card')
        vx, vy, x, y = cv2.fitLine(points[keep].astype(np.float32),
                                  cv2.DIST_HUBER, 0, .001, .001).ravel()
        lines.append((np.array([x, y]), np.array([vx, vy])))
    corners = []
    for i in range(4):
        p, v = lines[i-1]
        q, w = lines[i]
        z = np.linalg.solve(np.column_stack((v, -w)), q-p)
        corners.append(p+z[0]*v)
    quad = np.asarray(corners, np.float32)
    if not cv2.isContourConvex(quad) or abs(cv2.contourArea(quad)) <= 0:
        raise ValueError('Invalid fitted card quadrilateral')
    return quad


def fit_rounded_card(outline: dict) -> dict:
    """Return a compatible silhouette with fit diagnostics in original pixels.

    Four central sides receive independent robust line fits. Their intersections
    define a homography to a local rectangle whose aspect is estimated from
    average observed opposite side lengths. A shared corner radius is robustly
    fitted there. The smooth contour is projected BACK into the original photo.
    Circular corner shape and estimated aspect are assumptions, not physical GT.
    """
    points = np.asarray(outline['contour'], np.float32).reshape(-1, 2)
    if len(points) < 32 or not np.isfinite(points).all():
        raise ValueError('Need at least 32 finite observed boundary points')
    initial = outline.get('quadrilateral')
    if initial is None:
        initial = cv2.boxPoints(cv2.minAreaRect(points))
    initial = np.asarray(initial, np.float32).reshape(4, 2)
    quad = _robust_quad(points, initial)
    sides = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
    width, height = float((sides[0]+sides[2])/2), float((sides[1]+sides[3])/2)
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    transform = cv2.getPerspectiveTransform(quad, target)
    rectified = cv2.perspectiveTransform(points[:, None], transform)[:, 0]
    corner_distance = np.linalg.norm(rectified[:, None]-target[None], axis=2).min(axis=1)
    corner_points = rectified[corner_distance < .28*min(width, height)]
    if len(corner_points) < 20:
        raise ValueError('Insufficient rounded corner boundary support')
    def residual(parameter):
        r = parameter[0]
        q = np.abs(corner_points-[width/2, height/2])-[width/2-r, height/2-r]
        return np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(np.maximum(q[:, 0], q[:, 1]), 0)-r
    fit = least_squares(residual, [.04*min(width, height)],
                        bounds=(.0001*min(width, height), .24*min(width, height)),
                        loss='soft_l1', f_scale=max(1., min(width, height)*.002))
    radius = float(fit.x[0])
    smooth = cv2.perspectiveTransform(_rounded_contour(width, height, radius)[:, None],
                                     np.linalg.inv(transform))[:, 0]
    # Distance to adjacent polyline segments, not just sampled vertices: the
    # latter would falsely count sample spacing as fitting error on long sides.
    nearest = cKDTree(smooth).query(points)[1]
    distances = []
    for delta in (-1, 0):
        a = smooth[(nearest+delta) % len(smooth)]
        b = smooth[(nearest+delta+1) % len(smooth)]
        vector = b-a
        t = np.sum((points-a)*vector, axis=1)/np.maximum(np.sum(vector**2, axis=1), 1e-12)
        projection = a+np.clip(t, 0, 1)[:, None]*vector
        distances.append(np.linalg.norm(points-projection, axis=1))
    distance = np.minimum(*distances)
    raw_area = abs(cv2.contourArea(points)); fitted_area = abs(cv2.contourArea(smooth))
    diagnostics = dict(radius_rectified_px=radius, rectified_width_px=width,
                       rectified_height_px=height, corner_point_count=int(len(corner_points)),
                       residual_rms_px=float(np.sqrt(np.mean(distance**2))),
                       residual_median_px=float(np.median(distance)),
                       residual_p95_px=float(np.percentile(distance, 95)),
                       observed_area_px2=float(raw_area), fitted_area_px2=float(fitted_area),
                       area_change_percent=float((fitted_area/raw_area-1)*100),
                       observed_perimeter_px=float(cv2.arcLength(points[:, None], True)),
                       fitted_perimeter_px=float(cv2.arcLength(smooth[:, None], True)),
                       converged=bool(fit.success))
    return dict(contour=smooth, quadrilateral=quad,
                bbox=tuple(map(int, cv2.boundingRect(smooth))),
                metadata=dict(method='robust central sides plus shared circular corners in local rectangle',
                              source_method=outline.get('metadata', {}).get('method'),
                              smoothing_only=True, global_geometry_corrected=False,
                              aspect_source='average observed opposite quadrilateral side lengths; no nominal dimensions',
                              radius_definition='local image-pixel coordinates; physical circularity assumed',
                              fit=diagnostics), fit=diagnostics)
