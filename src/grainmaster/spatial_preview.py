"""Conservative joint planar candidate; ruler is the only metric reference."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from .reference_rectification import project, fit_scale, bounded_canvas


def detect_dish_rim(image, dish):
    """Observe radial intensity edges near the plastic rim, not blue-mask ellipses."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    center = np.asarray(dish['center'], float)
    radius = float(dish['radius'])
    angles = np.linspace(0, 2*np.pi, 240, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    distances = np.linspace(.97*radius, 1.15*radius, 90)
    p = center + directions[:, None, :]*distances[None, :, None]
    values = cv2.remap(gray, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    gradient = abs(np.diff(values, axis=1))
    index = gradient.argmax(axis=1)
    points = center + directions*((distances[index]+distances[index+1])/2)[:, None]
    strength = gradient[np.arange(len(index)), index]
    keep = strength > max(2., np.percentile(strength, 15))
    for _ in range(3):
        ellipse = cv2.fitEllipse(points[keep].astype(np.float32))
        c, axes, angle = ellipse
        t = np.radians(angle)
        rotation = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = (points-np.array(c))@rotation.T/(np.array(axes)/2)
        residual = abs(np.linalg.norm(q, axis=1)-1)
        keep &= residual < max(.015, np.median(residual[keep])*3)
    if keep.sum() < 60:
        raise ValueError('Insufficient consistent plastic rim observations')
    return points[keep]


def roundness(points):
    _, axes, _ = cv2.fitEllipse(np.asarray(points, np.float32))
    return float(min(axes)/max(axes))


def tick_metrics(points, indices):
    parts = np.array_split(np.arange(len(points)), 3)
    slopes = [fit_scale(points[ix], indices[ix]) for ix in parts]
    return dict(pixels_per_mm=fit_scale(points, indices), segment_pixels_per_mm=slopes,
                segment_cv_percent=float(100*np.std(slopes, ddof=1)/np.mean(slopes)))


def fit_joint_candidate(ticks, indices, card_corners, rims, image_shape):
    """Four near-identity parameters: metric anisotropy/shear and perspective.

    Alternating ruler ticks and last dish never enter the objective. No chart
    tick spacing, chart aspect ratio, dish diameter or lens coefficient is used.
    """
    h, w = image_shape[:2]
    scale = max(h, w)
    center = np.array([w/2, h/2])
    N = np.array([[1/scale, 0, -center[0]/scale], [0, 1/scale, -center[1]/scale], [0, 0, 1.]])
    train = np.arange(len(ticks))%2 == 0
    def matrix(theta):
        a, b, px, py = theta
        H = np.array([[np.exp(a), b, 0], [b, np.exp(-a), 0], [px, py, 1.]])
        return np.linalg.inv(N)@H@N
    def residual(theta):
        H = matrix(theta)
        p = project(ticks[train], H)
        n = indices[train]
        coefficients = np.linalg.lstsq(np.column_stack([n, np.ones(len(n))]), p, rcond=None)[0]
        tick_error = (p-n[:,None]*coefficients[0]-coefficients[1]).ravel()/1.5/np.sqrt(len(n))
        c = project(card_corners, H)
        edges = np.roll(c,-1,axis=0)-c
        unit = edges/np.linalg.norm(edges,axis=1)[:,None]
        shape = np.array([unit[0,0]*unit[2,1]-unit[0,1]*unit[2,0],
                          unit[1,0]*unit[3,1]-unit[1,1]*unit[3,0],
                          unit[0]@unit[1], unit[1]@unit[2]])/.015
        circles = [np.log(roundness(project(r, H)))/.015 for r in rims[:-1]]
        return np.r_[tick_error, shape, circles, np.asarray(theta)/np.array([.12,.12,.2,.2])*.15]
    def jacobian(theta):
        step = 1e-4
        return np.column_stack([(residual(theta+np.eye(4)[i]*step)-
                                 residual(theta-np.eye(4)[i]*step))/(2*step) for i in range(4)])
    fitted = least_squares(residual, np.zeros(4), jac=jacobian, bounds=([-.12,-.12,-.2,-.2],[.12,.12,.2,.2]),
                           loss='soft_l1', diff_step=1e-3, max_nfev=180)
    total_nfev = int(fitted.nfev)
    retried = False
    # A difficult frame can exhaust the first budget while still improving.
    # Continue the same bounded fit; keep genuine solver failures as failures.
    if fitted.status == 0 and np.isfinite(fitted.x).all():
        retried = True
        fitted = least_squares(residual, fitted.x, jac=jacobian,
                               bounds=([-.12,-.12,-.2,-.2],[.12,.12,.2,.2]),
                               loss='soft_l1', diff_step=1e-3, max_nfev=720)
        total_nfev += int(fitted.nfev)
    H, size = bounded_canvas(matrix(fitted.x), image_shape)
    held = ~train
    before = tick_metrics(ticks[held], indices[held])
    after = tick_metrics(project(ticks[held], H), indices[held])
    before_r = [roundness(r) for r in rims]
    after_r = [roundness(project(r, H)) for r in rims]
    accepted = (after['segment_cv_percent'] < before['segment_cv_percent'] and
                all(a >= b-.002 for a,b in zip(after_r,before_r)))
    return dict(matrix=H.tolist(), output_size=size, parameters=fitted.x.tolist(),
                optimizer_success=bool(fitted.success), accepted=bool(accepted), applied_to_measurements=False,
                optimizer_status=int(fitted.status), optimizer_message=str(fitted.message),
                optimizer_nfev=total_nfev, optimizer_retried=retried,
                optimizer_cost=float(fitted.cost),
                metric_reference='long_ruler_only', model='joint_near_identity_planar_homography',
                heldout_tick_count=int(held.sum()), heldout_dish_index=len(rims),
                before=before, after=after, dish_axis_ratios_before=before_r, dish_axis_ratios_after=after_r,
                limitations=['Conditional circular rims and common plane; edge localization uncertainty.',
                             'No independently verified lens distortion or physical measurement accuracy.'])
