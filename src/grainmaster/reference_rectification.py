"""Full-frame planar candidates with explicit independent ruler checks.

These transforms do not estimate lens coefficients or verify coplanarity.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def project(points, matrix):
    p = np.asarray(points, float).reshape(-1, 2)
    h = np.column_stack([p, np.ones(len(p))]) @ matrix.T
    return h[:, :2] / h[:, 2:]


def ruler_samples(spatial):
    p = np.asarray(spatial.metadata['tick_points'], float)
    gaps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    indices = np.r_[0., np.cumsum(np.maximum(1, np.rint(gaps / np.median(gaps))))]
    return p, indices


def fit_scale(points, indices):
    p = np.asarray(points, float)
    n = np.asarray(indices, float)
    direction = (p[-1]-p[0]) / np.linalg.norm(p[-1]-p[0])
    coordinate = p @ direction
    keep = np.ones(len(p), bool)
    for _ in range(4):
        slope, intercept = np.polyfit(n[keep], coordinate[keep], 1)
        error = coordinate - slope*n - intercept
        mad = np.median(abs(error[keep]-np.median(error[keep])))
        keep = abs(error-np.median(error[keep])) <= max(1.5, 3*1.4826*mad)
    slope, intercept = np.polyfit(n[keep], coordinate[keep], 1)
    return float(slope)


def scale_check(main, main_indices, card, card_indices):
    global_scale = fit_scale(main, main_indices)
    card_scale = fit_scale(card, card_indices)
    parts = np.array_split(np.arange(len(main)), 3)
    slopes = [fit_scale(main[ix], main_indices[ix]) for ix in parts]
    return dict(main_pixels_per_mm=global_scale, card_pixels_per_mm=card_scale,
                difference_percent=100*(card_scale/global_scale-1),
                main_segment_pixels_per_mm=slopes,
                main_segment_cv_percent=float(100*np.std(slopes, ddof=1)/np.mean(slopes)),
                main_last_vs_first_percent=100*(slopes[-1]/slopes[0]-1))


def card_outline_candidate(outline):
    corners = np.asarray(outline['quadrilateral'], np.float32)
    lengths = np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1)
    w, h = (lengths[0]+lengths[2])/2, (lengths[1]+lengths[3])/2
    # Observed mean side lengths provide an unverified target aspect ratio.
    target = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
    H = cv2.getPerspectiveTransform(corners, target)
    center = corners.mean(axis=0)
    shifted = np.eye(3)
    shifted[:2,2] = center-project(center[None],H)[0]
    return shifted @ H


def long_ruler_candidate(main, indices, image_shape):
    """One-axis projective denominator estimated from alternating long ticks.

    Remaining long ticks and the entire card are held out. The omitted transverse
    projective component and horizontal metric accuracy are not identified.
    """
    center = main.mean(axis=0)
    direction = (main[-1]-main[0])/np.linalg.norm(main[-1]-main[0])
    size = max(image_shape[:2])
    t = (main-center) @ direction / size
    train = np.arange(len(main)) % 2 == 0
    initial = np.polyfit(indices[train],t[train],1)
    def residual(theta):
        a,b,p = theta
        return (t[train]/(1+p*t[train])-a*indices[train]-b)*size
    fitted = least_squares(residual,[*initial,0.],loss='soft_l1',f_scale=1.5,
                           bounds=([-np.inf,-np.inf,-.5],[np.inf,np.inf,.5]))
    p = fitted.x[2]
    H = np.eye(3)
    H[2,:2] = p*direction/size
    H[2,2] = 1-p*(center@direction)/size
    # Keep the central ruler location fixed; no physical rescaling.
    translated = np.eye(3)
    translated[:2,2] = center-project(center[None],H)[0]
    return translated @ H, dict(parameter=float(p), training_tick_count=int(train.sum()),
                                held_out_tick_count=int((~train).sum()))


def bounded_canvas(matrix, image_shape):
    h,w = image_shape[:2]
    corners = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], float)
    denominators = np.column_stack([corners,np.ones(4)]) @ matrix[2]
    if np.min(denominators) <= .2:
        raise ValueError('Unsafe projective horizon in frame')
    mapped = project(corners,matrix)
    lo = np.floor(mapped.min(axis=0)); hi = np.ceil(mapped.max(axis=0))
    extent = hi-lo+1
    if np.any(extent > 2*np.array([w,h])) or np.prod(extent)>2.5*w*h:
        raise ValueError('Excessive full-frame extrapolation')
    shift = np.eye(3); shift[:2,2]=-lo
    return shift@matrix, extent.astype(int).tolist()


def evaluate_candidates(spatial, card_scale, outline, image_shape):
    main, indices = ruler_samples(spatial)
    card = np.asarray(card_scale['tick_points'],float)
    card_indices = np.asarray(card_scale['tick_indices'],float)*card_scale.get('tick_spacing_mm',1.)
    original = scale_check(main,indices,card,card_indices)
    long_matrix, long_metadata = long_ruler_candidate(main,indices,image_shape)
    results = []
    for name,matrix,metadata in [('card_outer_edges',card_outline_candidate(outline),
                                 {'target_aspect':'observed mean sides, not verified physical aspect',
                                  'fit_source':'card outer straight sides; full long ruler held out'}),
                                ('long_ruler_one_axis',long_matrix,long_metadata)]:
        record = dict(name=name,before=original,applied=False,accepted=False,
                      metadata=metadata,limitations=['conditional common measurement plane',
                      'no lens calibration; horizontal accuracy not independently verified'])
        try:
            matrix,size = bounded_canvas(matrix,image_shape)
            after_main, after_card = project(main,matrix), project(card,matrix)
            after = scale_check(after_main,indices,after_card,card_indices)
            held = np.arange(len(main))%2==1 if name=='long_ruler_one_axis' else np.ones(len(main),bool)
            independent_before=scale_check(main[held],indices[held],card,card_indices)
            independent_after=scale_check(after_main[held],indices[held],after_card,card_indices)
            reasons=[]
            if abs(independent_after['difference_percent'])>3:
                reasons.append('Remaining independent scale discrepancy exceeds 3 percent engineering gate')
            if independent_after['main_segment_cv_percent']>max(.2,independent_before['main_segment_cv_percent']*1.1):
                reasons.append('Held-out long-ruler segment uniformity worsens')
            if abs(independent_after['difference_percent'])>=abs(independent_before['difference_percent'])-.5:
                reasons.append('No independent scale improvement of at least 0.5 percentage point')
            record.update(matrix=matrix.tolist(),output_size=size,after=after,
                          independent_before=independent_before,independent_after=independent_after,
                          failure_reasons=reasons,accepted=not reasons,
                          status='conditional_candidate_pass' if not reasons else 'candidate_rejected')
        except ValueError as exc:
            record.update(status='candidate_rejected',failure_reasons=[str(exc)])
        results.append(record)
    return results
