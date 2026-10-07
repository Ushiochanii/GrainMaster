"""Automatic ruler localization and robust fitting of the metric tick lattice.

The P0 photographs use a dual-scale white ruler. The metric side is selected
by uniform fine-tick spacing over the full ruler, unlike fractional-inch ticks.
No image-specific position or hard-coded pixels/mm is used.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .contracts import SpatialCalibration


def _fit_ticks(positions):
    positions = np.asarray(positions, dtype=float)
    gaps = np.diff(positions)
    step = float(np.median(gaps))
    if step < 3 or len(positions) < 30:
        raise ValueError('Too few resolved ruler ticks')
    regularity = float(np.mean(np.abs(gaps - step) < max(1.5, step * .15)))
    indices = np.r_[0, np.cumsum(np.maximum(1, np.rint(gaps / step)))]
    keep = np.ones(len(positions), dtype=bool)
    for _ in range(5):
        slope, intercept = np.polyfit(indices[keep], positions[keep], 1)
        errors = positions - (indices * slope + intercept)
        mad = np.median(np.abs(errors[keep] - np.median(errors[keep])))
        keep = np.abs(errors - np.median(errors[keep])) <= max(1.5, 3 * 1.4826 * mad)
    residual = float(np.sqrt(np.mean(errors[keep] ** 2)))
    return float(slope), residual, regularity, keep


def _ordered_box(rect):
    points = cv2.boxPoints(rect)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(sums)], points[np.argmin(differences)],
                     points[np.argmax(sums)], points[np.argmax(differences)]], dtype=np.float32)


def calibrate_ruler(image_bgr):
    """Return metric calibration, raising ValueError when no reliable ruler exists."""
    height, width = image_bgr.shape[:2]
    scale = min(1., 1200. / width)
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 160, 255)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        short, long = sorted(rect[1])
        if short < 12 or long < small.shape[0] * .25 or long / short < 8:
            continue
        box = _ordered_box(rect) / scale
        # Normalize all rulers to portrait orientation, even when rotated.
        if np.linalg.norm(box[1]-box[0]) > np.linalg.norm(box[2]-box[1]):
            box = np.roll(box, -1, axis=0)
        cw = int(round(short / scale)); ch = int(round(long / scale))
        transform = cv2.getPerspectiveTransform(box, np.float32([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]]))
        crop = cv2.warpPerspective(image_bgr, transform, (cw,ch))
        cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for side, a, b in [('left', .04, .20), ('right', .80, .96)]:
            profile = cg[:, int(cw*a):int(cw*b)].mean(axis=1)
            signal = gaussian_filter1d(profile, 12) - gaussian_filter1d(profile, .6)
            peaks, _ = find_peaks(signal, prominence=15, distance=4)
            try:
                ppm, residual, regularity, keep = _fit_ticks(peaks)
            except ValueError:
                continue
            coverage = float((peaks[-1]-peaks[0])/ch)
            if regularity < .8 or coverage < .65 or residual > ppm * .75:
                continue
            score = regularity * coverage / (1 + residual / ppm)
            points = np.column_stack([np.full(len(peaks), cw*(a+b)/2), peaks]).astype(np.float32)
            ticks = cv2.perspectiveTransform(points[None], np.linalg.inv(transform))[0]
            x,y,w,h = cv2.boundingRect(box.astype(np.int32))
            candidates.append((score, SpatialCalibration(ppm,1/ppm,(x,y,w,h),residual,
                float(np.clip(score,0,1)), {'polygon':box.tolist(), 'tick_points':ticks[keep].tolist(),
                'tick_count':int(keep.sum()), 'side':side, 'tick_spacing_mm':1.,
                'regularity':regularity, 'coverage':coverage,
                'method':'white elongated ROI / uniform fine metric ticks / robust lattice fit',
                'units':'residual in rectified pixels from a global robust linear fit; fine metric ticks assumed 1 mm',
                'limitations':'single global scale; perspective and lens curvature are not spatially corrected'})))
    if not candidates:
        raise ValueError('No ruler with a sufficiently uniform metric tick lattice was found')
    return max(candidates, key=lambda item:item[0])[1]


def draw_ruler_overlay(image_bgr, calibration):
    overlay = image_bgr.copy()
    polygon = np.int32(calibration.metadata.get('polygon', []))
    if len(polygon):
        cv2.polylines(overlay,[polygon],True,(0,255,0),5)
    for x,y in calibration.metadata.get('tick_points',[]):
        cv2.circle(overlay,(int(x),int(y)),3,(0,0,255),-1)
    x,y,_,_ = calibration.bbox
    label = f'{calibration.pixels_per_mm:.4f} px/mm | RMS {calibration.residual:.2f}px'
    cv2.putText(overlay,label,(x,max(40,y-20)),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
    return overlay

