"""Mask-derived ellipse geometry and median CIELAB (sRGB, D65) seed color.

No segmentation or measurement ground truth is assumed by this module. The
same function can measure a predicted mask or a manually annotated mask.
"""
import cv2
import numpy as np

from .contracts import SeedTraits, SpatialCalibration


def measure_seed(mask, corrected_image_bgr, spatial: SpatialCalibration, *,
                 image_id, dish_id, seed_id, seg_confidence) -> SeedTraits:
    """Measure an instance at its native pixel scale, without resizing.

    Length/width are full fitted ellipse axes of the largest external contour;
    area counts all foreground pixels, including disconnected parts. Fragmented
    or tiny masks are flagged. Input image is corrected uint8 BGR sRGB; OpenCV's
    float conversion produces L* in [0,100], signed a*/b*, with D65 white.
    """
    mask = np.asarray(mask, dtype=bool)
    image = np.asarray(corrected_image_bgr)
    if mask.ndim != 2 or image.shape != (*mask.shape, 3):
        raise ValueError('Mask and corrected BGR image dimensions must match')
    if image.dtype != np.uint8:
        raise ValueError('Corrected BGR image must use uint8 sRGB values')
    if not np.isfinite(spatial.mm_per_pixel) or spatial.mm_per_pixel <= 0:
        raise ValueError('Spatial calibration must have a finite positive mm_per_pixel')
    if not mask.any():
        raise ValueError('Cannot measure an empty seed mask')
    flags = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contours) > 1:
        flags.append('fragmented_mask')
    if len(contour) >= 5 and cv2.contourArea(contour) > 0:
        _, axes, _ = cv2.fitEllipse(contour)
        width_px, length_px = sorted(axes)
    else:
        # A tiny object has no well-defined fitted ellipse; retain its area/color.
        length_px = width_px = float('nan')
        flags.append('ellipse_unavailable')
    if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
        flags.append('touches_crop_border')
    internal = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    if not internal.any():
        internal = mask
        flags.append('color_erosion_empty')
    # Convert only the tight bbox for speed on full-resolution photographs.
    ys, xs = np.nonzero(internal)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    lab = cv2.cvtColor(image[y0:y1, x0:x1].astype(np.float32) / 255., cv2.COLOR_BGR2LAB)
    L, a, b = np.median(lab[np.asarray(internal[y0:y1, x0:x1], dtype=bool)], axis=0)
    return SeedTraits(str(image_id), int(dish_id), int(seed_id),
                      float(length_px * spatial.mm_per_pixel),
                      float(width_px * spatial.mm_per_pixel),
                      float(mask.sum() * spatial.mm_per_pixel ** 2),
                      float(L), float(a), float(b), float(seg_confidence),
                      ';'.join(flags) if flags else 'ok')


def aggregate_dishes(traits: list[SeedTraits], dish_ids: list[int]) -> list[dict]:
    """Summarize all requested dishes, retaining empty dishes with count zero."""
    summary = []
    for dish_id in dish_ids:
        items = [item for item in traits if item.dish_id == dish_id]
        row = {'dish_id': dish_id, 'seed_count': len(items)}
        for name in ('length_mm', 'width_mm', 'area_mm2'):
            values = [getattr(item, name) for item in items
                      if np.isfinite(getattr(item, name))]
            row['mean_' + name] = float(np.mean(values)) if values else None
        summary.append(row)
    return summary
