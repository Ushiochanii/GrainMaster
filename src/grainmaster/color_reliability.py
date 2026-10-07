"""Existing-image diagnostics, not an estimate of absolute seed colour error.

Blue plastic is an unverified repeated surface: material, gloss and lighting
are confounded. Never use these measurements as a seed correction or bound.
"""
import cv2
import colour
import numpy as np


def background_tiles(image_bgr, labels, center, radius, *, inner_fraction=.70,
                     seed_margin=11, grid_size=6):
    """Sample equal-weight spatial tiles, excluding rim and expanded seed masks.

    Blue selection is made on the original image, before any colour correction.
    Returned medians are sRGB triplets, plus local tile centers for inspection.
    """
    h, w = labels.shape
    yy, xx = np.ogrid[:h, :w]
    interior = (xx-center[0])**2 + (yy-center[1])**2 < (radius*inner_fraction)**2
    excluded = cv2.dilate((labels > 0).astype(np.uint8),
                          np.ones((seed_margin, seed_margin), np.uint8)) > 0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 90) & (hsv[..., 0] <= 135) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 15)
    valid = interior & ~excluded & blue
    tiles = []
    for row in range(grid_size):
        for col in range(grid_size):
            y0, y1 = row*h//grid_size, (row+1)*h//grid_size
            x0, x1 = col*w//grid_size, (col+1)*w//grid_size
            local = valid[y0:y1, x0:x1]
            if np.count_nonzero(local) < 200:
                continue
            ys, xs = np.nonzero(local)
            tiles.append(dict(x=float(xs.mean()+x0), y=float(ys.mean()+y0),
                              pixels=int(len(xs)),
                              rgb=np.median(image_bgr[y0:y1, x0:x1][local][:, ::-1], axis=0)/255.))
    if len(tiles) < 4:
        raise ValueError('Fewer than four valid background tiles')
    return tiles, valid


def describe_tiles(rgb):
    rgb = np.asarray(rgb)
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb))
    representative = np.median(lab, axis=0)
    distances = colour.delta_E(lab, representative, method='CIE 2000')
    xyz = colour.sRGB_to_XYZ(rgb)
    return dict(L=float(representative[0]), a=float(representative[1]),
                b=float(representative[2]), Y=float(np.median(xyz[:, 1])),
                within_dish_tile_delta_e00_p90=float(np.percentile(distances, 90)),
                tile_count=len(rgb))


def between_dishes(rows):
    """Ranges describe observed surfaces, not ground-truth-relative errors."""
    labs = np.array([[r['L'], r['a'], r['b']] for r in rows])
    ys = np.array([r['Y'] for r in rows])
    differences = [float(colour.delta_E(labs[i], labs[j], method='CIE 2000'))
                   for i in range(len(labs)) for j in range(i+1, len(labs))]
    return dict(dish_count=len(rows),
                background_L_range=float(np.ptp(labs[:, 0])),
                background_L_range_percent_of_mean=float(np.ptp(labs[:, 0])/labs[:, 0].mean()*100),
                background_Y_range_percent_of_mean=float(np.ptp(ys)/ys.mean()*100),
                background_pair_delta_e00_max=max(differences, default=0),
                background_pair_delta_e00_mean=float(np.mean(differences)) if differences else 0)
