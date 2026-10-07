"""Geometry changes under candidate transforms, not ground-truth errors."""
import cv2
import numpy as np


def transformed_mask_geometry(mask, origin, transform, area_jacobian, pixels_per_mm):
    """Use the P0 ellipse convention and integrate transformed pixel-cell area.

    No segmentation rerun or image raster resampling. The Jacobian integration
    approximates continuous warp area at original pixel centres.
    """
    if not np.isfinite(pixels_per_mm) or pixels_per_mm<=0:
        raise ValueError('Invalid spatial scale')
    mask=np.asarray(mask,bool)
    cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cs:raise ValueError('Empty mask')
    contour=max(cs,key=cv2.contourArea).reshape(-1,2).astype(float)+origin
    if len(contour)<5 or cv2.contourArea(contour.astype(np.float32))<=0:
        raise ValueError('No valid seed ellipse')
    mapped=np.asarray(transform(contour),np.float32)
    _,axes,_=cv2.fitEllipse(mapped)
    width,length=sorted(axes)
    yy,xx=np.nonzero(mask)
    cells=np.column_stack([xx,yy])+origin
    factors=np.asarray(area_jacobian(cells),float)
    if not np.isfinite(factors).all() or np.any(factors<=0):
        raise ValueError('Invalid area Jacobian')
    return dict(length_mm=float(length/pixels_per_mm),width_mm=float(width/pixels_per_mm),
                area_mm2=float(factors.sum()/pixels_per_mm**2))


def radial_area_jacobian(points,model,zoom):
    u=(np.asarray(points,float)-model['center_px'])/model['normalization_px']
    r2=np.sum(u*u,axis=1)
    k=model['k1']/zoom**2
    return (1+k*r2)*(1+3*k*r2)


def projective_area_jacobian(points,matrix):
    p=np.column_stack([points,np.ones(len(points))])
    return np.linalg.det(matrix)/(p@matrix[2])**3
