import cv2
import numpy as np
import pytest
from grainmaster.geometry_sensitivity import (transformed_mask_geometry,
    radial_area_jacobian,projective_area_jacobian)


def fixture():
    mask=np.zeros((120,140),np.uint8)
    cv2.ellipse(mask,(70,60),(35,15),25,0,360,1,-1)
    return mask


def test_uniform_magnification_cancels_with_recalibrated_scale():
    mask=fixture()
    before=transformed_mask_geometry(mask,np.array([100,200]),lambda p:p,lambda p:np.ones(len(p)),10)
    after=transformed_mask_geometry(mask,np.array([100,200]),lambda p:p*1.3,
                                    lambda p:np.full(len(p),1.3**2),13)
    for key in before:assert after[key]==pytest.approx(before[key],rel=1e-5)


def test_anisotropic_affine_area_and_axes():
    mask=np.zeros((140,160),np.uint8);cv2.ellipse(mask,(80,70),(35,15),0,0,360,1,-1)
    before=transformed_mask_geometry(mask,np.array([0,0]),lambda p:p,lambda p:np.ones(len(p)),10)
    after=transformed_mask_geometry(mask,np.array([0,0]),lambda p:p*np.array([1.1,.9]),
                                    lambda p:np.full(len(p),.99),10)
    assert after['length_mm']/before['length_mm']==pytest.approx(1.1,rel=.001)
    assert after['width_mm']/before['width_mm']==pytest.approx(.9,rel=.001)
    assert after['area_mm2']/before['area_mm2']==pytest.approx(.99)


def test_jacobians_match_finite_difference():
    points=np.array([[200,300],[800,600]],float)
    H=np.array([[1.,.02,10],[.01,1.,20],[.00002,.00001,1.]])
    def warp(p):
        v=np.column_stack([p,np.ones(len(p))])@H.T
        return v[:,:2]/v[:,2:]
    step=.001
    jx=(warp(points+[step,0])-warp(points))/step
    jy=(warp(points+[0,step])-warp(points))/step
    determinant=jx[:,0]*jy[:,1]-jy[:,0]*jx[:,1]
    np.testing.assert_allclose(determinant,projective_area_jacobian(points,H),rtol=1e-6)
    assert np.all(radial_area_jacobian(points,dict(center_px=[500,500],normalization_px=1000,k1=0),1)==1)
