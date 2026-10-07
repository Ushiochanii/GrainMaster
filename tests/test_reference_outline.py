import cv2
import numpy as np
import pytest
from pathlib import Path
from grainmaster.reference_outline import detect_reference_outlines
from grainmaster.contracts import SpatialCalibration, ColorCalibration


def synthetic(angle):
    image=np.full((950,1600,3),130,np.uint8)
    # Rounded physical card, including right ruler strip beyond the patch grid.
    x,y,w,h,r=350,100,1058,635,26
    cv2.rectangle(image,(x+r,y),(x+w-r,y+h),(85,85,85),-1)
    cv2.rectangle(image,(x,y+r),(x+w,y+h-r),(85,85,85),-1)
    for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(image,(cx,cy),r,(85,85,85),-1)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask=(gray==85).astype(np.uint8)*255
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image,cs,-1,(25,25,25),3)
    centers=np.array([[485+150*c,194+150*r] for r in range(4) for c in range(6)],np.float32)
    for i,(cx,cy) in enumerate(centers.astype(int)):
        cv2.rectangle(image,(cx-55,cy-55),(cx+55,cy+55),((i*37)%150+90,(i*19)%150+90,(i*7)%150+90),-1)
    cv2.rectangle(image,(80,160),(125,780),(225,225,225),-1)
    for yy in range(180,750,10):cv2.line(image,(80,yy),(89,yy),(25,25,25),1)
    M=cv2.getRotationMatrix2D((800,475),angle,1)
    out=cv2.warpAffine(image,M,(1600,950),borderValue=(130,130,130))
    centers=cv2.transform(centers[None],M)[0]
    ruler=cv2.transform(np.float32([[[80,160],[125,160],[125,780],[80,780]]]),M)[0]
    bbox=cv2.boundingRect(ruler)
    spatial=SpatialCalibration(10,.1,bbox,0,1)
    color=ColorCalibration(np.eye(3),(0,0,1,1),0,0,{'patch_centers':centers.tolist()})
    expected=cv2.transform(cs[0].astype(np.float32),M).reshape(-1,2)
    return out,spatial,color,expected


@pytest.mark.parametrize('angle',[0,7,-7])
def test_actual_outer_silhouette_and_rotations(angle):
    image,s,c,expected=synthetic(angle)
    result=detect_reference_outlines(image,s,c)
    detected=result['card']['contour']
    area=cv2.contourArea(detected)
    assert abs(area/cv2.contourArea(expected)-1)<.025
    assert len(detected)>1000
    assert cv2.isContourConvex(result['card']['quadrilateral'])
    # Test actual border reaches beyond last color patch, including scale margin.
    assert area > 600000
    assert result['card']['metadata']['corner_tracing']
    assert abs(cv2.contourArea(result['ruler']['contour'])/(45*620)-1)<.03


def test_real_0462_outer_boundary():
    import json
    path=Path('artifacts/prototype_p0/0462/calibration.json')
    if not path.exists():pytest.skip('Local image/calibration unavailable')
    j=json.loads(path.read_text())
    image=cv2.imread('data/raw/0462.jpg')
    result=detect_reference_outlines(image,SpatialCalibration(**j['spatial']),ColorCalibration(**j['color']))
    x,y,w,h=result['card']['bbox']
    assert 1120<w<1200 and 700<h<770
    assert x<2450 and x+w>3520
    assert 750000<cv2.contourArea(result['card']['contour'])<850000
    assert len(result['ruler']['contour'])>6000

