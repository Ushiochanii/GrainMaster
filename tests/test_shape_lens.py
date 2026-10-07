import numpy as np
import pytest
from grainmaster.shape_lens import (fit_shape_lens, transform_shape_points,
                                    shape_edge_metrics, remap_shape_lens,
                                    extract_shape_edges)


def synthetic(k=.12,zoom=1):
    model={'k1':k,'center_px':[599.5,449.5],'normalization_px':1200.}
    edges=[]
    for i, (a,b) in enumerate([((80,100),(1100,100)),((100,750),(1100,750)),
                               ((130,80),(130,820)),((1070,80),(1070,820))]):
        true=np.linspace(a,b,600)
        u=(true-np.array(model['center_px']))/1200
        target=np.linalg.norm(u,axis=1)
        r=target.copy()
        for _ in range(15):
            r-=(r+k/zoom**2*r**3-target)/(1+3*k/zoom**2*r**2)
        observed=np.array(model['center_px'])+1200*u*(r/target)[:,None]
        edges.append({'object':'card','edge_index':i,'points':observed})
    return edges,model


def test_recovers_radial_from_varied_zoom_and_photo_holdout():
    records=[dict(image_id=str(i),zoom=z,edges=synthetic(zoom=z)[0])
             for i,z in enumerate([1,1.08,1.2])]
    fit=fit_shape_lens(records,(1200,900))
    assert fit['k1']==pytest.approx(.12,abs=.002)
    held,_=synthetic(zoom=1.15)
    before=shape_edge_metrics(held)['mean_edge_rms_px']
    after=shape_edge_metrics(held,fit,1.15)['mean_edge_rms_px']
    assert after<before*.02
    assert fit['alternating_holdout_loss_after']<fit['alternating_holdout_loss_before']*.001


def test_zero_radial_and_identity_render():
    edges,zero=synthetic(k=0)
    fit=fit_shape_lens([dict(image_id='a',edges=edges)],(1200,900))
    assert abs(fit['k1'])<.002
    points=np.array([[0,0],[800,600]],float)
    np.testing.assert_array_equal(transform_shape_points(points,zero),points)
    image=np.arange(80*60*3,dtype=np.uint8).reshape(60,80,3)
    np.testing.assert_array_equal(remap_shape_lens(image,zero),image)


def test_render_finite_and_reject_fold():
    image=np.ones((120,160,3),np.uint8)*128
    model=dict(k1=.15,center_px=[79.5,59.5],normalization_px=160.)
    output=remap_shape_lens(image,model,1.1)
    assert output.shape==image.shape
    assert output.dtype==np.uint8
    with pytest.raises(ValueError):
        remap_shape_lens(image,dict(model,k1=-5))
    with pytest.raises(ValueError):
        transform_shape_points([[0,0]],model,0)


def test_edge_extraction_keeps_original_samples_and_two_ruler_sides():
    q=np.array([[0,0],[1000,0],[1000,100],[0,100]],float)
    contour=np.concatenate([np.linspace(a,b,300) for a,b in zip(q,np.roll(q,-1,axis=0))])
    edges=extract_shape_edges({name:dict(contour=contour,quadrilateral=q)
                              for name in ('ruler','card')})
    assert sum(e['object']=='ruler' for e in edges)==2
    assert sum(e['object']=='card' for e in edges)==4
    for e in edges:
        assert all(any(np.array_equal(p,c) for c in contour) for p in e['points'])


def test_no_dense_edges_rejected():
    with pytest.raises(ValueError):
        fit_shape_lens([dict(image_id='a',edges=[])],(1200,900))
