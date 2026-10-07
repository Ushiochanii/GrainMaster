from types import SimpleNamespace
import numpy as np
import pytest
from grainmaster.reference_rectification import (project, long_ruler_candidate,
    scale_check, bounded_canvas, evaluate_candidates)


def test_projective_long_ruler_fit_reduces_held_out_variation():
    indices=np.arange(300.)
    true=np.column_stack([np.full(300,200.),100+indices*10])
    H=np.array([[1.,0,0],[0,1.,0],[0,-.000008,1.]])
    observed=project(true,H)
    correction,_=long_ruler_candidate(observed,indices,(4000,5000))
    held=np.arange(300)%2==1
    transformed=project(observed,correction)
    segments=np.array_split(np.flatnonzero(held),3)
    slopes=[np.polyfit(indices[ix],transformed[ix,1],1)[0] for ix in segments]
    assert np.std(slopes)/np.mean(slopes)<1e-5


def test_no_forced_two_reference_scale_agreement():
    main=np.column_stack([np.full(300,200.),100+np.arange(300)*10.])
    indices=np.arange(300.)
    card=np.column_stack([np.full(50,1200.),2000+np.arange(50)*10.6])
    H,_=long_ruler_candidate(main,indices,(4000,5000))
    result=scale_check(project(main,H),indices,project(card,H),np.arange(50.))
    assert result['difference_percent']==pytest.approx(6,abs=1e-5)


def test_invalid_horizon_rejected():
    H=np.eye(3);H[2,1]=-.001
    with pytest.raises(ValueError,match='horizon'):
        bounded_canvas(H,(4000,5000))


def test_known_rectangle_homography_full_held_out_ruler():
    H=np.array([[1.,.01,0],[.02,1.,0],[0,.00001,1.]])
    quad=np.array([[1000,2000],[2100,2000],[2100,2650],[1000,2650]],float)
    main=project(np.column_stack([np.full(300,200.),100+np.arange(300)*10.]),H)
    card=project(np.column_stack([np.full(50,2050.),2010+np.arange(50)*10.]),H)
    spatial=SimpleNamespace(metadata={'tick_points':main.tolist()})
    results=evaluate_candidates(spatial,{'tick_points':card.tolist(),'tick_indices':list(range(50))},
                                {'quadrilateral':project(quad,H)},(4000,5000))
    candidate=results[0]
    assert abs(candidate['after']['difference_percent'])<.03
    assert candidate['after']['main_segment_cv_percent']<1e-4
    assert candidate['applied'] is False
