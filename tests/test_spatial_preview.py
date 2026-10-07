import numpy as np
from grainmaster.spatial_preview import fit_joint_candidate


def scene():
    n=np.arange(100,dtype=float)
    ticks=np.column_stack([np.full(100,100.),100+8*n])
    card=np.array([[300,700],[600,700],[600,850],[300,850]],float)
    t=np.linspace(0,2*np.pi,150,endpoint=False)
    circle=np.column_stack([np.cos(t),np.sin(t)])*65
    rims=[circle+[x,y] for x,y in [(300,300),(500,300),(700,300),(300,500)]]
    return ticks,n,card,rims


def test_identity_scene_remains_metric_and_safe():
    ticks,n,card,rims=scene()
    r=fit_joint_candidate(ticks,n,card,rims,(1000,1000,3))
    assert abs(r['after']['pixels_per_mm']-8)<.01
    assert max(abs(np.array(r['parameters'])))<.001
    assert r['metric_reference']=='long_ruler_only'
    assert r['applied_to_measurements'] is False
    assert r['heldout_tick_count']==50


def test_validation_observations_do_not_enter_fit():
    ticks,n,card,rims=scene()
    baseline=fit_joint_candidate(ticks,n,card,rims,(1000,1000,3))
    ticks[1::2,1]+=np.linspace(0,10,50)
    rims[-1]=rims[-1]*[1.1,1.]
    changed=fit_joint_candidate(ticks,n,card,rims,(1000,1000,3))
    np.testing.assert_allclose(changed['parameters'],baseline['parameters'],atol=1e-8)
    assert changed['dish_axis_ratios_after'][-1]<.95
