from pathlib import Path
import numpy as np
import pytest
from iptc.data import Recording
from iptc.geometry import make_blocks
from iptc.baselines import raw_knots
from iptc.events import relations
from iptc.prune_band_csr import prune
from iptc.danger_band import active_predicates

def test_closed_boundary_and_stable_interior():
    assert active_predicates(np.array([8.,10.,12.,7.,13.]),np.array([10.]),1.,np.zeros((1,2)))[:,0].tolist()==[True,True,True,False,False]

def test_masks_are_per_threshold_and_cover_roundoff():
    masks=active_predicates(np.array([5.,10.,20.,0.]),np.array([5.,10.,20.]),.2,np.zeros((2,2)))
    assert np.array_equal(masks[:3],np.eye(3,dtype=bool))
    assert not masks[3].any()
    # Translation-dependent guard widens the band and never excludes its boundary.
    assert active_predicates(np.array([10.4+1e-10]),np.array([10.]),.2,np.full((2,2),1e6)).item()

def test_all_stable_positive_relations_need_no_dynamic_edges(tmp_path):
    xy=np.array([[0.,0.],[1.,.01],[2.,0.],[0.,.1],[1.,.11],[2.,.1]])
    rec=Recording(dict(ticks_per_second=1),['a','b'],np.repeat(np.arange(2,dtype=np.int32),3),np.tile(np.arange(3),2),xy)
    blocks=make_blocks(rec,4);truth=relations(rec,[10.])
    knots,stats=prune(rec,blocks,raw_knots(blocks),[10.],.2,truth,'FULL-DAG',100,1e99,tmp_path,None,True)
    assert list(map(len,knots))==[2,2]
    assert stats['profile']['spatial_pairs']==3
    assert stats['profile']['dynamic_pairs']==stats['profile']['dependency_edges']==0

def test_geometrically_infeasible_seed_rejected(tmp_path):
    rec=Recording(dict(ticks_per_second=1),['a'],np.zeros(3,dtype=np.int32),np.arange(3),np.array([[0.,0.],[0.,4.],[2.,0.]]))
    blocks=make_blocks(rec,4)
    with pytest.raises(ValueError,match='seed'):
        prune(rec,blocks,[np.array([0,2])],[1.],.1,relations(rec,[1.]),'FULL-DAG',100,1e99,tmp_path,None,True)

def test_overlapping_threshold_bands(tmp_path):
    rng=np.random.default_rng(44)
    xy=np.concatenate([np.column_stack((np.arange(12)*.4+rng.normal(0,.08,12),np.full(12,y)+rng.normal(0,.08,12))) for y in [0.,1.1,1.8]])
    rec=Recording(dict(ticks_per_second=4),list('abc'),np.repeat(np.arange(3,dtype=np.int32),12),np.tile(np.arange(12),3),xy)
    blocks=make_blocks(rec,2);radii=[1.,1.5,2.];truth=relations(rec,radii);paths=[]
    for band in [False,True]:
        knots,stats=prune(rec,blocks,raw_knots(blocks),radii,.6,truth,'FULL-DAG',100,1e99,tmp_path,None,True,danger_band=band)
        paths.append(knots)
    assert all(np.array_equal(a,b) for a,b in zip(*paths))
    p=stats['profile'];assert p['band_predicates']>p['dynamic_pairs']

@pytest.mark.parametrize('epsilon',[.05,.2,.5])
def test_sparse_schedule_preserves_full_path(tmp_path,epsilon):
    rng=np.random.default_rng(6016)
    xy=np.concatenate([np.column_stack((np.arange(15)*.7+rng.normal(0,.05,15),np.full(15,y)+rng.normal(0,.03,15))) for y in [0.,1.,4.,50.]])
    rec=Recording(dict(ticks_per_second=5),list('abcd'),np.repeat(np.arange(4,dtype=np.int32),15),np.tile(np.arange(15),4),xy)
    blocks=make_blocks(rec,2);truth=relations(rec,[1.,4.,10.]);results=[]
    for band in [False,True]:
        knots,stats=prune(rec,blocks,raw_knots(blocks),[1.,4.,10.],epsilon,truth,'FULL-DAG',100,1e99,tmp_path,None,True,danger_band=band)
        results.append(knots)
    assert all(np.array_equal(a,b) for a,b in zip(*results))
