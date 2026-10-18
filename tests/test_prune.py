import itertools
import time
import numpy as np
import pytest
from iptc.prune import shortest_subsequence,greedy_subsequence,shortest_feasible_path,shortcut_xy,prune
from iptc.local import refine
from iptc.geometry import make_blocks,independent_knots,reconstruct,interpolate
from iptc.events import relations
from test_core import recording,flip_recording


def test_dag_can_remove_jointly_redundant_points_when_single_deletion_cannot():
    valid={(0,1),(1,2),(2,3),(0,3)}
    feasible=lambda a,b:(a,b) in valid
    knots=np.arange(4)
    assert greedy_subsequence(knots,feasible).tolist()==[0,1,2,3]
    assert shortest_subsequence(knots,feasible).tolist()==[0,3]


def test_shortest_path_matches_exhaustive_subsets():
    rng=np.random.default_rng(47)
    for size in range(2,9):
        for trial in range(8):
            edges={(i,i+1) for i in range(size-1)}|{(i,j) for i in range(size) for j in range(i+2,size) if rng.random()<.3}
            valid=lambda a,b:(a,b) in edges
            result=shortest_subsequence(np.arange(size),valid)
            optimum=size
            for flags in itertools.product([False,True],repeat=size-2):
                path=[0]+[i+1 for i,f in enumerate(flags) if f]+[size-1]
                if all(valid(a,b) for a,b in zip(path,path[1:])):optimum=min(optimum,len(path))
            assert len(result)==optimum
            assert all(valid(int(a),int(b)) for a,b in zip(result,result[1:]))


def test_shortest_path_propagates_improvement_through_adjacent_edge():
    valid={(0,1),(1,2),(2,3),(0,2)}
    assert shortest_subsequence(np.arange(4),lambda a,b:(a,b) in valid).tolist()==[0,2,3]


@pytest.mark.parametrize('mode',['GREEDY-PRUNE','DAG-PRUNE','FULL-DAG'])
def test_real_coordinate_shortcuts_preserve_geometry_and_relations(mode,tmp_path):
    rng=np.random.default_rng(9)
    for trial in range(3):
        rec=recording([[(t,float(i*4.99+rng.normal(0,.1)),float(rng.normal(0,.1))) for t in range(12)] for i in range(5)])
        blocks=make_blocks(rec,8);coarse=independent_knots(rec,blocks,.2);truth=relations(rec,[5,10,20],True)
        seed,_=refine(rec,blocks,coarse,[5,10,20],.2,truth,100,time.perf_counter()+20,'ENDPOINT-REPAIR',tmp_path)
        final,stats=prune(rec,blocks,seed,[5,10,20],.2,truth,mode,100,time.perf_counter()+20,tmp_path)
        xy=reconstruct(rec,blocks,final)
        rebuilt=type(rec)(rec.metadata,rec.tracks,rec.track,rec.time,xy,rec.dt)
        assert relations(rebuilt,[5,10,20],True)==truth
        assert np.linalg.norm(xy-rec.xy,axis=1).max()<=.2
        assert sum(map(len,final))<=sum(map(len,seed))


def test_missing_times_and_two_point_blocks(tmp_path):
    rec=recording([[(0,0.,0.),(1,0.,0.),(4,0.,0.)],[(0,5.,0.),(1,5.,0.),(4,5.,0.)]])
    blocks=make_blocks(rec,10);knots=[np.arange(len(b.rows)) for b in blocks]
    final,stats=prune(rec,blocks,knots,[5],0.,relations(rec,[5]),'DAG-PRUNE',10,time.perf_counter()+10,tmp_path)
    assert stats['removed_points']==0


def test_full_candidate_graph_can_use_a_previously_unselected_anchor():
    edges={(0,2),(2,3),(3,5),(0,1),(1,5)}
    incumbent=np.array([0,2,3,5]);valid=lambda a,b:(a,b) in edges
    assert shortest_subsequence(incumbent,valid).tolist()==incumbent.tolist()
    assert shortest_feasible_path(6,valid).tolist()==[0,1,5]


def test_full_candidate_graph_matches_exhaustive_without_adjacent_edge_assumption():
    rng=np.random.default_rng(333)
    for size in range(2,9):
        for trial in range(5):
            edges={(0,size-1)} if trial==0 else {(i,i+1) for i in range(size-1) if rng.random()<.6}
            edges|={(i,j) for i in range(size) for j in range(i+2,size) if rng.random()<.3}
            paths=[]
            for flags in itertools.product([False,True],repeat=size-2):
                path=[0]+[i+1 for i,f in enumerate(flags) if f]+[size-1]
                if all((a,b) in edges for a,b in zip(path,path[1:])):paths.append(path)
            if paths:assert len(shortest_feasible_path(size,lambda a,b:(a,b) in edges))==min(map(len,paths))
            else:
                with pytest.raises(ValueError,match='No feasible'):
                    shortest_feasible_path(size,lambda a,b:(a,b) in edges)


def test_joint_shortcut_removes_geometry_trap_in_actual_coordinates(tmp_path):
    rec=recording([[(0,0.,0.),(1,-.1,0.),(2,.1,0.),(3,0.,0.)]])
    blocks=make_blocks(rec,10);seed=[np.arange(4)];truth=relations(rec,[5])
    results={}
    for mode in ['GREEDY-PRUNE','DAG-PRUNE']:
        final,_=prune(rec,blocks,seed,[5],.1,truth,mode,10,time.perf_counter()+10,tmp_path)
        results[mode]=final[0].tolist()
    assert results=={'GREEDY-PRUNE':[0,1,2,3],'DAG-PRUNE':[0,3]}


@pytest.mark.parametrize('mode',['GREEDY-PRUNE','DAG-PRUNE'])
def test_geometrically_removable_knot_can_be_required_by_relationship(mode,tmp_path):
    rec=recording([[(0,0.,0.),(1,.1,0.),(2,0.,0.)],[(0,5.05,0.),(1,5.05,0.),(2,5.05,0.)]])
    blocks=make_blocks(rec,10);seed=[np.arange(len(b.rows)) for b in blocks]
    final,_=prune(rec,blocks,seed,[5],.1,relations(rec,[5]),mode,10,time.perf_counter()+10,tmp_path)
    assert 1 in final[0]


def test_certified_geometric_count_skips_without_changing_full_solution(tmp_path):
    from iptc.geo_opt import segment_errors,optimal_knots
    rng=np.random.default_rng(91)
    rec=recording([[(t,float(i*4.95+rng.normal(0,.15)),float(rng.normal(0,.15))) for t in range(12)] for i in range(4)])
    blocks=make_blocks(rec,8);truth=relations(rec,[5,10,20],True)
    lower=optimal_knots(segment_errors(rec,blocks),.2)
    seed,_=refine(rec,blocks,lower,[5,10,20],.2,truth,100,time.perf_counter()+20,'ENDPOINT-REPAIR',tmp_path)
    a,_=prune(rec,blocks,seed,[5,10,20],.2,truth,'FULL-DAG',100,time.perf_counter()+20,tmp_path)
    b,stats=prune(rec,blocks,seed,[5,10,20],.2,truth,'FULL-DAG',100,time.perf_counter()+20,tmp_path,list(map(len,lower)))
    assert all(np.array_equal(x,y) for x,y in zip(a,b,strict=True))
    c,_=prune(rec,blocks,seed,[5,10,20],.2,truth,'FULL-DAG',100,time.perf_counter()+20,tmp_path,list(map(len,lower)),True)
    assert all(np.array_equal(x,y) for x,y in zip(a,c,strict=True))
    d,_=prune(rec,blocks,seed,[5,10,20],.2,truth,'SEED-DAG',100,time.perf_counter()+20,tmp_path,list(map(len,lower)),True,[np.arange(len(b.rows)) for b in blocks])
    assert all(np.array_equal(x,y) for x,y in zip(a,d,strict=True))


def test_shortcut_fast_path_is_bitwise_equal_to_checked_decoder_interpolation():
    rng=np.random.default_rng(5)
    for offset in [0.,1e6,1e12]:
        times=np.cumsum(rng.integers(1,5,60)).astype(np.int64)
        xy=rng.normal(0,1,(60,2))+offset
        for trial in range(30):
            left=int(rng.integers(0,59));right=int(rng.integers(left+1,60))
            assert shortcut_xy(times,xy,left,right).tobytes()==interpolate(times[left:right+1],times[[left,right]],xy[[left,right]]).tobytes()
