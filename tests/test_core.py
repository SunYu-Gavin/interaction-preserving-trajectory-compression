from __future__ import annotations
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pytest
from iptc.data import Recording,load_citysim
from iptc.geometry import make_blocks,simplify,reconstruct,independent_knots
from iptc.codecs import encode,decode
from iptc.events import relations,intervals,spatial_pairs,pack_index,expand_index
from iptc.certificates import constraint_case,build_constraints
from iptc.joint import exact_cover,greedy_cover,small_component_gaps
from iptc.baselines import raw_knots,repair,upgraded_knots,uniform_knots
from iptc.evaluate import evaluate


def recording(trajectories,recording_id="r"):
    tracks = [str(i) for i in range(len(trajectories))]
    rows = [(i,t,x,y) for i,trajectory in enumerate(trajectories) for t,x,y in trajectory]
    rows.sort()
    rec = Recording(dict(source_dataset="unit-test",source_version="1",recording_id=recording_id,
                         source_file="FORBIDDEN.csv",ticks_per_second=1),tracks,
                    np.array([r[0] for r in rows],dtype=np.int32),
                    np.array([r[1] for r in rows],dtype=np.int64),
                    np.array([[r[2],r[3]] for r in rows],dtype=np.float64))
    rec.validate()
    return rec


def flip_recording():
    return recording([[(0,0,0),(1,.1,0),(2,0,0)],[(0,5.05,0),(1,5.05,0),(2,5.05,0)]])


def test_raw_roundtrip_single_two_and_gaps(tmp_path):
    rec = recording([[(0,1.25,2)],[(0,4,5),(1,6,7),(3,8,9)]])
    blocks = make_blocks(rec,2)
    assert [len(b.rows) for b in blocks] == [1,2,1]
    path = tmp_path/"raw.npz"
    encode(path,rec,blocks,raw_knots(blocks),[5.])
    rebuilt,index,radii = decode(path)
    assert np.array_equal(rec.xy,rebuilt.xy)
    assert np.array_equal(rec.time,rebuilt.time)
    assert index is None and radii == [5.]
    with np.load(path,allow_pickle=False) as z:
        assert all(z[k].dtype != object for k in z.files)


def test_recording_namespaces():
    a,b = recording([[(0,0,0)]],"a"),recording([[(0,0,0)]],"b")
    assert a.tracks == b.tracks
    assert (a.metadata["source_dataset"],a.metadata["recording_id"],a.tracks[0]) != (b.metadata["source_dataset"],b.metadata["recording_id"],b.tracks[0])


def test_duplicate_time_nonfinite_and_disorder_rejected():
    with pytest.raises(ValueError,match="Duplicate"):
        recording([[(0,0,0),(0,1,0)]])
    with pytest.raises(ValueError,match="Nonfinite"):
        recording([[(0,np.nan,0)]])
    rec = recording([[(0,0,0),(1,0,0)]])
    rec.time = rec.time[::-1]
    with pytest.raises(ValueError,match="sorted"):
        rec.validate()


@pytest.mark.parametrize("epsilon",[0,.05,.1,.2,.5])
def test_synchronous_bound_after_disk_decode(tmp_path,epsilon):
    t = np.arange(25)
    rec = recording([[(int(s),float(s*s/100),float(np.sin(s))) for s in t]])
    blocks = make_blocks(rec,8)
    knots = independent_knots(rec,blocks,epsilon)
    path = tmp_path/f"{epsilon}.npz"
    encode(path,rec,blocks,knots,[5.])
    rebuilt,_,_ = decode(path)
    assert np.linalg.norm(rec.xy-rebuilt.xy,axis=1).max() <= epsilon


def test_synchronous_not_perpendicular_rdp():
    # 全部点共线，但中间点到同步位置的误差为 8；垂距 RDP 会漏掉。
    t,xy = np.array([0,1,2]),np.array([[0.,0],[9.,0],[2.,0]])
    assert simplify(t,xy,.1).tolist() == [0,1,2]


@pytest.mark.parametrize("distance,positive",[(5.,True),(np.nextafter(5.,0),True),(np.nextafter(5.,10),False)])
def test_exact_query_boundary(distance,positive):
    rec = recording([[(0,0,0)],[(0,distance,0)]])
    assert bool(relations(rec,[5.])[5.]) == positive


def test_small_geometry_error_can_flip_relationship():
    rec = flip_recording()
    blocks = make_blocks(rec,10)
    rebuilt = Recording(rec.metadata,rec.tracks,rec.track,rec.time,reconstruct(rec,blocks,independent_knots(rec,blocks,.1)))
    assert np.linalg.norm(rec.xy-rebuilt.xy,axis=1).max() <= .1
    assert relations(rec,[5.])[5.] ^ relations(rebuilt,[5.])[5.] == {(1,0,1)}


@pytest.mark.parametrize("eu,ev,margin,case",[(.1,.1,1,"none"),(.3,.3,.5,"cover"),(.6,.1,.5,"u"),(.1,.6,.5,"v"),(.6,.6,.5,"both"),(0,0,0,"both")])
def test_five_constraint_cases(eu,ev,margin,case):
    assert constraint_case(eu,ev,margin) == case


def test_certificate_failure_without_actual_flip():
    rec = recording([[(0,0,0),(1,0,.1),(2,0,0)],[(0,5.05,0),(1,5.05,0),(2,5.05,0)]])
    blocks = make_blocks(rec,10)
    coarse = reconstruct(rec,blocks,independent_knots(rec,blocks,.1))
    rebuilt = Recording(rec.metadata,rec.tracks,rec.track,rec.time,coarse)
    assert relations(rec,[5.]) == relations(rebuilt,[5.])
    assert build_constraints(rec,coarse,blocks,[5.],.1).uncertified > 0


def test_cover_exact_vs_independent_bruteforce():
    cost = {0:0,1:4,2:2,3:5}
    edges = {(0,1),(1,2),(2,3)}
    answer,value = exact_cover(cost,edges)
    exhaustive = []
    for subset in itertools.chain.from_iterable(itertools.combinations(cost,k) for k in range(5)):
        if all(set(subset).intersection(e) for e in edges):
            exhaustive.append(sum(cost[v] for v in subset))
    assert value == min(exhaustive)
    greedy,stats = greedy_cover(cost,edges)
    assert all(greedy.intersection(e) for e in edges)
    assert stats["logical_upgrade_bytes"] >= value
    assert 0 in greedy
    forced,stats = greedy_cover(cost,edges,{1})
    assert 1 in forced
    assert small_component_gaps(cost,edges,set())[0]["absolute_gap"] >= 0


def test_greedy_nonoptimal_example_is_reported():
    cost = {0:3,1:2,2:2}
    edges = {(0,1),(0,2),(1,2)}
    opt,value = exact_cover(cost,edges)
    chosen,stats = greedy_cover(cost,edges)
    assert value == 4
    assert stats["logical_upgrade_bytes"] >= value
    assert small_component_gaps(cost,edges,set())[0]["heuristic_bytes"] == stats["logical_upgrade_bytes"]


def test_spatial_filter_matches_all_pairs():
    rng = np.random.default_rng(24)
    for radius in [.1,5,20]:
        xy = rng.normal(size=(70,2))*20
        expected = {(i,j) for i,j in itertools.combinations(range(70),2) if np.linalg.norm(xy[i]-xy[j]) <= radius}
        assert {tuple(v) for v in spatial_pairs(xy,radius)} == expected
    rec = recording([[(t,float(i*3),float(t/2)) for t in [0,1,3]] for i in range(8)])
    assert relations(rec,[5,10,20]) == relations(rec,[5,10,20],exhaustive=True)


def test_intervals_break_on_missing_and_time_gaps():
    events = {(t,0,1) for t in [0,1,3,4,8]}
    assert intervals(events,1) == {(0,1,0,1,2),(0,1,3,4,2),(0,1,8,8,1)}
    rec = recording([[(0,0,0),(1,0,0),(3,0,0)],[(0,1,0),(3,1,0)]])
    assert intervals(relations(rec,[5])[5.],1) == {(0,1,0,0,1),(0,1,3,3,1)}


def test_repair_and_joint_post_serialization(tmp_path):
    rec = flip_recording()
    blocks = make_blocks(rec,10)
    coarse = independent_knots(rec,blocks,.1)
    truth = relations(rec,[5.,10.])
    knots,stats = repair(rec,blocks,coarse,[5.,10.],truth,10,time.perf_counter()+10,tmp_path)
    assert stats["repair_iterations"] >= 1
    assert stats["repair_upgrades"] == 2
    c = build_constraints(rec,reconstruct(rec,blocks,coarse),blocks,[5.,10.],.1)
    costs = {n:24*(len(b.rows)-len(k)) for n,(b,k) in enumerate(zip(blocks,coarse))}
    selected,_ = greedy_cover(costs,c.edges,c.forced)
    for name,rep in [("repair",knots),("joint",upgraded_knots(blocks,coarse,selected))]:
        path = tmp_path/(name+".npz")
        encode(path,rec,blocks,rep,[5.,10.])
        decoded,index,_ = decode(path)
        metrics,_ = evaluate(rec,decoded,[5.,10.],.1,truth)
        assert all(m["flips"]==0 and m["geometry_pass"] and m["interval_symmetric_difference"]==0 for m in metrics)
    again,stats = repair(rec,blocks,raw_knots(blocks),[5.,10.],truth,10,time.perf_counter()+10)
    assert stats["repair_iterations"] == 0
    with pytest.raises(TimeoutError):
        repair(rec,blocks,coarse,[5.],{5.:truth[5.]},0,time.perf_counter()+10)


def test_repair_rechecks_new_flips_after_upgrade(tmp_path):
    # B2: A/B 在 t1 翻转；升级 A/B 后，A/C 的原本抵消误差消失并产生新翻转。
    rec = recording([[(0,0,0),(1,.1,0),(2,0,0)],
                     [(0,5.05,0),(1,5.05,0),(2,5.05,0)],
                     [(0,-4.95,0),(1,-4.85,0),(2,-4.95,0)]])
    blocks = make_blocks(rec,10)
    coarse = independent_knots(rec,blocks,.11)
    knots,stats = repair(rec,blocks,coarse,[5.],relations(rec,[5.]),10,time.perf_counter()+10)
    assert stats["repair_iterations"] == 2
    assert stats["repair_upgrades"] == 3


def test_index_coordinate_and_end_to_end_separate(tmp_path):
    rec = flip_recording()
    blocks = make_blocks(rec,10)
    truth = relations(rec,[5.,10.])
    index = pack_index(truth,[5.,10.],1)
    path = tmp_path/"index.npz"
    encode(path,rec,blocks,independent_knots(rec,blocks,.1),[5.,10.],index)
    decoded,loaded,radii = decode(path)
    metrics,_ = evaluate(rec,decoded,radii,.1,truth,loaded)
    assert metrics[0]["flips"] == 1 and metrics[0]["index_flips"] == 0
    assert expand_index(loaded,radii,1) == truth


def test_decoder_in_subprocess_cannot_access_original(tmp_path):
    rec = flip_recording()
    blocks = make_blocks(rec,10)
    path = tmp_path/"only_artifact.npz"
    encode(path,rec,blocks,raw_knots(blocks),[5.])
    code = """
import sys
def guard(event,args):
    if event == 'open' and str(args[0]).endswith('.csv'):
        raise RuntimeError('Original CSV access forbidden')
sys.addaudithook(guard)
from iptc.codecs import decode
from iptc.events import relations
from pathlib import Path
rec,index,radii=decode(Path(sys.argv[1]))
assert len(rec.time)==6
assert len(relations(rec,radii)[5.])==1
print('self-contained decode/query passed')
"""
    output = subprocess.check_output([sys.executable,"-c",code,str(path)],text=True,cwd=tmp_path)
    assert "passed" in output


def test_uniform_false_positive_is_not_missed():
    rec = recording([[(0,0,0),(1,-.2,0),(2,0,0)],[(0,4.95,0),(1,4.95,0),(2,4.95,0)]])
    blocks = make_blocks(rec,10)
    decoded = Recording(rec.metadata,rec.tracks,rec.track,rec.time,reconstruct(rec,blocks,uniform_knots(blocks,2)))
    metrics,_ = evaluate(rec,decoded,[5.],.05,relations(rec,[5.]))
    assert metrics[0]["false_positive"] == 1
    assert not metrics[0]["geometry_pass"]


def test_joint_numerical_boundary_exact_fallback(tmp_path):
    rec = recording([[(0,0,0),(1,0,0),(2,0,0)],[(0,5,0),(1,5,0),(2,5,0)]])
    blocks = make_blocks(rec,10)
    knots = independent_knots(rec,blocks,.1)
    c = build_constraints(rec,reconstruct(rec,blocks,knots),blocks,[5.],.1)
    assert c.forced == {0,1}


def test_joint_random_small_matches_exhaustive_relations(tmp_path):
    rng = np.random.default_rng(4)
    for iteration in range(8):
        rec = recording([[(t,float(i*4.99+rng.normal(0,.05)),float(rng.normal(0,.05))) for t in range(10)] for i in range(5)])
        blocks = make_blocks(rec,5)
        coarse = independent_knots(rec,blocks,.2)
        constraints = build_constraints(rec,reconstruct(rec,blocks,coarse),blocks,[5,10,20],.2)
        costs = {n:24*(len(b.rows)-len(k)) for n,(b,k) in enumerate(zip(blocks,coarse))}
        selected,stats = greedy_cover(costs,constraints.edges,constraints.forced)
        path = tmp_path/f"joint{iteration}.npz"
        encode(path,rec,blocks,upgraded_knots(blocks,coarse,selected),[5,10,20])
        rebuilt,_,_ = decode(path)
        assert relations(rec,[5,10,20],True) == relations(rebuilt,[5,10,20],True)
        assert np.linalg.norm(rec.xy-rebuilt.xy,axis=1).max() <= .2


def test_resource_monitor_repeated_calls_do_not_accumulate_types():
    from iptc.resources import memory_bytes
    initial = memory_bytes()
    for _ in range(4000):
        measured = memory_bytes()
        assert measured > 0
    assert memory_bytes()-initial < 8*1024**2
