import time
import numpy as np
import pytest
from iptc.local import refine,state_case
from iptc.geometry import make_blocks,independent_knots,reconstruct
from iptc.events import relations
from iptc.codecs import encode,decode
from test_core import recording,flip_recording


@pytest.mark.parametrize("flags,case",[
    ([True,False,False,True],"none"),([False,True,True,True],"cover"),
    ([False,True,False,True],"u"),([False,False,True,True],"v"),
    ([False,False,False,True],"both")])
def test_four_state_constraints(flags,case):
    assert state_case(flags)==case


def test_single_endpoint_allocation_is_cheaper_than_both(tmp_path):
    rec=flip_recording()
    blocks=make_blocks(rec,10)
    coarse=independent_knots(rec,blocks,.1)
    truth=relations(rec,[5,10,20])
    outputs={}
    for mode in ["POINT-REPAIR","LOCAL-JOINT"]:
        knots,stats=refine(rec,blocks,coarse,[5,10,20],.1,truth,20,time.perf_counter()+10,mode,tmp_path)
        path=tmp_path/(mode+".npz")
        encode(path,rec,blocks,knots,[5,10,20])
        rebuilt,_,_=decode(path)
        assert relations(rebuilt,[5,10,20])==truth
        assert np.linalg.norm(rec.xy-rebuilt.xy,axis=1).max()<=.1
        outputs[mode]=stats["local_added_points"]
    assert outputs["LOCAL-JOINT"]==1
    assert outputs["POINT-REPAIR"]==2


@pytest.mark.parametrize("mode",["POINT-REPAIR","ENDPOINT-REPAIR","LOCAL-JOINT"])
def test_insertion_requires_geometry_repair(mode,tmp_path):
    rec=recording([[(0,0,0),(1,-.1,0),(2,.1,0),(3,0,0)],
                   [(0,4.95,0),(1,4.95,0),(2,4.95,0),(3,4.95,0)]])
    blocks=make_blocks(rec,10)
    coarse=independent_knots(rec,blocks,.1)
    truth=relations(rec,[5])
    knots,stats=refine(rec,blocks,coarse,[5],.1,truth,20,time.perf_counter()+10,mode,tmp_path)
    assert stats["geometry_repair_points"]>=1
    assert any(h["geometric_violations"]>0 for h in stats["refinement_history"])
    assert np.linalg.norm(rec.xy-reconstruct(rec,blocks,knots),axis=1).max()<=.1


@pytest.mark.parametrize("mode",["POINT-REPAIR","ENDPOINT-REPAIR","LOCAL-JOINT"])
def test_new_relationship_flips_are_rechecked(mode,tmp_path):
    rec=recording([[(0,0,0),(1,.1,0),(2,0,0)],
                   [(0,5.05,0),(1,5.05,0),(2,5.05,0)],
                   [(0,-4.95,0),(1,-4.85,0),(2,-4.95,0)]])
    blocks=make_blocks(rec,10)
    coarse=independent_knots(rec,blocks,.11)
    truth=relations(rec,[5])
    knots,stats=refine(rec,blocks,coarse,[5],.11,truth,20,time.perf_counter()+10,mode,tmp_path)
    assert stats["local_iterations"]>=2
    assert stats["local_added_points"]<=len(rec.time)


def test_local_boundary_missing_and_iteration_limit(tmp_path):
    rec=recording([[(0,0,0),(1,.1,0),(3,0,0)],
                   [(0,5,0),(1,5,0),(3,5,0)]])
    blocks=make_blocks(rec,10)
    coarse=independent_knots(rec,blocks,.2)
    truth=relations(rec,[5])
    knots,stats=refine(rec,blocks,coarse,[5],.2,truth,10,time.perf_counter()+10,"LOCAL-JOINT",tmp_path)
    assert stats["local_iterations"]==0
    rec=flip_recording();blocks=make_blocks(rec,10);coarse=independent_knots(rec,blocks,.1)
    with pytest.raises(TimeoutError,match="iteration"):
        refine(rec,blocks,coarse,[5],.1,relations(rec,[5]),0,time.perf_counter()+10,"LOCAL-JOINT",tmp_path)


def test_local_random_matches_complete_enumeration(tmp_path):
    rng=np.random.default_rng(8)
    for trial in range(5):
        rec=recording([[(t,float(i*4.98+rng.normal(0,.1)),float(rng.normal(0,.1))) for t in range(15)] for i in range(6)])
        blocks=make_blocks(rec,8);coarse=independent_knots(rec,blocks,.2)
        truth=relations(rec,[5,10,20],True)
        for mode in ["POINT-REPAIR","ENDPOINT-REPAIR","LOCAL-JOINT"]:
            knots,stats=refine(rec,blocks,coarse,[5,10,20],.2,truth,100,time.perf_counter()+20,mode,tmp_path)
            path=tmp_path/f"{trial}_{mode}.npz";encode(path,rec,blocks,knots,[5,10,20])
            rebuilt,_,_=decode(path)
            assert relations(rebuilt,[5,10,20],True)==truth
            assert np.linalg.norm(rec.xy-rebuilt.xy,axis=1).max()<=.2


def test_shared_endpoint_cover_beats_error_first_independent_choices(tmp_path):
    rec=recording([[(0,0,0),(1,.04,0),(2,0,0)],
                   [(0,5.03,0),(1,4.995,.1),(2,5.03,0)],
                   [(0,5.03,0),(1,4.995,-.1),(2,5.03,0)]])
    blocks=make_blocks(rec,10);coarse=independent_knots(rec,blocks,.11)
    truth=relations(rec,[5,10,20])
    counts={}
    for mode in ["POINT-REPAIR","ENDPOINT-REPAIR","LOCAL-JOINT"]:
        knots,stats=refine(rec,blocks,coarse,[5,10,20],.11,truth,20,time.perf_counter()+10,mode,tmp_path)
        counts[mode]=stats["local_added_points"]
        rebuilt=recording([[(int(t),float(x),float(y)) for t,(x,y) in zip(rec.time[rec.track==i],reconstruct(rec,blocks,knots)[rec.track==i])] for i in range(3)])
        assert relations(rebuilt,[5,10,20])==truth
    assert counts=={"POINT-REPAIR":3,"ENDPOINT-REPAIR":2,"LOCAL-JOINT":1}
