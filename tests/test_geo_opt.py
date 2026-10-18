import itertools
import numpy as np
from iptc.geo_opt import segment_errors,optimal_knots
from iptc.geometry import make_blocks,reconstruct
from iptc.local import refine
from iptc.prune import prune
from iptc.events import relations
import time
from test_core import recording


def test_geometric_dag_has_minimum_count_over_all_original_candidates():
    rng=np.random.default_rng(53)
    for trial in range(6):
        rec=recording([[(t,float(t),float(rng.normal(0,.15))) for t in range(8)]])
        blocks=make_blocks(rec,20);graphs=segment_errors(rec,blocks)
        for eps in [0.,.05,.2,.5]:
            knots=optimal_knots(graphs,eps);best=8
            for flags in itertools.product([False,True],repeat=6):
                path=[0]+[i+1 for i,f in enumerate(flags) if f]+[7]
                if all(graphs[0][a,b]<=eps for a,b in zip(path,path[1:])):best=min(best,len(path))
            assert len(knots[0])==best
            assert np.linalg.norm(reconstruct(rec,blocks,knots)-rec.xy,axis=1).max()<=eps


def test_alternating_relationship_can_require_all_original_anchors(tmp_path):
    n=11
    rec=recording([[(t,float(t),0.) for t in range(n)],
                   [(t,float(t),4.9 if t%2==0 else 5.1) for t in range(n)]])
    blocks=make_blocks(rec,20);coarse=optimal_knots(segment_errors(rec,blocks),.2)
    assert sum(map(len,coarse))==4
    truth=relations(rec,[5],True)
    seed,_=refine(rec,blocks,coarse,[5],.2,truth,100,time.perf_counter()+10,'ENDPOINT-REPAIR',tmp_path)
    final,_=prune(rec,blocks,seed,[5],.2,truth,'FULL-DAG',100,time.perf_counter()+10,tmp_path)
    assert len(final[0])==2 and len(final[1])==n
