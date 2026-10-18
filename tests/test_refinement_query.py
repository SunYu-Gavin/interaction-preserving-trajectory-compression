import numpy as np
import pytest
from iptc.refinement_query import RefinementQuery
from iptc.events import relations
from iptc.data import Recording
from test_core import recording


@pytest.mark.parametrize("offset",[0.,1e7,1e12])
def test_cached_candidates_match_exhaustive_and_expand(offset):
    rng=np.random.default_rng(437)
    rec=recording([[(t,offset+float(i*4.97+rng.normal(0,.2)),offset+float(rng.normal(0,.1)))
                    for t in [0,1,3,4]] for i in range(10)])
    query=RefinementQuery(rec,[5,10,20],.1)
    for error in [.05,.09,.8,0.,1.5]:
        xy=rec.xy+rng.uniform(-error,error,rec.xy.shape)
        rebuilt=Recording(rec.metadata,rec.tracks,rec.track,rec.time,xy,rec.dt)
        assert query.relations(rebuilt)==relations(rebuilt,[5,10,20],True)
    assert query.rebuilds>=2


def test_distant_original_pair_can_become_positive():
    rec=recording([[(0,0.,0.)],[(0,30.,0.)]])
    query=RefinementQuery(rec,[5],.1)
    assert query.relations(rec)=={5.:set()}
    rebuilt=Recording(rec.metadata,rec.tracks,rec.track,rec.time,np.array([[14.,0.],[16.,0.]]))
    assert query.relations(rebuilt)=={5.:{(0,0,1)}}
    assert query.rebuilds==2


def test_exact_threshold_and_identity_checks():
    rec=recording([[(0,0.,0.),(2,0.,0.)],[(0,5.,0.),(2,5.,0.)]])
    query=RefinementQuery(rec,[5],0.)
    assert query.relations(rec)==relations(rec,[5],True)
    bad=Recording(rec.metadata,rec.tracks,rec.track,rec.time+1,rec.xy)
    with pytest.raises(ValueError,match="identical"):
        query.relations(bad)
