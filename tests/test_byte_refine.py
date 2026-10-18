import time
import numpy as np
from iptc.byte_refine import variable_bytes, candidate_paths, refine_bytes
from iptc.byte_encoder import compress
from iptc.paper_codec import encode, decode
from iptc.geometry import make_blocks
from iptc.events import relations
from test_core import recording


def test_stream_objective_matches_actual_file_differences(tmp_path):
    rec = recording([[(i, float(i), float(i % 3)*.01) for i in range(12)],
                     [(i, float(i), 5.+float(i % 2)*.01) for i in range(12)]])
    blocks = make_blocks(rec, 5)
    offsets = []
    for stride in [1, 2, 3, 5]:
        knots = [np.union1d(np.arange(0, len(b.rows), stride), [len(b.rows)-1]).astype(np.int64) for b in blocks]
        size = encode(tmp_path/f'{stride}.npz', rec, blocks, knots, [5.])
        offsets.append(size-variable_bytes(rec, blocks, knots))
    assert len(set(offsets)) == 1


def test_candidate_graph_does_not_assume_original_adjacent_edges():
    allowed = {(0, 2), (2, 4), (0, 3), (3, 4)}
    paths = candidate_paths(5, lambda a, b: (a, b) in allowed)
    assert {tuple(p) for p in paths} == {(0, 2, 4), (0, 3, 4)}


def test_fresh_encoder_preserves_threshold_boundary_and_file_nonincrease(tmp_path):
    rec = recording([[(0, 0., 0.), (1, .1, 0.), (2, 0., 0.), (3, .05, 0.)],
                     [(0, 0., 5.), (1, .1, 5.), (2, 0., 5.), (3, .05, 5.)]])
    from iptc.paper_encoder import compress as frozen
    b, initial, _, _, _, _ = frozen(rec, .1, [5.], 10, 'FULL-DAG', time.perf_counter()+30, tmp_path/'old')
    b, knots, _, _, _, truth = compress(rec, .1, [5.], 10, 'FULL-DAG-BYTE', time.perf_counter()+30, tmp_path/'new')
    assert variable_bytes(rec, b, knots) <= variable_bytes(rec, b, initial)
    artifact = tmp_path/'new.npz'
    encode(artifact, rec, b, knots, [5.])
    rebuilt, _, _, _ = decode(artifact)
    assert np.linalg.norm(rebuilt.xy-rec.xy, axis=1).max() <= .1
    assert relations(rebuilt, [5.]) == truth
