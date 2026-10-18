"""Portable command line for the public research implementation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from .data import Recording
from .geometry import make_blocks
from .baselines import raw_knots
from .paper_codec import encode, decode, query_file
from .events import relations, intervals
from .resources import memory_bytes

METHODS = ('RAW','GEOMETRY','RDP-REPAIR','RDP-GREEDY','GEO-GREEDY',
           'MARGIN-DAG','RAW-GREEDY','SEED-DAG','RAW-DAG','FULL-DAG',
           'FULL-DAG-BYTE','FULL-INDEX','XOR-INDEX','XOR-SCAN')


def csv_recording(path, ticks, dt):
    if ticks <= 0 or dt <= 0:
        raise ValueError('Positive ticks-per-second and dt required')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'agent_id','time','x','y'} <= set(reader.fieldnames or []):
            raise ValueError('CSV columns: agent_id,time,x,y; time integer, positions meters')
        rows = [(r['agent_id'],int(r['time']),float(r['x']),float(r['y'])) for r in reader]
    names = sorted({r[0] for r in rows}); mapping = {v:i for i,v in enumerate(names)}
    rows.sort(key=lambda r:(mapping[r[0]],r[1]))
    rec = Recording(dict(source_dataset='user CSV',recording_id=path.name,
                         ticks_per_second=ticks,position_unit='m'),names,
                    np.array([mapping[r[0]] for r in rows],dtype=np.int32),
                    np.array([r[1] for r in rows],dtype=np.int64),
                    np.array([[r[2],r[3]] for r in rows],dtype=np.float64),dt)
    rec.validate()
    return rec


def dispatch(rec, eps, radii, block_seconds, method, deadline, scratch, implementation):
    if method == 'FULL-DAG-BYTE':
        if implementation != 'frozen':
            raise ValueError('Byte extension uses the frozen FULL-DAG initialization')
        from .byte_encoder import compress
    elif method in {'RDP-REPAIR','RDP-GREEDY','GEO-GREEDY'}:
        from .strong_controls import compress
    elif method in {'RAW-GREEDY','SEED-DAG','XOR-SCAN'}:
        from .baseline_extensions import compress
    elif implementation == 'danger-band':
        from .encoder_band_csr import compress
    else:
        from .paper_encoder import compress
    return compress(rec,eps,radii,block_seconds,method,deadline,scratch)


def compress_file(source, output, method, epsilon, block_seconds=2., radii=None,
                  seconds=3600., implementation='frozen'):
    radii = [5.,10.,20.] if radii is None else radii
    if seconds <= 0:
        raise ValueError('Positive encoding time budget required')
    rec, _, _, mode = decode(source)
    if mode != 'none':
        raise ValueError('Input must be normalized RAW coordinates without an event index')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Dataset decoding is separate; encoding starts from normalized arrays.
    with tempfile.TemporaryDirectory(prefix='iptc-') as scratch:
        start = time.perf_counter()
        blocks, knots, index, mode, stages, _ = dispatch(
            rec,epsilon,radii,block_seconds,method,start+seconds,Path(scratch),implementation)
        size = encode(output,rec,blocks,knots,radii,index,mode)
        encoder_s = time.perf_counter()-start
    start = time.perf_counter(); rebuilt, _, _, _ = decode(output)
    decode_s = time.perf_counter()-start
    geometry_error = float(np.linalg.norm(rec.xy-rebuilt.xy,axis=1).max())
    truth = relations(rec,radii); native = relations(rebuilt,radii)
    start = time.perf_counter(); answers = query_file(output)
    query_s = time.perf_counter()-start
    start = time.perf_counter(); answer_intervals = query_file(output,True)
    interval_s = time.perf_counter()-start
    flips = sum(len(answers[r]^truth[r]) for r in radii)
    interval_difference = sum(len(answer_intervals[r]^intervals(truth[r],rec.dt)) for r in radii)
    if geometry_error > epsilon or (method != 'GEOMETRY' and (flips or interval_difference)):
        raise RuntimeError('Output violates the requested geometry or query contract')
    result = dict(method=method,implementation=implementation,epsilon=epsilon,radii=radii,
                  block_seconds=block_seconds,records=len(rec.time),points=sum(map(len,knots)),
                  bytes=size,encoder_s=encoder_s,decode_s=decode_s,file_query_s=query_s,
                  file_interval_query_s=interval_s,max_geometry_error=geometry_error,
                  coordinate_event_difference=sum(len(native[r]^truth[r]) for r in radii),
                  query_event_difference=flips,query_interval_difference=interval_difference,
                  peak_process_bytes=memory_bytes(),stages=stages)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    imp = sub.add_parser('import',help='Convert user-held data to normalized RAW NPZ')
    imp.add_argument('--format',choices=['csv','citysim','interhub'],default='csv')
    imp.add_argument('--input',type=Path,required=True)
    imp.add_argument('--output',type=Path,required=True)
    imp.add_argument('--ticks-per-second',type=int,default=10)
    imp.add_argument('--dt',type=int,default=1)
    imp.add_argument('--root',type=Path)
    imp.add_argument('--member',help='Feather member inside an InterHub cache ZIP')
    imp.add_argument('--records',type=int,default=20000)
    run = sub.add_parser('compress')
    run.add_argument('--input',type=Path,required=True)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--method',choices=METHODS,default='FULL-DAG')
    run.add_argument('--epsilon',type=float,required=True)
    run.add_argument('--radii',type=float,nargs='+',default=[5.,10.,20.])
    run.add_argument('--block-seconds',type=float,default=2.)
    run.add_argument('--seconds',type=float,default=3600.)
    run.add_argument('--implementation',choices=['frozen','danger-band'],default='frozen')
    query = sub.add_parser('query')
    query.add_argument('artifact',type=Path);query.add_argument('--intervals',action='store_true')
    query.add_argument('--output',type=Path,help='Write complete answers as JSON')
    export = sub.add_parser('decode')
    export.add_argument('artifact',type=Path);export.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.command == 'import':
        if a.records < 1:
            raise ValueError('Positive record prefix target required')
        if a.format == 'csv':
            rec = csv_recording(a.input,a.ticks_per_second,a.dt)
        elif a.format == 'citysim':
            from .data import load_citysim
            root = (a.root or a.input.parent).resolve()
            rec = load_citysim(a.input.resolve(),root,dict(position_unit='ft',unit_evidence=['CitySim schema'],
                               records_per_recording=a.records,frame_order_policy='sort',
                               source_dataset='CitySim',source_version='user-held release'))
            rec.metadata.pop('metadata_file',None)
            rec.metadata['source_file'] = rec.metadata['recording_id']
        else:
            if not a.member:
                p.error('--member is required for InterHub')
            from .interhub import load_interhub
            rec = load_interhub(a.member,dict(data_root=str(a.input.parent),cache_archive=a.input.name,
                                  records_per_recording=a.records,unit_evidence=['Native cache schema']))
            rec.metadata['source_file'] = a.input.name+'::'+a.member
        nbytes = encode(a.output,rec,make_blocks(rec,2.),raw_knots(make_blocks(rec,2.)),[5.,10.,20.])
        print(json.dumps(dict(records=len(rec.time),vehicles=len(rec.tracks),bytes=nbytes)))
    elif a.command == 'compress':
        result = compress_file(a.input,a.output,a.method,a.epsilon,a.block_seconds,a.radii,a.seconds,a.implementation)
        print(json.dumps({k:v for k,v in result.items() if k!='stages'}))
    elif a.command == 'decode':
        rec, _, _, _ = decode(a.artifact)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(['agent_id','time','x','y'])
            w.writerows((rec.tracks[v],int(t),float(x),float(y)) for v,t,(x,y) in zip(rec.track,rec.time,rec.xy,strict=True))
    else:
        result = query_file(a.artifact,a.intervals)
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            with a.output.open('x',encoding='utf8') as f:
                json.dump({str(r):sorted(values) for r,values in result.items()},f)
        print(json.dumps({str(r):len(v) for r,v in result.items()}))


if __name__ == '__main__':
    main()
