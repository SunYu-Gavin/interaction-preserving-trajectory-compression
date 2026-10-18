"""统一论文比较格式：坐标、完整索引和 XOR 差分索引使用相同容器。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .data import Recording
from .geometry import Block,interpolate
from .events import relations,expand_index


def encode(path: Path, rec: Recording, blocks: list[Block], knots: list[np.ndarray],
           radii: list[float], index: np.ndarray | None = None, index_mode: str = 'none') -> int:
    rec.validate()
    if index_mode not in {'none','full','xor'} or (index is None)!=(index_mode=='none'):
        raise ValueError('Index presence must agree with explicit none/full/xor semantics')
    if not radii or len(set(radii))!=len(radii) or min(radii)<=0 or not np.isfinite(radii).all():
        raise ValueError('Unique positive finite query radii required')
    times=[];points=[];positions=[];offsets=[0];koffsets=[0];tracks=[]
    for b,k in zip(blocks,knots,strict=True):
        if not np.issubdtype(k.dtype,np.integer) or not len(k) or k[0]!=0 or k[-1]!=len(b.rows)-1 or np.any(np.diff(k)<=0):
            raise ValueError('Sorted original-point knots with both block endpoints required')
        times.extend(rec.time[b.rows]);positions.extend(k);points.extend(rec.xy[b.rows[k]])
        offsets.append(len(times));koffsets.append(len(points));tracks.append(b.track)
    metadata=dict(format='IPTC-NPZ-2',recording=rec.metadata,tracks=rec.tracks,dt=rec.dt,radii=radii,
                  index_mode=index_mode,coordinate_dtype='float64',compression='zip-deflate')
    arrays=dict(metadata=np.frombuffer(json.dumps(metadata,ensure_ascii=False,sort_keys=True).encode('utf8'),dtype=np.uint8),
                time=np.array(times,dtype=np.int64),block_offsets=np.array(offsets,dtype=np.int64),
                block_track=np.array(tracks,dtype=np.int32),knot_offsets=np.array(koffsets,dtype=np.int64),
                knot_position=np.array(positions,dtype=np.int64),xy=np.array(points,dtype=np.float64).reshape(-1,2))
    if index is not None:
        if index.dtype!=np.int64 or index.ndim!=2 or index.shape[1]!=6:raise ValueError('Index must be an int64 interval table with six columns')
        arrays['event_index']=index
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    return path.stat().st_size


def decode(path: Path) -> tuple[Recording,np.ndarray | None,list[float],str]:
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(z['metadata'].tobytes().decode('utf8'))
        if meta.get('format')!='IPTC-NPZ-2' or meta.get('index_mode') not in {'none','full','xor'}:
            raise ValueError('Unsupported paper artifact format or index semantics')
        t=z['time'];offsets=z['block_offsets'];ko=z['knot_offsets'];bt=z['block_track'];kp=z['knot_position'];anchors=z['xy']
        if len(offsets)!=len(bt)+1 or len(ko)!=len(offsets) or offsets[0]!=0 or offsets[-1]!=len(t) or ko[0]!=0 or ko[-1]!=len(kp) or len(kp)!=len(anchors):
            raise ValueError('Invalid block or knot array boundaries')
        track=np.empty(len(t),dtype=np.int32);xy=np.empty((len(t),2),dtype=np.float64)
        for n,v in enumerate(bt):
            a,b=int(offsets[n]),int(offsets[n+1]);ka,kb=int(ko[n]),int(ko[n+1]);k=kp[ka:kb]
            if b<=a or not len(k) or k[0]!=0 or k[-1]!=b-a-1 or np.any(np.diff(k)<=0):raise ValueError('Invalid stored knot positions')
            xy[a:b]=interpolate(t[a:b],t[a:b][k],anchors[ka:kb]);track[a:b]=v
        index=z['event_index'].copy() if meta['index_mode']!='none' else None
    rec=Recording(meta['recording'],meta['tracks'],track,t.copy(),xy,int(meta['dt']));rec.validate()
    if index is not None and (index.dtype!=np.int64 or index.ndim!=2 or index.shape[1]!=6):raise ValueError('Invalid stored interval table')
    return rec,index,meta['radii'],meta['index_mode']


def query(rec: Recording,index: np.ndarray | None,radii: list[float],index_mode: str) -> dict:
    """部署查询只接收解码产物；XOR 同时修正假阳性和假阴性。"""
    if index_mode=='none' and index is None:return relations(rec,radii)
    if index_mode not in {'full','xor'} or index is None:raise ValueError('Invalid query index semantics')
    correction=expand_index(index,radii,rec.dt)
    if index_mode=='full':return correction
    actual=relations(rec,radii)
    return {float(r):actual[float(r)]^correction[float(r)] for r in radii}


def query_file(path: Path,interval_output: bool = False) -> dict:
    """完整文件查询：完整索引可跳过坐标解码，差分索引计入坐标查询成本。"""
    from .events import intervals
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(z['metadata'].tobytes().decode('utf8'))
        if meta.get('format')!='IPTC-NPZ-2' or meta.get('index_mode') not in {'none','full','xor'}:
            raise ValueError('Unsupported paper query artifact')
        if meta['index_mode']=='full':
            index=z['event_index'];radii=meta['radii'];dt=int(meta['dt'])
            if not interval_output:return expand_index(index,radii,dt)
            result={float(r):set() for r in radii}
            for ri,i,j,start,end,count in index:
                if int(end)<int(start) or int(count)<1 or int(end)-int(start)!=(int(count)-1)*dt:
                    raise ValueError('Invalid stored interval')
                result[float(radii[int(ri)])].add((int(i),int(j),int(start),int(end),int(count)))
            return result
    rec,index,radii,mode=decode(path);events=query(rec,index,radii,mode)
    return {r:intervals(v,rec.dt) for r,v in events.items()} if interval_output else events


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('artifact',type=Path)
    args=parser.parse_args();rec,index,radii,mode=decode(args.artifact)
    result=query(rec,index,radii,mode)
    print(json.dumps(dict(format='IPTC-NPZ-2',index_mode=mode,records=len(rec.time),
                          positive_relations={str(r):len(v) for r,v in result.items()})))


if __name__=='__main__':main()
