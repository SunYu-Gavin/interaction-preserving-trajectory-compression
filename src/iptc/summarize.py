"""Summarize matched benchmark outputs without mixing input cohorts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',default='GEO-GREEDY');a=p.parse_args()
    measurements=json.loads((a.run/'measurements.json').read_text(encoding='utf8'))
    cases={}
    for r in measurements:
        key=(r['recording_index'],r['epsilon'],r['block_seconds'],tuple(r['radii']),r['method'])
        cases.setdefault(key,[]).append(r)
    combined={}
    for key,repeats in cases.items():
        if len({r['bytes'] for r in repeats})!=1:
            raise ValueError('Repeated encodings changed file sizes')
        row={k:float(np.median([r[k] for r in repeats])) for k in
             ['bytes','points','encoder_s','decode_s','file_query_s','file_interval_query_s']}
        combined[key]=row
    methods=sorted({k[-1] for k in combined});output=[]
    expected={k[:-1] for k in combined if k[-1]==a.baseline}
    if not expected:
        raise ValueError('No baseline cases')
    for method in methods:
        subset={k[:-1]:v for k,v in combined.items() if k[-1]==method}
        if set(subset)!=expected:
            raise ValueError('Methods must cover exactly the same configurations')
        savings=[100*(1-r['bytes']/combined[k+(a.baseline,)]['bytes']) for k,r in subset.items()]
        ratios=[r['encoder_s']/combined[k+(a.baseline,)]['encoder_s'] for k,r in subset.items()]
        output.append(dict(method=method,cases=len(subset),baseline=a.baseline,
                           median_bytes=float(np.median([r['bytes'] for r in subset.values()])),
                           median_paired_saving_pct=float(np.median(savings)),
                           smaller=sum(s>0 for s in savings),equal=sum(s==0 for s in savings),larger=sum(s<0 for s in savings),
                           median_encoder_s=float(np.median([r['encoder_s'] for r in subset.values()])),
                           median_paired_encoder_ratio=float(np.median(ratios))))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
