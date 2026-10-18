"""Serial, fresh-process benchmark from a portable YAML configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg=yaml.safe_load(a.config.read_text(encoding='utf8'))
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for n,source in enumerate(cfg['inputs']):
        for ei,eps in enumerate(cfg['epsilons']):
            for repeat in range(cfg.get('repetitions',1)):
                methods=cfg['methods'];shift=(n+ei+repeat)%len(methods)
                for method in methods[shift:]+methods[:shift]:
                    artifact=out/f'rec{n}_eps{eps}_rep{repeat}_{method}.npz'
                    subprocess.run([sys.executable,'-m','iptc.cli','compress','--input',str(source),
                                    '--output',str(artifact),'--method',method,'--epsilon',str(eps),
                                    '--block-seconds',str(cfg.get('block_seconds',2)),
                                    '--seconds',str(cfg.get('seconds_per_case',3600)),
                                    '--implementation',cfg.get('implementation','frozen'),
                                    '--radii',*map(str,cfg.get('radii',[5,10,20]))],check=True)
                    result=json.loads(artifact.with_suffix('.json').read_text(encoding='utf8'))
                    result.update(recording_index=n,repeat=repeat)
                    rows.append(result)
                    (out/'measurements.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
