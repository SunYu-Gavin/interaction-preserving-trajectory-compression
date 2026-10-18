# Interaction-Preserving Compression of Multi-Agent Trajectory Databases

[中文说明](README.zh-CN.md)

CPU research implementation of joint trajectory compression for autonomous-driving scenario archives. The method bounds synchronous positional error while preserving declared proximity relationships directly in reconstructed coordinates at the original common valid sampling times.

## Quick start

Python 3.12 or later:

```bash
python -m venv .venv
# Activate the environment using your shell's usual command.
python -m pip install -e ".[test]"
python -m pytest -q
python -m iptc.cli import --input examples/synthetic.csv --output runs/raw.npz --ticks-per-second 10
python -m iptc.benchmark --config configs/demo.yaml
python -m iptc.cli query runs/demo/rec0_eps0.1_rep0_FULL-DAG-BYTE.npz --intervals
```

The 90-row example is generated synthetic data for demonstrating the interface; it is not experimental evidence. Outputs are created exclusively: use a fresh output name or directory for a new run.

## Methods

| Method | Purpose |
|---|---|
| RAW-DAG | Conditional path optimization initialized from original samples |
| FULL-DAG | Conditional path optimization initialized from repaired geometric simplification |
| FULL-DAG-BYTE | Two actual-byte refinement sweeps after the original FULL-DAG |
| GEO-GREEDY / RDP-GREEDY | Strong geometric simplification, relationship repair and greedy deletion |
| FULL-INDEX / XOR-INDEX | Exact declared answers via complete or difference event indexes |
| RAW-GREEDY / SEED-DAG / XOR-SCAN | Optimizer, candidate-pool and index-budget controls |

All 14 supported method names, implementation variants and guarantees are described in [Methods](docs/METHODS.md).

## Recorded storage results

The fixed final cohort has 12 recordings/scenes and four error bounds (48 configurations), with proximity radii 5, 10 and 20 m in every configuration.

| Version | Median actual-byte savings vs GEO-GREEDY | Smaller files |
|---|---:|---:|
| FULL-DAG | 1.108% | 48/48 |
| FULL-DAG-BYTE | 1.986% | 48/48 |

The byte extension further reduces all 48 FULL-DAG files. Its median additional savings are 0.866%. Geometry, declared events and their intervals are preserved. The extension trades additional encoding search for storage. Complete controlled encoding/query/memory comparisons for the extension are not yet available; its recorded refinement time starts from saved FULL-DAG outputs.

See [result definitions and all tables](results/README.md). These are recorded experimental results, not expected performance on the synthetic example or arbitrary datasets.

## Data and reproduction

- [Input format and dataset access](docs/DATA.md)
- [Reproduction commands and measurement scope](docs/REPRODUCE.md)
- [Final input manifest](docs/inputs_final.json)
- [Dependencies and external baseline scope](docs/DEPENDENCIES.md)

Real trajectory files, native caches, compressed trajectory artifacts and manuscripts are not distributed. Experiments use CitySim and the nuPlan/Lyft portions of InterHub, not the official INTERACTION dataset. Users obtain the data from its providers and import their own copies. The included data readers require no model training or GPU.

## Project status

Research code for the manuscript titled above. Citation and publication information will be added when available; see [Publication information](docs/PUBLICATION.md).

No LICENSE file is included in this snapshot.

## Acknowledgments

This work is supported by the General Program of National Natural Science Foundation of China (No.52672414) and the Youth Program of the National Natural Science Foundation of China (No.52602596).
