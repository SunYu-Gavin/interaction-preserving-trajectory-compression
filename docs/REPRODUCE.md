# Reproduction

## Environment

Install Python 3.12+ and `python -m pip install -e ".[test]"`. NumPy and PyYAML are required; PyArrow is optional for native InterHub import. `requirements-recorded.txt` records the measured environment, while `pyproject.toml` declares portable requirements. Outputs depend on NumPy/DEFLATE versions and metadata as well as the selected anchors.

## Run the example

```bash
python -m iptc.cli import --input examples/synthetic.csv --output runs/raw.npz --ticks-per-second 10
python -m iptc.benchmark --config configs/demo.yaml
python -m iptc.cli decode runs/demo/rec0_eps0.1_rep0_FULL-DAG.npz --output runs/reconstructed.csv
python -m iptc.cli query runs/demo/rec0_eps0.1_rep0_XOR-INDEX.npz --intervals --output runs/intervals.json
python -m iptc.summarize --run runs/demo --output runs/demo/summary.csv --baseline GEO-GREEDY
```

Each compressed file receives a sibling JSON containing points, actual file bytes, geometry error, coordinate/query discrepancies, complete encoding and query times, and whole-process peak memory. The query command reports answer counts and optionally writes complete answers. Indexed files must be queried through this interface rather than by ignoring their index.

## Reproduce the 48 configurations

1. Obtain the user-held data listed in `inputs_final.json`, following [DATA.md](DATA.md).
2. Import each recording to `data/recN_raw.npz`. For native prefix readers, use the manifest's actual selected record count as the target: the complete-frame boundary is identical. Match source hashes, native time range, units, all-vehicle selection and recorded count. No original source paths are needed.
3. Run `python -m iptc.benchmark --config configs/paper.yaml` from the repository root. The fixed error bounds are 0.05/0.1/0.2/0.5 m, radii 5/10/20 m, and block duration 2 s.
4. Compare fresh method outputs within each recording/error configuration. A positive paired byte saving is `100 * (1 - method_bytes / comparator_bytes)`. Take the median of these paired values, not a ratio of median file sizes.

Historical containers include the experiment metadata available on the original machine. Public import uses portable source IDs rather than absolute paths; source-version and provenance fields may also differ. Thus fresh NPZ byte counts/hashes need not exactly match the recorded historical containers. Historical full-file measurements, the unchanged common encoding format, per-case values and source fingerprints are supplied separately. Use identical imported inputs and metadata for all methods in a fresh comparison.

## Timing and memory

The public runner executes each method-case in a fresh child process and rotates method order. Encoding starts after normalized RAW loading and includes method-required initialization, optimization and final serialization. Subsequent geometry/query evaluation is separate. File queries start from the artifact on disk. Peak process memory includes input loading, encoding and subsequent evaluation; it is not an isolated encoder-only allocation.

The 12-case recorded three-repeat timing cohort differs from the 48-case storage cohort. Do not combine their medians. Set `repetitions: 3` in a copied config for new repeated timing, and aggregate the per-case medians before computing paired ratios.

FULL-DAG-BYTE in the public runner executes the complete original FULL-DAG and the byte extension from RAW. The recorded extension-only measurements in `results/byte_refinement_measurements.json` instead start from saved FULL-DAG outputs. Their median 38.792 s is not complete encoder time and was collected with cases distributed across processes. It cannot be added to a median from another cohort to infer an end-to-end speed ratio.

## Tests

`python -m pytest -q` exercises geometry, missing observations, exact threshold boundaries, containers, graph paths, relationship repair, metadata handling, danger-band equivalence and the exact byte-change objective. The synthetic demo checks that the public import/compress/query entry points run without any private dataset.
