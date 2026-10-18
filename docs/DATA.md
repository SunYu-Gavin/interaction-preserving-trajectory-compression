# Data

## Included example

`examples/synthetic.csv` is generated from simple mathematical trajectories. It contains no observations copied from CitySim, nuPlan, Lyft or InterHub.

The portable CSV interface expects:

```csv
agent_id,time,x,y
a,0,0.0,0.0
a,1,0.3,0.0
b,0,0.0,5.0
b,1,0.3,5.0
```

`time` is an integer tick, `x,y` are meters, and `agent_id` is a string. Set ticks per second and native step explicitly. Rows may be unordered: import sorts by vehicle/time, rejecting duplicates and nonfinite coordinates. Time gaps are preserved. CSV import uses all supplied rows, while native readers support a complete-frame prefix target.

## Real datasets

- [CitySim provider repository](https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset): this reader expects `frameNum`, `carId`, `carCenterXft`, `carCenterYft` and the corresponding `metadata.xlsx`. Coordinates are converted from feet to meters. Frame rate is read from metadata, not inferred.
- [InterHub provider repository](https://github.com/zxc-tju/InterHub): the implemented native reader uses the nuPlan and Lyft cached vehicle tables from the locally used v7 archive. Each selected Feather file is paired with its native scene metadata. Install `.[native]` for PyArrow. The limited metadata reader interprets container syntax without invoking cached Python classes or reducers.

Obtain datasets and permissions from their providers. This repository does not mirror datasets or distribute normalized/decoded/compressed copies of real trajectories. The experiments do not use the official INTERACTION dataset.

## Import examples

CitySim: set `--root` to the directory relative to which the manifest recording ID is resolved. Keep the dataset's metadata in its original directory structure.

```bash
python -m iptc.cli import --format citysim --input "data/CitySim/location/Trajectories/recording.csv" --root data/CitySim --records 40000 --output data/rec0_raw.npz
```

The CitySim importer checks the full CSV, sorts frames stably and selects a complete-frame prefix. The selected count may exceed the requested target to avoid dropping vehicles from the final frame.

InterHub: `--input` is the user-held ZIP, and `--member` is the exact cache member from `inputs_final.json`.

```bash
python -m pip install -e ".[native]"
python -m iptc.cli import --format interhub --input data/native-cache.zip --member "CACHE_MEMBER_FROM_MANIFEST" --records 15000 --output data/rec2_raw.npz
```

All vehicle rows including ego are retained. Position units, native dt, source-member hash and scene metadata hash are kept with the imported data. Data import does not execute model training or download additional files.
