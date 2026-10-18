# Methods and guarantees

## Contract

Coordinates are float64 planar vehicle centers in meters. Timestamps are integer ticks; `ticks_per_second` and the native step `dt` are explicit. Each recording is canonicalized by vehicle and then time. Duplicate vehicle/time keys and nonfinite positions are rejected. Gaps split blocks; no missing points are filled. Fixed-duration blocks use a common recording-time origin, and retain their endpoints.

For every original observation, synchronous Euclidean reconstruction error must not exceed epsilon. For each declared radius, the predicate is reconstructed center distance `<= radius` at common valid original times. Consecutive intervals break at missing observations or a native-step gap. No continuous-time collision, arbitrary-radius, intent or causal guarantee is made.

## Available methods

| Identifier | Representation / optimization |
|---|---|
| RAW | Every original point |
| GEOMETRY | Minimum-point geometric shortcut path; no relationship guarantee |
| RDP-REPAIR | Synchronous-error recursive simplification followed by endpoint repair |
| RDP-GREEDY | RDP-REPAIR followed by feasible greedy point deletion |
| GEO-GREEDY | Geometric minimum-point simplification, endpoint repair, greedy deletion |
| MARGIN-DAG | Geometric paths under independent relationship-margin budgets |
| RAW-GREEDY | Feasible greedy deletion starting from RAW |
| SEED-DAG | Fixed repaired seed candidate pool, permitting candidate reuse |
| RAW-DAG | Full original candidate paths from RAW initialization |
| FULL-DAG | Full original candidate paths from repaired geometric initialization |
| FULL-DAG-BYTE | FULL-DAG followed by two actual-byte refinement sweeps |
| FULL-INDEX | Geometric coordinates plus all original declared event intervals |
| XOR-INDEX | Geometric coordinates plus symmetric-difference event intervals |
| XOR-SCAN | Select actual smallest total bytes across five geometric budgets and RAW plus differences |

Coordinate-preserving methods satisfy both geometry and declared relationship requirements. Complete and difference indexes provide exact answers through `paper_codec.query_file`; their decoded coordinates alone can change relationships. Query consumers must respect the representation's index mode.

## Implementations

`--implementation frozen` uses `paper_encoder.py`, the encoder used for the recorded final storage results. `--implementation danger-band` selects `encoder_band_csr.py` for its supported core methods. It filters dynamically tested constraints using the geometric error bound and uses a predicate CSR layout. The strong controls and additional controls retain their recorded encoders regardless of that switch. The byte extension always uses the frozen FULL-DAG initialization; combining it with the danger-band switch is rejected rather than silently choosing another initialization.

Source-module correspondence is recorded in `implementation.json`. Packaging changes affect data imports and entry points; copied algorithm modules retain their original source hashes.

## Actual-byte extension

Original FULL-DAG accepts strict retained-point-count improvements. Its conditional minimum-point guarantee concerns its path solver, and its fixed-point property concerns its converged original stage.

The extension traverses the same blocks twice. With other vehicles fixed, a bounded deterministic path list retains up to eight alternatives at the shortest count and up to four alternatives with one extra anchor per DAG vertex. The final candidate list is evaluated using actual NPZ DEFLATE byte changes. Only the compressed knot offsets, knot positions and retained float64 coordinates vary; fixed metadata/time/block streams and ZIP overhead cancel in a file-size difference. An update is accepted only if total file bytes strictly decrease. The resulting coordinates are used by subsequent block decisions.

The extension preserves the feasibility contract. It may retain more points to produce fewer bytes, so the original final-output point-count stationarity claim is not transferred to the extension. Two sweeps are a fixed search budget, not a byte-optimality claim.

## Storage

`IPTC-NPZ-2` uses NumPy arrays without pickle objects and ZIP DEFLATE. A file includes metadata, vehicle mapping, all valid times, block boundaries, retained anchor positions and their original float64 coordinates, plus an optional complete/difference interval index. Decoding and queries require only that file. `codecs.py` implements the legacy internal `IPTC-NPZ-1` container used during original solver iterations; public final outputs use version two.
