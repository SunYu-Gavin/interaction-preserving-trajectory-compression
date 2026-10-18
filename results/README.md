# Recorded experimental results

These files contain aggregate metrics and non-coordinate provenance only. They contain no trajectories or decoder-ready real-data artifacts.

| File | Coverage |
|---|---|
| `storage_all48.csv` | All 14 core methods; file-size medians, paired savings, win/tie/loss counts |
| `storage_per_case_all48.csv` | All 48 matched configurations; actual full-file byte counts for every core method |
| `historical_storage_measurements.json` | Original 480 method-case records, including points and costs |
| `byte_refinement_measurements.json` | New byte-extension point/byte counts, geometric error, event/interval differences, artifact hashes and refinement-only time |
| `timing_12cases_3repeats.csv` | Independent repeated-timing cohort; aggregate per-method costs |
| `historical_timing_measurements.json` | Per-case repeated-timing medians for the original methods |
| `strengthened_baseline_timing48_single.csv` | Six methods in the separate serial single-encoding cohort |
| `external24.csv` | Historical native and repaired external comparisons, 24 cases |
| `scalability_all.csv` | All 26 original duration, concurrency, density and temporal-copy measurements |

`median_kib` is the median actual complete file size divided by 1024. Savings are first calculated for each matched configuration and then summarized by the median. A ratio of median file sizes gives a different statistic. In `storage_all48.csv`, `new_*` means FULL-DAG-BYTE relative to the row's comparator; positive savings mean the new version is smaller.

| Main comparison | Median savings | Smaller files |
|---|---:|---:|
| Original FULL-DAG vs GEO-GREEDY | 1.108% | 48/48 |
| FULL-DAG-BYTE vs GEO-GREEDY | 1.986% | 48/48 |
| FULL-DAG-BYTE vs original FULL-DAG | 0.866% | 48/48 |
| FULL-DAG-BYTE vs XOR-INDEX | 1.158% | 34/48 |

Both coordinate-preserving methods and indexed answers satisfy their declared contracts; geometry-only and native external methods are explicitly distinguished. No formal INTERACTION data were used. Input identity and ranges are in `../docs/inputs_final.json`.

The byte extension's controlled complete timing/query/peak-memory comparison is not available in these records. The 26 scalability entries belong to the original implementations; the 10,281,906-observation entry is controlled temporal replication, not newly collected real observations. Individual external and strengthened-baseline times use their separately stated cohorts and are not mixed with the three-repeat cohort.
