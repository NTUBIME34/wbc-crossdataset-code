# Reproduction plan and executable preflight

`study_tasks.json` maps the main paper and supplement to actual existing commands,
inputs, checkpoints, expected output files and remaining manual steps. It is a coverage
registry, not a statement that all experiments have been rerun. The detailed scientific
definitions and references remain in `docs/PAPER_MAP.md`.

Three distinct operations are available:

```bash
python -B reproduce/run_study.py --plan
python -B reproduce/run_study.py --check --tasks cnn-external,dinobloom-external \
  --config /path/to/local_reproduction.json --keep-going
python -B reproduce/run_study.py --run --tasks cnn-external,dinobloom-external \
  --config /path/to/local_reproduction.json
```

Planning needs no data, weights or scientific Python dependencies. `--check` never runs
models. It checks the selected interpreter and every requested seed's checkpoint and
input path; it does not establish image membership, model contents, correct library
versions or numerical agreement. The thirteen released weight files do not cover the
CNN baselines, the other recipe variants and the sixty LODO models; retrain them with
`train/scripts/` (run folders under `MODEL_ROOT`, layout in `eval/README.md`). Simply
rerunning the public verifier is not an alternative to supplying those models.

Example configuration (keep the real paths private):

```json
{
  "variables": {
    "DATA_ROOT": "/data/wbc_prepared",
    "MODEL_DIR": "/data/released_core_weights",
    "MODEL_ROOT": "/data/extended_models/model_root",
    "OUT_ROOT": "/data/reproduction_outputs",
    "DEVICE": "cuda"
  },
  "interpreters": {
    "journal_env": "/opt/conda/envs/journal_env/bin/python",
    "cnn_env": "/opt/conda/envs/cnn_env/bin/python",
    "detector_env": "/opt/conda/envs/detector_env/bin/python",
    "cellpose_env": "/opt/conda/envs/cellpose_env/bin/python"
  }
}
```

Set interpreters to the environments actually installed on your machine. If none is
specified for a task, its `PYTHON` environment variable or the runner's interpreter is
used, and recorded; the environment label in the registry does not install or attest to
its versions. Any prepared-data variable in an underlying runner can also be configured
under `variables`, including `WBCB_P1_CROPS`, `CORPUS_ROOT`, `INTERNAL_TEST_ROOT`,
`LODO_TEST_ROOT_LMU` and the other held-out pools. `MODEL_DIR` contains the original
flat thirteen-file release; `MODEL_ROOT` contains the additional structured checkpoints.
Use absolute paths. Do not set `OUT_ROOT` to the source repository or to an original
evidence archive. Model/evaluation scripts may emit benchmark image IDs and montages;
all outputs remain private and outside this repository.

## Suggested execution order

1. `offline-regressions`, `public-evidence`: code fixtures and retained public evidence.
2. Supply/verify datasets and retained crop/split manifests. Existing folder names or
   matching class totals are insufficient to attest to the original held-out split.
3. Inference: `proposed-external`, `bpaco-external`, `cnn-external`,
   `dinobloom-external`, `resnet-configurations`, `recipe-comparators`,
   `policy-sweep`, `no-flip-comparison`, `full-pipeline`, `localization-routes`,
   `resolution-sweep`, `proposed-internal-mapped`, and the four `lodo-*` arms.
4. Statistics/figures: `proposed-bootstrap`, `cnn-bootstrap-*`, `cnn-per-class-*`,
   `bpaco-paired-statistics`, `recipe-statistics`, `thresholds`, `calibration`,
   `lodo-statistics-*`, `fig3`, `figS2`.
5. Assemble the regenerated private evidence layout described in `analysis/README.md`
   and run `private-evidence`. Inspect the verifier's PASS/FAIL/SKIPPED counts; a zero
   exit with remaining SKIPPED checks is not whole-study reproduction.

Compare newly generated predictions with public retained counts explicitly:

```bash
python -B reproduce/compare_results.py \
  --run backbone_comparison/swin_proposed_seed42 /data/new_run/seed_42/predictions.csv \
  --out /data/new_run/proposed_seed42_comparison.json
```

`compare-proposed-42` (also `62`, `82`) and `compare-cnn-<model>-<seed>` are registered
versions of this numerical check. They compare independently assembled confusion counts
and metric deltas, not only command exits. For a coverage-aware full-pipeline comparison,
also pass `--detections REFERENCE_KEY /path/to/detections.csv` to the comparison tool.
Its JSON preserves exact-count and printed-precision differences; a difference returns
nonzero rather than being silently relabeled as a match. A check that compares counts
does not establish probability-vector identity, retraining equivalence or whole-study
reproduction.

Private input replay tasks `replay-historical-phase1`, `replay-historical-phase2` and
`replay-internal-split` use the checksum-verified manifests described in
`data_prep/REPLAY.md`. Configure `PRIVATE_INPUTS` and, for the internal tree,
`CORPUS_SOURCE_ROOT`. These tasks do not run a new detector.

For prepared folders with file symlinks into a separate raw download, also configure
`REPLAY_P1_RAW_ROOT` or `REPLAY_P2_RAW_ROOT` to that permitted physical image root.
Their defaults are the respective prepared-field roots; no other location is implicitly
authorized. Logical class folders and field identifiers remain unchanged.

Native nine-class internal tasks (`internal-native-*`) call `eval/evaluate_internal.py` with trainer-specific
deterministic preprocessing; they must not be replaced by the seven-class hflip task
`proposed-internal-mapped`. The native evaluator requires a new output directory.

Task IDs are literal: pass comma-separated full IDs rather than the `*` notation above.
The registry order is not a dependency scheduler. Selected tasks execute in the order
given to `--tasks`; prerequisites are checked immediately before each task, so outputs
of an earlier selected task can satisfy a later one. `--tasks all --check --keep-going`
is a useful inventory. `--tasks all --run` is not a one-command scientific replication:
manual steps intentionally block it, full training is expensive, and numerical targets
still require separate comparison.

Neural-network training is never authorized by preflight. To run a selected training
task, add `--allow-training`, for example:

```bash
python -B reproduce/run_study.py --run --tasks train-proposed \
  --config /path/to/local_reproduction.json --allow-training \
  --timeout-seconds 172800
```

The `fit` task `frozen-probe-refit` fits a new logistic-regression probe and may extract
features for the entire corpus. It is not the original retained probe. Network training
and refitting are not required to test already retained weights, and should not be
performed merely to turn unavailable-input statuses into green checks.

## Reports and truthful status labels

Each check/run writes a unique private directory beneath
`OUT_ROOT/reproduction_runs/` with `report.json` and per-task command logs. It records
the task registry/runner hashes, actual interpreter, command exit codes, wall time,
output paths, freshness and output hashes. Never substitute this report for the
original run logs.

- `READY`: paths/interpreter satisfy preflight; no experiment has run.
- `EVIDENCE_CHECK_EXECUTED`: an evidence checker ran; its own log determines coverage.
- `EXECUTED_OUTPUTS_PRESENT`: command exited zero and all declared outputs were newly
  written. This is not an assertion that they match the paper.
- `MISSING_INPUTS`, `CONFIGURATION_ERROR`, `COMMAND_FAILED`, `TIMED_OUT`,
  `MISSING_OUTPUTS`, `UNCHANGED_OR_UNASSERTED_OUTPUTS`, `TRAINING_NOT_AUTHORIZED` and
  `MANUAL` are not passes. The overall exit remains nonzero, even with `--keep-going`.

The runner does not clip predictions, modify historical reference values or hide
rounding differences. `numerical_agreement` remains `NOT_ASSESSED` until a separate
numerical comparison is made. `complete_study_reproduced` is never inferred from
process exits or the existence of files. The hematologist review, the Fig. 1 illustration,
license-restricted source material and any unimplemented/manual steps remain visible.

Every task has a timeout; timed-out command process groups are terminated, including
their worker processes. An interrupted/failed task does not erase earlier logs or
successful outputs. Re-running training runners may reuse an already-completed training
folder; unchanged output files are not counted as fresh training by this wrapper.

Offline tests:

```bash
python -B -m unittest discover -s tests -p test_reproduction.py -v
```
