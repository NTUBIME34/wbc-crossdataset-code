# stats/

CPU-only scripts that turn the per-image outputs of `eval/` and `pipeline/` into the
statistics reported in the paper. Every script takes its inputs as command-line paths,
prints the statistic with the paper's rounding, and writes a machine-readable copy to
`--out-dir` (default `$OUT_ROOT/stats`; `OUT_ROOT` defaults to `./outputs` and must lie
outside the repository). `python stats/<script>.py --help` describes the inputs in full.

| Script | Input | Output | Paper |
|---|---|---|---|
| `seed_summary.py` | `--run NAME` + one `predictions.csv` per seed (evaluator, `--use-mappings`; or pipeline output) | seed mean ± population SD of macro-F1, weighted F1, accuracy, balanced accuracy; per-class F1 (`--per-class` adds recall); `seed_summary.csv` | Table 3 external columns, per-seed values in S7, S10 |
| `coverage_aware_score.py` | `--predictions` (one per seed: evaluator output on the crops folder, or `pipeline/classify.py` output) + `--detections` (`detections.csv`) | detected-only and coverage-aware metrics per seed and mean ± SD, coverage counts; `--compare-to` counts changed predictions; `coverage_aware_<name>.json` | Table 2b, Table 2a, S6, S1 field counts, S17 changed predictions |
| `bootstrap_ensemble_ci.py` | `--model NAME` + one `predictions.csv` with `prob_<class>` per seed | probability-ensemble macro-F1 and 95% CI (rows sorted by file name, `default_rng(0)` per model, 2,000 × `integers(0, n, n)`, percentile); `external_macrof1_bootstrap_ci.csv` | S9; Results 3.3 |
| `paired_tests.py` | `--model NAME` + one file per seed (evaluator `predictions.csv` or `--dump-preds` path,true,pred; seed 42 first), `--compare A B` | McNemar exact p, Δ, paired-bootstrap 95% and 90% CI (5,000 replicates), TOST on the 90% CI, tie-rule sensitivity, probability-ensemble Δ, seed-level paired t-test; `paired_tests.{json,txt}` | S7 table, footnote and interpretation; Results 3.4; Methods (64 disagreements) |
| `recipe_ensembles.py` | `--group NAME` + one `predictions.csv` with probabilities per seed, in the paper's order (color aug, class-weighted CE, mixup + BPaCo + cRT) | ensemble accuracy, balanced accuracy, macro-F1, minimum recall, BLAST/PROMY F1; per-group and paired 1,000-replicate bootstraps; `recipe_ensembles/*.csv` | S8; Results 3.6 recipe-selection bootstraps |
| `threshold_analysis.py` | `--internal` and `--external` `predictions.csv` with probabilities, one per seed | internal-tuned and oracle one-vs-rest thresholds for Blast group and Promyelocyte; `threshold_validation_locked.csv` | S11; Results 3.6 |
| `calibration.py` | `--predictions`, one external `predictions.csv` with probabilities per seed | ECE (15 bins), multiclass Brier score, mean confidence vs accuracy, top-bin share, reliability and per-class tables; `calibration_ensemble.json`, `calibration_bins.csv` | Results 3.6; Supplementary Fig. S2 |
| `lodo_present_class.py` | `--fold NAME` + one nine-class `predictions.csv` per seed (`eval/evaluate_lodo.py`) or `analysis/evidence/lodo*/heldout_*.csv.gz` | present-class macro-F1 mean ± SD, accuracy, present-class balanced accuracy, per-class n/recall/F1, fixed nine-class and union macro-F1, error routing of collapsed classes; `lodo_present_class.json`, `lodo_per_class.csv` | Table 4, S14, Supplementary Note 1, S16 proposed column |
| `detector_overlap_audit.py` | `--txl-root` (TXL-PBC with `metadata_file.xlsx` and `images/<split>`) + `--heldout SOURCE files` | TXL-PBC composition and splits; exact file-name overlap of the detector's training split with each held-out set; `detector_overlap_audit.json` | S1 (detector/classifier source overlap) |
| `internal_seven_class.py` | `--model NAME` + one internal 9×9 confusion matrix per seed (trainer `confusion_matrix.csv` / `test_results.json`, `analysis/evidence/` matrices, or nine-class `predictions.csv`) | internal nine-class macro-F1 and its collapse to the seven evaluation classes; `internal_seven_class.json` | S17 internal seven-class sensitivity; Table 3 internal column |

Conventions shared by all scripts:

- Seven evaluation classes; macro-F1 is the unweighted mean of the class F1 scores
  (`labels=` the seven classes, F1 = 0 for a class with no prediction and no support).
  Seed aggregates are the mean and the population SD (ddof=0).
- Files of different seeds are joined by image id: the file name without extension and
  without the detector's `__detNN` crop suffix.
- Probability ensemble: mean of the `prob_<class>` columns over seeds, then argmax.
  Majority vote: the most frequent seed label; a 1-1-1 tie takes the first file's label,
  so pass the seed-42 file first.
- A bootstrap depends on the image order as well as the RNG seed. Each script fixes its
  order and states it in `--help`: file name (`bootstrap_ensemble_ci.py`), class folder
  plus file name (`paired_tests.py`), the row order of the first file
  (`recipe_ensembles.py --order file`, the archived run; `--order image_id` sorts).
- Per-image outputs on WBCBench2026 (predictions, detections, the files these scripts
  read) must stay outside the repository; only aggregate tables may be shared.

Example (Supplementary Table S9):

```bash
OUT_ROOT=/path/outside/repo python stats/bootstrap_ensemble_ci.py \
  --model "Swin (proposed)" swin/seed_42/predictions.csv swin/seed_62/predictions.csv swin/seed_82/predictions.csv \
  --model VGG16-BN vgg16_bn/seed_42/predictions.csv vgg16_bn/seed_43/predictions.csv vgg16_bn/seed_44/predictions.csv
```
