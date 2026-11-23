# Inference with the retained frozen DinoBloom probe

`evaluate_frozen_probe.py` evaluates the already-fitted probe without extracting
training features, fitting a new scaler/classifier, searching C, or selecting a model.
It uses the original pretrained DinoBloom backbone, not a fine-tuned classifier.

The released weights record holds only the 13 core checkpoints; it does not include the
files below. They belong to the authors' extended model bundle, whose layout the paths
follow:

- `model_root/frozen_probe/probe.joblib`: fitted scaler, logistic regression, ordered
  classes and the retained unweighted C=0.01 configuration;
- `model_root/assets/dinobloom_config/config.json` and `model.safetensors`: the pinned
  original backbone and configuration, with separate Apache-2.0 attribution;
- `model_root/ADDITIONAL_ASSETS.json` and `FOUNDATION_ASSETS.json`: their SHA-256 values.

Without these files, `frozen_probe.py` refits the probe, with its whole configuration
grid, from the pinned public backbone.

After checking the bundle's SHA list, pass those exact hashes explicitly:

```bash
python -B eval/evaluate_frozen_probe.py \
  --probe /data/weights/model_root/frozen_probe/probe.joblib \
  --probe-sha256 PROBE_SHA_FROM_ADDITIONAL_ASSETS \
  --trust-probe \
  --model-name local-dir:/data/weights/model_root/assets/dinobloom_config \
  --backbone-sha256 BACKBONE_SHA_FROM_FOUNDATION_ASSETS \
  --config-sha256 CONFIG_SHA_FROM_FOUNDATION_ASSETS \
  --test-root /data/prepared/phase1_crops \
  --mode external --device cuda --output-dir /data/outputs/retained_probe_external
```

Use `--mode internal` and the exact historical internal test split for the native
nine-class test score. Internal inference uses the center view only. External inference
classifies center and horizontal-flip embeddings separately, averages their nine-class
probabilities, then maps the nine-class argmax to the seven benchmark classes, exactly
as `frozen_probe.py`. It does not average embeddings before classification or replace
the nine-class decision with a seven-class argmax.

The original backbone is constructed with `pretrained=True`, `num_classes=0` and
`img_size=224`. The local files avoid a network lookup. The original pinned config's
advertised default resolution does not override this explicit evaluation resolution.
The fitted probe is a Python joblib object: load only an author artifact that you trust
and whose SHA you checked. `--trust-probe` is required; hashes are checked before loading.

Outputs are `predictions.csv`, `metrics.json` and `provenance.json` in a new directory
outside the code repository. No embeddings or training data are written. A successful
run is an inference result, not automatically a match to the paper; compare the metrics
and confusion counts separately, using the same original input manifest.
