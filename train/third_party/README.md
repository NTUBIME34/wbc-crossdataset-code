# Third-party code fetched at setup time

## BPaCo loss (`bpaco_losses.py`, not included)

The BPaCo comparison arms use `BPaCoLoss` from `losses.py` of the BPaCo authors'
repository (balanced parametric contrastive learning, MICCAI 2024):

| | |
|---|---|
| Repository | https://github.com/Davidczy/BPaCo |
| Commit | `cbe577570a0f23e6a49191c531ac27e4523df7da` |
| Path | `losses.py` (repository root) |
| git blob id | `8a802682d6d634ae9e791ad6d32c1ae2f79fa7ca` |
| SHA-256 | `d37005638fecd5892a7bb9c9706398faab775581b2f5e236123acb94853a1559` (23,835 bytes) |

That repository has no licence, so the file is not redistributed here. Fetch and verify
it once before running a BPaCo arm:

```bash
bash train/third_party/fetch_bpaco_loss.sh            # downloads to train/third_party/bpaco_losses.py
bash train/third_party/fetch_bpaco_loss.sh --check    # re-verify later
```

Without network access, verify and install a copy from your own clone of the
repository at that commit with `--from-file /path/to/BPaCo/losses.py`.
The script refuses any file whose git blob id, SHA-256 or size differs.
`bpaco_losses.py` must stay out of version control.

Who needs it: only runs with `--bpaco-weight > 0`, that is
`train/scripts/run_swin_bpaco_adamw.sh`, `train/scripts/run_resnet50_bpaco_adamw.sh`,
`train/scripts/run_resnet50_bpaco_sgd.sh`, `train/scripts/run_swin_mixup_bpaco_crt.sh` and
the BPaCo variant of `train/scripts/run_lodo_dinobloom.sh`. The proposed model and all
class-weighted cross-entropy runs never import it. The trainers load the file lazily
and stop with an instruction to run the fetch script if it is missing.

Note: the upstream `BPaCoLoss.cal_weight_for_classes` moves its class weights to
`torch.device('cuda')`, so the BPaCo arms need a CUDA GPU.

## Other third-party code in `train/`

`train/randaugment.py` is redistributed with its notices (timm, Apache-2.0; PyContrast,
BSD-2-Clause; PaCo, MIT); see the header of that file and `THIRD_PARTY_NOTICES.md`.
