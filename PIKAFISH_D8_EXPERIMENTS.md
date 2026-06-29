# Pikafish D8 Supervised Structure Experiments

Date: 2026-06-29

## Dataset

Generated a fixed Pikafish depth-8 label dataset for repeatable supervised evaluation.

- SQLite: `eval/pikafish-random-1m-d8-20260629.sqlite`
- FEN list: `eval/random.fens`
- CSV labels: `eval/pikafish-labels.csv`
- Rows: 1,000,000
- Split:
  - train: 900,000
  - valid: 50,000
  - test: 50,000

The final structure experiments used a fixed quick-eval subset:

- train: first 200,000 rows from `split='train'`
- valid: first 50,000 rows from `split='valid'`
- epoch: 1

## Tooling Added

### Concurrent label generation

`pikafish-label-random` now supports:

- `--workers`
- `--commit-every`
- resumable labeling by skipping already labeled FENs at the same depth
- stable `split` column in SQLite

### Fixed-label supervised training

Command:

```powershell
cargo run --profile fast --features gpu-train -- az-train-pikafish-labels <model> eval\pikafish-random-1m-d8-20260629.sqlite `
  --train-limit 200000 `
  --valid-limit 50000 `
  --epochs 1 `
  --batch-size-per-gpu 1024 `
  --lr 0.0006 `
  --value-weight 0.5 `
  --policy-weight 1.0 `
  --output <output.safetensors>
```

### PyTorch prototype

`tools/torch_pika_experiment.py` was used for quick structure checks after exporting binary arrays with:

```powershell
cargo run --profile fast -- az-export-pikafish-labels eval\pikafish-random-1m-d8-20260629.sqlite eval\torch\pika_train200k --split train --limit 200000
cargo run --profile fast -- az-export-pikafish-labels eval\pikafish-random-1m-d8-20260629.sqlite eval\torch\pika_valid50k --split valid --limit 50000
```

The exported `eval/torch` cache is disposable and was removed during cleanup.

## Main Findings

### Training setup mattered more than shared blocks

Original no-block h320 baseline used:

- batch: 4096
- lr: 0.0003
- value_weight: 1.0

It reached:

- valid policy CE: 3.0013
- valid value MSE: 0.6809

Fair no-block h320 with revised training used:

- batch: 1024
- lr: 0.0006
- value_weight: 0.1

It reached:

- valid policy CE: 2.0940
- valid value MSE: 0.6064

Conclusion: the large policy improvement came mostly from training objective/optimizer settings, not from shared trunk blocks.

### Shared residual blocks had small policy impact

Fixed setup:

- batch: 1024
- lr: 0.0006
- value_weight: 0.1

| hidden | no blocks CE | 1 block CE | 2 blocks CE | 3 blocks CE |
|---:|---:|---:|---:|---:|
| 128 | 2.1712 | 2.1381 | 2.1160 | 2.1310 |
| 192 | 2.1195 | 2.0910 | 2.0829 | 2.0850 |
| 320 | 2.0940 | 2.0606 | 2.0603 | 2.0595 |

Conclusion: shared blocks help a little, but the effect is small. More than 2 blocks is not worthwhile here.

### Dual tower solved value/policy conflict better

Final architecture merged into `master`:

```text
shared:
  sparse embedding sum
  structural embedding sum
  sparse attention residual
  ReLU
  RMSNorm

policy tower:
  2 residual blocks
  inner = 2 * hidden
  SiLU
  RMSNorm
  original full policy head

value tower:
  1 residual block
  inner = 2 * hidden
  SiLU
  RMSNorm
  value head + moves-left head
```

Dual tower did not massively reduce policy CE, but it allowed higher value weight without damaging policy much.

| hidden | value_weight | valid policy CE | valid value MSE | sims/sec |
|---:|---:|---:|---:|---:|
| 128 | 0.5 | 2.1266 | 0.3353 | 38.4k |
| 128 | 1.0 | 2.1362 | 0.3168 | not measured |
| 192 | 0.5 | 2.0818 | 0.3263 | 25.9k |
| 192 | 1.0 | 2.0871 | 0.3165 | 26.0k |
| 320 | 0.5 | 2.0527 | 0.3267 | 12.9k |
| 320 | 1.0 | 2.0630 | 0.3179 | 11.9k |

Conclusion:

- `value_weight=0.5` is the better default for policy/value balance.
- `value_weight=1.0` slightly hurts policy but improves value.
- h192 dual is the best current speed/quality tradeoff.
- h320 dual is stronger but much slower.

## Recommended Default

Use:

```text
hidden = 192
policy tower blocks = 2
value tower blocks = 1
batch_size_per_gpu = 1024
lr = 0.0006
policy_weight = 1.0
value_weight = 0.5
```

Representative model kept:

```text
eval/model-pikafish-d8-rust-dual-h192-v05-b1024-200k-e1.safetensors
```

## Files Kept After Cleanup

Kept long-term data:

- `eval/pikafish-random-1m-d8-20260629.sqlite`
- `eval/random.fens`
- `eval/pikafish-labels.csv`
- `eval/pikafish-random-5000-d8.sqlite`

Kept representative trained models:

- `eval/model-pikafish-d8-rust-dual-h128-v05-b1024-200k-e1.safetensors`
- `eval/model-pikafish-d8-rust-dual-h128-v1-b1024-200k-e1.safetensors`
- `eval/model-pikafish-d8-rust-dual-h192-v05-b1024-200k-e1.safetensors`
- `eval/model-pikafish-d8-rust-dual-h192-v1-b1024-200k-e1.safetensors`
- `eval/model-pikafish-d8-rust-dual-h320-v05-b1024-200k-e1.safetensors`
- `eval/model-pikafish-d8-rust-dual-h320-v1-b1024-200k-e1.safetensors`

Removed disposable files:

- random initialization `.safetensors`
- failed/intermediate shared-trunk models
- policy-wide experiment models
- PyTorch exported binary cache under `eval/torch`

## Next Steps

1. Train h192 dual on the full 900k train split for 1 epoch.
2. Evaluate on fixed 50k valid and 50k test.
3. Compare h192 dual against h320 dual using actual search/play, not only CE.
4. Consider better policy labels:
   - MultiPV soft labels
   - PV top-k labels
   - label smoothing
5. Consider value target tuning:
   - `cp_scale = 300, 450, 600, 800`
   - `value_weight = 0.5` vs `1.0`
