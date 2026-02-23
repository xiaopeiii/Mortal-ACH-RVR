# ACH+RVR Three-Stage Playbook (Mortal)

## 1) Goal

Run ACH+RVR with this fixed pipeline:

1. Offline Teacher (privileged)
2. Offline Student Distill (private)
3. Online Strict ACH+RVR

Then evaluate strength against old value-based / old policy-based checkpoints with reproducible commands.

This document does not change algorithm/loss. It is only runbook + evaluation protocol.

## 2) One-Time Prep

```powershell
cd E:/Mahjong\Mortal
$env:PYO3_PYTHON="C:/Users/34491/anaconda3/envs/mahjong-review/python.exe"
cargo build -p libriichi --release
Copy-Item E:/Mahjong\Mortal\target\release\riichi.dll E:/Mahjong\Mortal\mortal\libriichi.pyd -Force
```

## 3) Stage-0 Gate (Strict Alignment Audit)

Run this before any online strict training.

```powershell
$env:MORTAL_CFG="E:/Mahjong/Mortal/mortal/config.mode16.konten.3060.ach_rvr.strict.toml"
cd E:/Mahjong\Mortal\mortal
python -u audit_alignment_strict.py --limit 100
python -u audit_alignment_strict.py
```

Pass criteria:

1. `mapping_coverage == 1.0`
2. `missing_old_logp == 0`
3. `ambiguous_mapping == 0`
4. `missing_trace == 0`
5. `missing_kan_fields == 0`
6. `action_mismatch == 0`

## 4) Stage-1 Offline Teacher

Config:
`Mortal/mortal/config.mode16.konten.3060.ach_rvr.offline.teacher.toml`

```powershell
$env:MORTAL_CFG="E:/Mahjong/Mortal/mortal/config.mode16.konten.3060.ach_rvr.offline.teacher.toml"
cd E:/Mahjong\Mortal\mortal
python -u train_ach_rvr_offline_teacher.py
```

Checkpoint output (default):
`output/checkpoints/mortal_mode16_konten_3060_ach_rvr_offline_teacher.pth`

## 5) Stage-2 Offline Student Distill

Config:
`Mortal/mortal/config.mode16.konten.3060.ach_rvr.offline.student.toml`

Ensure this field points to Stage-1 output:
`distill.teacher_state_file`

```powershell
$env:MORTAL_CFG="E:/Mahjong/Mortal/mortal/config.mode16.konten.3060.ach_rvr.offline.student.toml"
cd E:/Mahjong\Mortal\mortal
python -u train_ach_rvr_offline_student.py
```

Checkpoint output (default):
`output/checkpoints/mortal_mode16_konten_3060_ach_rvr_offline_student.pth`

## 6) Stage-3 Online Strict ACH+RVR

Config:
`Mortal/mortal/config.mode16.konten.3060.ach_rvr.strict.toml`

Recommended bootstrap:
set `control.bootstrap_state_file` to Stage-2 student checkpoint.

Open 3 terminals (same `MORTAL_CFG`):

Terminal A:

```powershell
$env:MORTAL_CFG="E:/Mahjong/Mortal/mortal/config.mode16.konten.3060.ach_rvr.strict.toml"
cd E:/Mahjong\Mortal\mortal
python -u server.py
```

Terminal B (1~N clients):

```powershell
$env:MORTAL_CFG="E:/Mahjong/Mortal/mortal/config.mode16.konten.3060.ach_rvr.strict.toml"
cd E:/Mahjong\Mortal\mortal
python -u client.py
```

Terminal C (trainer):

```powershell
$env:MORTAL_CFG="E:/Mahjong/Mortal/mortal/config.mode16.konten.3060.ach_rvr.strict.toml"
cd E:/Mahjong\Mortal\mortal
python -u train_ach_rvr_online.py
```

Checkpoint output (default):
`output/checkpoints/mortal_mode16_konten_3060_ach_rvr_strict.pth`

## 7) Evaluation Protocol (Comparable and Fair)

## 7.1 Rules

1. Fix seed settings for all comparisons (`seed_start`, `seed_key`).
2. Use same `--games`, same `--pts`, same device.
3. Prefer `2v2` as primary metric (both models appear equally often).
4. Use `1v3` as secondary metric (script already warns role imbalance).
5. Keep `--enable-compile` off unless Triton is installed and stable.

## 7.2 Compare ACH+RVR policy vs old value-based model

```powershell
cd E:/Mahjong\Mortal\mortal
python -u eval_match.py `
  --mode 2v2 `
  --model-a E:/Mahjong\output\checkpoints\mortal_mode16_konten_3060_ach_rvr_offline_teacher_step5500.pth `
  --model-b E:/Mahjong/output/checkpoints/mortal_mode16_konten_3060_step185000_b4.pth `
  --name-a ach_rvr_policy `
  --name-b old_value `
  --decision-head-a policy `
  --decision-head-b value `
  --challenger a `
  --games 200 `
  --seed-start 10000 `
  --seed-key 20260218 `
  --device cuda:0 `
  --log-dir E:/Mahjong/output/eval/achrvr_vs_oldvalue_2v2
```

## 7.3 Compare ACH+RVR policy vs old policy-based model

Use only if model-b checkpoint contains `policy` state.

```powershell
python -u eval_match.py `
  --mode 2v2 `
  --model-a E:/Mahjong/output/checkpoints/mortal_mode16_konten_3060_ach_rvr_strict.pth `
  --model-b E:/Mahjong/output/checkpoints/<old_policy_checkpoint>.pth `
  --name-a ach_rvr_policy `
  --name-b old_policy `
  --decision-head-a policy `
  --decision-head-b policy `
  --policy-stochastic-a `
  --policy-stochastic-b `
  --challenger a `
  --games 20000 `
  --seed-start 10000 `
  --seed-key 20260218 `
  --device cuda:0 `
  --enable-amp `
  --log-dir E:/Mahjong/output/eval/achrvr_vs_oldpolicy_2v2
```

## 7.4 Compare internal head quality (same checkpoint)

If checkpoint includes both `policy` and `current_dqn`:

```powershell
python -u eval_match.py `
  --mode 2v2 `
  --model-a E:/Mahjong/output/checkpoints/mortal_mode16_konten_3060_ach_rvr_strict.pth `
  --model-b E:/Mahjong/output/checkpoints/mortal_mode16_konten_3060_ach_rvr_strict.pth `
  --name-a ach_policy `
  --name-b ach_value `
  --decision-head-a policy `
  --decision-head-b value `
  --policy-stochastic-a `
  --challenger a `
  --games 20000 `
  --seed-start 10000 `
  --seed-key 20260218 `
  --device cuda:0 `
  --enable-amp `
  --log-dir E:/Mahjong/output/eval/ach_policy_vs_ach_value_2v2
```

## 7.5 Stage-to-stage ablation

Recommended fixed-seed matrix:

1. Teacher vs OldValue
2. Student vs Teacher
3. OnlineStrict vs Student
4. OnlineStrict vs OldValue
5. OnlineStrict vs OldPolicy

Use same template command as above; only replace checkpoint paths/names.

## 8) Quick health checks during training

1. Teacher TB: `loss/actor_loss`, `loss/value_loss`, `rv/loss`, `ern/loss`.
2. Student TB: `distill/kl`, `distill/value_mse`, `distill/bc_anchor`.
3. Online strict TB:
  - `strict_ach/old_logp_coverage` should stay near 1.0
  - `aligned/four_seat_coverage` stable
  - `strict_ach/c_gate_rate_pos` and `strict_ach/c_gate_rate_neg` non-degenerate

## 9) Recommended result table (for repeatability)

Store each run with columns:

1. run_id
2. stage
3. model_a / head_a
4. model_b / head_b
5. mode
6. games
7. seed_start
8. seed_key
9. avg_pt_a
10. avg_rank_a
11. avg_pt_b
12. avg_rank_b
13. notes

## 10) Important compatibility notes

1. Offline two-stage (`teacher`, `student`) requires `strict.enabled=false`.
2. Online strict requires new trace fields and strict alignment pass.
3. Old logs without strict fields are for offline/non-strict only.


