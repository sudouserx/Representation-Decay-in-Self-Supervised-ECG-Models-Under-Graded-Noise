# Representation Decay in Self-Supervised ECG Models Under Graded Noise

A reproducible, patient-safe benchmark of self-supervised ECG encoders under graded, deterministic corruption: paired **task**, **calibration**, and **representation-shift** analysis with uncertainty. The composite robustness index is a pre-registered **secondary** reporting convenience.

**Dataset**: PTB-XL 1.0.3 (21,799 records / 18,869 patients before canonical-label exclusion; 12-lead, 500 Hz, 10 s)
**Target**: 5 diagnostic superclasses — NORM · MI · STTC · CD · HYP
**Execution environment**: Kaggle T4 GPU notebooks (16 GB VRAM, &lt;12 h). Deployment numbers are **server-proxy**, not edge hardware.

This is an engineering / methodological benchmark on a single-center German cohort. It is **not** a clinical validation study.

---

## Repository Structure

```
ECG_SSL_Haqila_Lab/
├── ecg_ssl_utils/               # Reusable utility package
│   ├── config.py                # Central dataclasses
│   ├── repro.py                 # Global seeds + deterministic DataLoaders
│   ├── artifact.py              # Config / seed / git-hash snapshots
│   ├── version.py               # Package version written into artifacts
│   ├── data/                    # Load, sosfiltfilt 0.05–45 Hz, labels
│   ├── noise/                   # NSTDB (hard-fail) + synthetic generators
│   ├── models/                  # ViT-S 1D, ResNet-18 1D, projectors
│   ├── ssl/                     # SimCLR, CLOCS-adapted, MAE, I-JEPA-adapted, BYOL, SwAV
│   ├── probe/                   # Linear probe + temperature scaling
│   ├── eval/                    # AUROC, PR-AUC, F1, ECE, Brier, CKA, ER, bootstrap, DeLong
│   ├── score/                   # Robustness index, gates (filters), Kendall-τ stability
│   ├── deploy/                  # ONNX encoder+probe, INT8, server-proxy profiler
│   └── report/                  # Leaderboard, curves, risk reports, guidelines
├── scripts/                     # Pipeline 00–07 (each script is a Kaggle session)
├── tests/
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

| # | Script | Phase | Notes |
|---|--------|-------|-------|
| 00 | `00_data_preparation.py` | Data | Raw dumps, 0.05–45 Hz `sosfiltfilt`, train-only z-score, canonical labels |
| 01 | `01_noise_injection.py` | Noise | NSTDB hard-fail + PSD plots + 162-condition manifest |
| 02a | `02a_pretrain_simclr.py` | SSL | SimCLR × `--seed` |
| 02b | `02b_pretrain_clocs.py` | SSL | CLOCS-adapted × `--seed` |
| 02c | `02c_pretrain_mae.py` | SSL | MAE × `--seed` |
| 02d | `02d_pretrain_jepa.py` | SSL | I-JEPA-adapted × `--seed` |
| 02e1 | `02e1_pretrain_byol.py` | SSL | BYOL sensitivity arm (1 seed, flagged) |
| 02e2 | `02e2_pretrain_swav.py` | SSL | SwAV 2-view sensitivity arm (1 seed, flagged) |
| 02e3 | `02e3_pretrain_clocs_resnet.py` | SSL | CLOCS-ResNet18 sensitivity arm (1 seed, flagged) |
| 02f | `02f_supervised_baseline.py` | SSL | End-to-end ViT-S, same splits |
| 03 | `03_linear_probe.py` | Probe | Frozen CLS → linear head; T applied **once**; val-tuned F1 thresholds |
| 04 | `04_corruption_eval.py` | Eval | Inject on **raw mV**, then identical front-end; SHA-256 immutability |
| 05 | `05_decay_metrics.py` | Stats | Clustered bootstrap + Stouffer + BH; no p-value averaging |
| 06 | `06_deploy_profile.py` | Deploy | Train-calibrated INT8, parity numbers, RSS memory, real ECG input |
| 07 | `07_dss_and_report.py` | Reports | Secondary robustness index + Kendall-τ stability + parity/AUROC/ECE filters |

---

## Methodology (what the code actually does)

### I. Data, labels, preprocessing, noise

- **Filter**: 4th-order Butterworth **0.05–45 Hz** via `sosfiltfilt` (diagnostic-mode; preserves ST content for STTC).
- **Normalization**: per-lead z-score from **train only**.
- **Labels**: canonical PTB-XL superclasses with likelihood **threshold 50**; records with an all-zero superclass vector are **excluded**. Threshold-0 variants are also saved (`superclass_labels_raw_*`) for the dataset table.
- **Split**: folds 1–8 / 9 / 10, patient-disjointness asserted. `--strict-dataset` asserts 21,799 records and 18,869 patients **before** exclusion.
- **00 also writes** unfiltered `signals_{split}_raw.npy`.
- **Noise types**: NSTDB BW/MA/EM (`resample_poly` 360→500; **hard-fail** if files missing; 2-channel → 12-lead replication is a documented approximation) plus physically motivated **synthetic** powerline (50 Hz + harmonics, \(A_h=1/h\)), electrode pop, inverter. Not “clinical realism.”
- **Injection (Option A)**: script 04 injects on raw mV, then applies the **same** band-pass + saved train stats. SNR is therefore physiological. A 45 Hz front-end will attenuate 50 Hz mains — that is a finding.
- **Grid**: 6 types × 6 SNR (−6…24 dB) × 3 noise seeds + 3 mixtures × 6 × 3 = **162** conditions. Deterministic given `(record_id, type, SNR, seed)`.

### II. Pretraining

Shared ViT-S 1D backbone (12L / 384d / 6H / patch 50 / ~22M) except 02e3 (ResNet-18 1D). **Epoch parity** with a disclosed per-method budget — not compute parity.

| Encoder | Script | Seeds | Notes |
|---------|--------|-------|-------|
| SimCLR | 02a | 42, 123, 456 | NT-Xent, batch 256 × accum 4 = 1024 |
| CLOCS-adapted | 02b | 42, 123, 456 | τ=0.5, interpolated halves, Gaussian lead fill; batch 128 × accum 8 |
| MAE | 02c | 42, 123, 456 | 75% mask; full decoder in checkpoint |
| I-JEPA-adapted | 02d | 42, 123, 456 | Target patches+CLS, L2; collapse warning; full EMA/predictor ckpt |
| BYOL | 02e1 | 1 seed | Sensitivity arm |
| SwAV (2-view) | 02e2 | 1 seed | Sinkhorn `Q *= B`; sensitivity arm |
| CLOCS-ResNet18 | 02e3 | 1 seed | Sensitivity arm |
| Supervised ViT-S | 02f | 42 | End-to-end baseline on the same splits |

`--seed` / `PRETRAIN_SEED` is required. Outputs: `ssl-<paradigm>-seed<seed>/`.

Collapse diagnostics (CLS embed std &lt; 1e-4 or mean cosine &gt; 0.95):

- **SimCLR / CLOCS-ViT**: halt after 5 consecutive bad checks.
- **BYOL / SwAV / CLOCS-ResNet**: logged every epoch; no automatic halt.
- **I-JEPA**: warning printed; training continues.
- **MAE**: reconstruction loss only; no CLS-std / cosine halt.

Residual AMP/cuDNN nondeterminism is possible even with `cudnn.deterministic=True`.

### III. Frozen evaluation

- Linear probe on frozen CLS features; early-stop on val macro-AUROC.
- Temperature fitted on **raw** val logits; applied **exactly once** to clean and noisy paths (`predict_proba`).
- Per-class F1 thresholds swept on validation (0.05…0.95) and reused for clean and noisy F1.
- Representations stored fp32. SHA-256 hash of encoder and probe before/after each condition.

### IV. Statistics

Three variance sources: **patient** (sampling), **noise seed** (corruption), **pretraining seed** (training).

- Point estimates: full-sample metrics. CIs: patient-clustered percentile bootstrap (1000) on **noise-seed-averaged probabilities** — CI bounds are never averaged.
- Primary test: paired patient-bootstrap ΔAUROC; Stouffer across 3 noise seeds; BH-FDR over the rows in `hypothesis_tests.parquet` (encoder × noise-type × SNR × class). `bh_family.json` records the size.
- DeLong is appendix-only (independence assumption is violated by multi-record patients).
- Primary paradigms: mean ± SD across pretraining seeds. Sensitivity arms are flagged and must not be ranked against primaries without that caveat.

Metrics: macro/per-class AUROC, PR-AUC, F1 (tuned thresholds), OvR ECE (15 equal-width primary + equal-mass check), Brier, linear CKA (variance guard), effective rank of the CLS matrix `(n_samples × embed_dim)` (two-sided ΔER in the index).

### V. Robustness index (secondary)

\[
R = 0.30(1-\Delta\mathrm{AUROC}_n) + 0.30(1-\Delta\mathrm{ECE}_n) + 0.20(1-\Delta\mathrm{CKA}_n) + 0.20(1-\Delta\mathrm{ER}_n)
\]

- \(\Delta\mathrm{CKA}=1-\mathrm{CKA}\); \(\Delta\mathrm{ER}=|1-\mathrm{ER}_n/\mathrm{ER}_c|\) (two-sided); \(\Delta\mathrm{AUROC}=\max(0,\mathrm{AUC}_c-\mathrm{AUC}_n)\); \(\Delta\mathrm{ECE}=\max(0,\mathrm{ECE}_n-\mathrm{ECE}_c)\).
- Deltas min-max normalized **within the current run**; anchors printed to `normalization_anchors.json`. Do not compare \(R\) across papers.
- **Gates are filters only** (AUROC &lt; 0.70 or ECE &gt; 0.15 → REJECT in decision support). They do **not** zero \(R\).
- Weight stability: Kendall τ vs the pre-registered ranking over 1024 Dirichlet draws (`weight_stability.json`).
- Primary evidence is the raw 4-dimension table (`raw_dimensions.parquet`).

### VI. Server-proxy deployment

ONNX opset 17, **encoder + probe**. INT8 dynamic / static; static calibration uses **train** signals. Parity cosine(FP32, INT8) is measured on 500 val samples in script 06; script 07 REJECT if cosine &lt; `cfg.deploy.parity_min_cosine` (default 0.999). Profiler: real ECG input, warmup 50 / 1000 runs, p50/p95, throughput, **process RSS delta** (psutil), platform / CPU / GPU / ORT version. Energy is **not measured** and is not reported. There is no “selective attention-FP32 / FFN-INT8” mode.

---

## Kaggle execution

```
00 → 01
02a–d × seeds {42,123,456}   (12 sessions)
02e1, 02e2, 02e3, 02f        (1 seed each)
03 → 04 → 05 → 06 → 07
```

Upload `ecg_ssl_utils/` as a dataset. In each notebook:

```python
import sys
sys.path.insert(0, '/kaggle/input/ecg-ssl-utils/')
from ecg_ssl_utils.config import get_config
```

Purge stale 05/07 artifacts before citing any number. Existing unseeded 02 checkpoints are not valid for paper tables.

---

## Reproducibility

- `requirements.txt` lists runtime packages. After a paper run, save `pip freeze` from that image as `environment.lock.txt`.
- Every script writes `config.json` and/or `run_snapshot.json` (full dataclasses, seed, package version, git hash when available).
- Data version: PTB-XL **1.0.3**.
- Tests: `pytest tests/` (CKA/ER/ECE/bootstrap, SNR round-trip, Sinkhorn, temperature-once, seeded SimCLR first-batch loss).

---

## Known limitations

- Contrastive arms share one 6-augmentation policy; MAE/JEPA use masking / block prediction — paradigm vs augmentation effects cannot be fully separated.
- CKA/ER are CLS-token, last-layer **shift / dimensionality** diagnostics, not proof that task-relevant information was lost. Tie “decay” language to ΔAUROC / ΔECE / ΔF1.
- Score normalization is within-run only.
- DeLong ignores patient clustering (hence appendix-only).
- Single-center German PTB-XL; no prospective or multi-site clinical validation.
- CLOCS and I-JEPA are **adaptations**, not official implementations.
- Deployment is a Kaggle T4 / Xeon **server proxy**.
- A 45 Hz front-end removes most 50 Hz mains content after Option-A injection.

---

## Configuration defaults

| Parameter | Value |
|-----------|--------|
| Band-pass | 0.05–45 Hz, order 4, `sosfiltfilt` |
| Label threshold | 50; exclude empty superclass rows |
| SSL epochs | 200; AdamW |
| Effective batch | 1024 where VRAM allows (CLOCS ViT: 128×8) |
| Pretrain seeds (primary) | 42, 123, 456 |
| Noise seeds | 42, 123, 456 |
| Probe | 50 epochs, val early-stop, val temperature, val F1 thresholds |
| ECE | 15 equal-width bins (equal-mass robustness column) |
| Bootstrap | 1000 patient resamples; ER bootstrap 200 |
| Index weights | AUROC 0.30, ECE 0.30, CKA 0.20, ER 0.20 |
| Gates (filters) | AUROC 0.70, ECE 0.15 |
| Quantization | FP32, INT8 dynamic, INT8 static (train calib) |
| Parity gate | cosine ≥ 0.999 (applied in script 07) |
