# Manuscript helper — seed-42 full run

Use this after the seed-42 robustness report. Treat every number here as **single-seed (42)**. After seeds 123 and 456, replace point estimates with mean ± SD for the primary arms and keep BYOL / SwAV / CLOCS-ResNet18 as flagged sensitivity arms.

**Verdict:** the experiment behaved as designed. Proceed with the remaining two pretraining seeds for SimCLR, CLOCS-ViT, MAE, I-JEPA, and the supervised reference. Do not add seeds for the three sensitivity arms.

---

## 1. What this run is

| Item | Value |
| --- | --- |
| Dataset | PTB-XL 1.0.3, official folds 1–8 / 9 / 10, patient-disjoint |
| Task | Multi-label diagnostic superclasses: NORM, MI, STTC, CD, HYP |
| Backbone | ViT-S 1D (~22M), except CLOCS-ResNet18 1D |
| Probe | Frozen CLS, clean-only linear head, one validation temperature |
| Grid | 6 single noises + 3 mixtures × SNR {24, 18, 12, 6, 0, −6} dB × 3 noise seeds |
| This report | 8 encoders × 9 types × 6 SNRs = **432 / 432 cells**, noise seeds already averaged |
| Missing for the paper | Pretrain seeds 123 and 456; keep `decay-metrics-results` for bootstrap CIs and BH-FDR |

Primary arms (need 3 seeds): SimCLR, CLOCS-adapted ViT, MAE, I-JEPA-adapted, supervised ViT-S.

Sensitivity arms (1 seed, already done): BYOL, SwAV (2-view), CLOCS-ResNet18.

---

## 2. One-sentence findings (seed 42)

1. **Clean ranking is conventional.** Supervised ViT-S leads (AUROC 0.902). MAE is the strongest SSL encoder (0.892). Contrastive and predictive SSL sit in 0.86–0.89. CLOCS-ResNet18 is weaker on clean data (0.838), as expected for a smaller CNN.
2. **Decay is graded, monotonic, and type-specific.** No AUROC increase as SNR falls. Muscle artifact (EM) is hardest, then MA and EM-containing mixtures; baseline wander and inverter are mild. This matches ECG physics and the 45 Hz front end (which attenuates 50 Hz powerline).
3. **The distinctive result is a backbone × objective interaction.** The same CLOCS objective is nearly invariant on ResNet-18 and brittle on ViT-S once high-frequency artifact is strong. CLOCS-ViT is not a collapsed pretrain: clean AUROC 0.878, clean ECE 0.020, clean CKA = 1.
4. **Geometry is not a substitute for task.** BYOL can move CKA by 0.3–0.6 while ΔAUROC stays ~0.02. Report ΔAUROC, ΔECE, and ΔF1 as primary evidence; treat ΔCKA / ΔER as diagnostics.
5. **The robustness index is a secondary, run-normalized summary.** It currently ranks CLOCS-ResNet18 first because that model has tiny deltas *and* because CLOCS-ViT stretches the min-max anchors. Do not lead the paper with *R*.

---

## 3. Tables you can paste (label: seed 42)

### 3.1 Clean reference (risk-report baselines)

| Encoder | Role | AUROC | ECE | F1-macro | F1-NORM | F1-MI | F1-STTC | F1-CD | F1-HYP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised ViT-S | Reference | 0.902 | 0.025 | 0.709 | 0.841 | 0.696 | 0.727 | 0.707 | 0.572 |
| MAE | Primary | 0.892 | 0.016 | 0.697 | 0.835 | 0.658 | 0.723 | 0.676 | 0.592 |
| SimCLR | Primary | 0.887 | 0.016 | 0.690 | 0.830 | 0.659 | 0.733 | 0.673 | 0.556 |
| I-JEPA-adapted | Primary | 0.884 | 0.014 | 0.688 | 0.821 | 0.650 | 0.722 | 0.670 | 0.574 |
| BYOL | Sensitivity | 0.879 | 0.015 | 0.672 | 0.823 | 0.626 | 0.706 | 0.647 | 0.558 |
| CLOCS-ViT | Primary | 0.878 | 0.020 | 0.672 | 0.799 | 0.658 | 0.663 | 0.676 | 0.562 |
| SwAV | Sensitivity | 0.864 | 0.016 | 0.655 | 0.810 | 0.629 | 0.689 | 0.665 | 0.484 |
| CLOCS-ResNet18 | Sensitivity | 0.838 | 0.019 | 0.623 | 0.758 | 0.599 | 0.546 | 0.688 | 0.525 |

Literature anchor (do not over-compare): Wagner / Strodthoff PTB-XL *diagnostic superclass* CNNs are ~0.92–0.93 (resnet1d_wang 0.930). Our supervised 0.902 is a ViT-S, epoch-parity, no architecture search. The gap is expected, not a data bug.

### 3.2 Prespecified primary family — 0 dB, mean of BW / MA / EM

| Encoder | AUROC | ΔAUROC | ECE | ΔECE | ΔCKA | ΔER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLOCS-ResNet18* | 0.835 | 0.003 | 0.022 | 0.004 | 0.012 | 0.029 |
| Supervised ViT-S | 0.889 | 0.013 | 0.031 | 0.006 | 0.179 | 0.075 |
| MAE | 0.874 | 0.017 | 0.065 | 0.049 | 0.293 | 0.072 |
| I-JEPA-adapted | 0.865 | 0.019 | 0.058 | 0.044 | 0.196 | 0.034 |
| BYOL* | 0.852 | 0.026 | 0.068 | 0.053 | 0.387 | 0.073 |
| SimCLR | 0.854 | 0.033 | 0.039 | 0.022 | 0.202 | 0.115 |
| SwAV* | 0.824 | 0.040 | 0.063 | 0.046 | 0.096 | 0.043 |
| CLOCS-ViT | 0.796 | 0.082 | 0.066 | 0.046 | 0.324 | 0.240 |

BH-FDR in the paper applies only to encoder × {BW, MA, EM} × class at 0 dB. Pull p-values from `decay-metrics-results/hypothesis_tests.parquet`, not from this report folder.

### 3.3 Worst operating point — −6 dB, mean over all 9 types

| Encoder | AUROC | ΔAUROC | ECE | ΔCKA | F1-macro | *R* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised ViT-S | 0.868 | 0.034 | 0.039 | 0.318 | 0.624 | 0.789 |
| MAE | 0.848 | 0.043 | 0.107 | 0.476 | 0.470 | 0.648 |
| I-JEPA-adapted | 0.843 | 0.041 | 0.077 | 0.404 | 0.550 | 0.697 |
| SimCLR | 0.828 | 0.059 | 0.053 | 0.300 | 0.546 | 0.730 |
| CLOCS-ResNet18* | 0.825 | 0.014 | 0.049 | 0.076 | 0.566 | 0.886 |
| BYOL* | 0.815 | 0.064 | 0.095 | 0.662 | 0.451 | 0.595 |
| SwAV* | 0.789 | 0.075 | 0.085 | 0.201 | 0.470 | 0.709 |
| CLOCS-ViT | 0.720 | 0.158 | 0.116 | 0.606 | 0.445 | 0.337 |

### 3.4 Headline interaction — −6 dB AUROC (and ΔAUROC)

| Noise | Supervised | MAE | I-JEPA | SimCLR | CLOCS-ViT | CLOCS-ResNet* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BW | 0.884 (0.018) | 0.870 (0.021) | 0.859 (0.025) | 0.835 (0.052) | 0.769 (0.109) | 0.834 (0.004) |
| MA | 0.863 (0.039) | 0.839 (0.053) | 0.825 (0.059) | 0.802 (0.085) | **0.647 (0.231)** | 0.825 (0.014) |
| EM | 0.809 (0.093) | 0.805 (0.086) | 0.795 (0.089) | 0.784 (0.103) | **0.619 (0.259)** | 0.816 (0.023) |
| Powerline | 0.887 (0.015) | 0.844 (0.048) | 0.854 (0.030) | 0.841 (0.046) | 0.751 (0.127) | 0.834 (0.004) |
| Electrode pop | 0.874 (0.028) | 0.876 (0.016) | 0.867 (0.017) | 0.870 (0.017) | 0.812 (0.066) | 0.793 (0.046) |

Electrode-pop is the odd type: model ranking of ΔAUROC correlates only ~0.2 with the other eight types. CLOCS-ResNet F1 falls to 0.335 there despite AUROC 0.793 — mention thresholded F1 separately.

---

## 4. Result paragraphs (draft, then insert CIs)

**Clean performance.** Under the frozen-encoder protocol, the supervised ViT-S reference reached macro-AUROC 0.902 and macro-F1 0.709 on the official PTB-XL test fold. Among self-supervised encoders, MAE was closest (0.892 / 0.697), followed by SimCLR (0.887) and the I-JEPA adaptation (0.884). CLOCS-ViT matched other contrastive methods on clean data (0.878) and did not show embedding collapse. The CLOCS-ResNet18 control was weaker on clean data (0.838), consistent with lower capacity rather than a training failure. All clean ECEs were ≤ 0.025 after a single validation-fitted temperature.

**Graded corruption.** Macro-AUROC declined monotonically as SNR decreased from 24 dB to −6 dB for every encoder × noise type (72 curves; no reversal larger than 0.2 percentage points). At 24 dB, mean ΔAUROC was ≤ 0.0004 and linear CKA remained ≥ 0.99, so the high-SNR end of the grid is an empirical clean check. Averaged across encoders at −6 dB, the noise ranking was EM (ΔAUROC 0.109) > EM+powerline (0.086) > MA (0.082) > BW+MA mixtures (0.057–0.063) > powerline (0.049) ≈ BW (0.043) > electrode pop (0.030) ≈ inverter (0.028). The mild powerline effect is expected: corruption is injected in raw millivolts and then passed through the 0.05–45 Hz front end, which attenuates 50 Hz mains content.

**Prespecified 0 dB family.** On BW / MA / EM at 0 dB, supervised ΔAUROC was 0.013 and MAE / I-JEPA remained within 0.02 of their clean scores. SimCLR decayed by 0.033. CLOCS-ViT was already an outlier (mean ΔAUROC 0.082), driven by MA (0.101) and EM (0.116) rather than BW (0.028). CLOCS-ResNet18 moved by only 0.003.

**Severe noise and the CLOCS dissociation.** At −6 dB, supervised AUROC remained 0.868 (Δ 0.034) and still won 52 of 54 type × SNR cells on absolute AUROC. MAE and I-JEPA lost 0.04 on average; SimCLR lost 0.059. CLOCS-ViT lost 0.158 and was the only encoder to fail the AUROC &lt; 0.70 gate (EM 0.619, MA 0.647, EM+powerline 0.659, BW+MA 0.697). The same CLOCS losses on ResNet-18 were 0.023 / 0.014 / 0.013 / 0.008. Because clean CLOCS-ViT performance is normal, the failure is a noise-time interaction of the ViT + patient/lead/temporal NT-Xent recipe with high-frequency artifact, not a dead run. This claim is the main reason the other two pretraining seeds are required.

**Calibration.** Clean models are well calibrated. At −6 dB, mean ECE rose most for CLOCS-ViT (0.116) and MAE (0.107) and least for supervised (0.039) and SimCLR (0.053). ECE &gt; 0.15 occurred in 7 of 432 cells, all at −6 dB, mostly EM or EM+powerline.

**Representation geometry.** Linear CKA and two-sided effective-rank change track ΔAUROC globally (Spearman 0.81 and 0.83) but can decouple on individual models. BYOL at 0 dB shows ΔCKA of 0.32–0.59 with ΔAUROC of only 0.012–0.032: the CLS geometry moves while the linear probe still separates the five superclasses. Conversely, every large task drop (ΔAUROC &gt; 0.08) also had ΔCKA &gt; 0.20. Follow the protocol: interpret robustness from task and calibration first.

**Secondary index.** With pre-registered weights (0.30 ΔAUROC, 0.30 ΔECE, 0.20 ΔCKA, 0.20 ΔER) and within-run min-max anchors, *R* at −6 dB ranks CLOCS-ResNet18 (0.886) &gt; supervised (0.789) &gt; SimCLR (0.730) ≫ CLOCS-ViT (0.337). Weight-perturbation Kendall τ = 0.81; top-1 agreement = 0.96 (almost always the ResNet). SimCLR (τ-stability 0.38) and SwAV (0.20) are sensitive to weight choice. State clearly that *R* is not a cross-paper absolute score and will change when later seeds update the anchors.

---

## 5. Claims allowed now versus after three seeds

| Claim | After seed 42 | After 42 / 123 / 456 |
| --- | --- | --- |
| The benchmark is executable and the corruption grid is well-behaved | Yes | Yes |
| Supervised &gt; MAE &gt; other SSL on clean PTB-XL superclasses | Pilot observation | Primary table (mean ± SD) |
| EM / MA dominate; BW and inverter are mild; powerline is front-end-limited | Yes | Yes, with CIs |
| CLOCS-ViT is uniquely brittle; CLOCS-CNN is uniquely stable | **Hypothesis / qualitative** | Primary claim if the gap replicates |
| BYOL geometry–task decoupling | Allowed as sensitivity | Keep flagged as 1-seed |
| Index ranking / “recommended configuration” | Appendix, qualified | Appendix; do not promote to abstract |
| Clinical safety or edge-device deployment | Never | Never |

Do not write “SSL is more robust than supervised” or the reverse as a blanket statement. Supervised wins absolute AUROC; the CNN-CLOCS control wins ΔAUROC; ViT-CLOCS loses both under EM/MA.

---

## 6. Suggested paper skeleton

**Title direction:** representation / task decay of ECG SSL under graded noise; the interesting noun is the CLOCS-ViT vs CLOCS-CNN dissociation, not a new index.

**Abstract (after 3 seeds):** one sentence setup (PTB-XL, frozen probe, 162 conditions); one sentence clean ranking; one sentence graded EM/MA decay; one sentence CLOCS backbone interaction with a mean ± SD gap; one sentence that geometry ≠ task and that the composite index is secondary. Last sentence: single-center engineering benchmark, not clinical validation.

**Figures (reuse the report atlas):**
1. Method schematic (already in the README mermaid).
2. AUROC vs SNR for BW, MA, EM (the three primary types). CLOCS-ViT vs CLOCS-ResNet vs supervised vs MAE is the story.
3. Heatmap: encoder × noise type at −6 dB, colored by ΔAUROC.
4. Scatter: ΔCKA vs ΔAUROC (BYOL off-diagonal, CLOCS-ViT on-diagonal).
5. Appendix: remaining types, mixtures, ECE, F1, *R* leaderboard, deployment profile.

**Discussion points that are already supported:**
- Epoch parity, not compute parity; contrastive methods share one augmentation policy, MAE/JEPA do not — differences are objective *and* view strategy.
- 2-channel NSTDB → 12-lead replication is an approximation; say so.
- Option-A injection (noise in raw mV, then the standard front end) is a feature: it is what a deployed model would see.
- Linear probe on clean data only is a frozen-encoder robustness test, not a claim about fine-tuning under noise.
- HYP is the weakest class on clean F1 for every encoder (0.48–0.59); do not over-interpret class-wise robustness until the three-seed CIs exist.

---

## 7. Pretrain the remaining seeds — practical order

1. **CLOCS-ViT 123, 456** — if the EM/MA collapse is seed-specific, the main story changes.
2. **Supervised 123, 456** — needed for every “SSL vs supervised” sentence.
3. **MAE, I-JEPA, SimCLR 123, 456** — needed for the primary mean ± SD table.

Then rerun 03 → 04 → 05 → 06 → 07 on the full checkpoint set. Drop stale script-05 and script-07 artifacts before citing numbers. Export `environment.lock.txt`.

Do **not** spend extra seeds on BYOL, SwAV, or CLOCS-ResNet18. The protocol already forbids ranking them as statistically equivalent to the primary set.

Approximate added cost: 10 pretrain sessions (5 arms × 2 seeds) plus one eval sweep.

---

## 8. Do not put these in the manuscript as-is

- **`decision_flowchart.md` names BYOL as “best balanced.”** That is a generator bug (`iloc[0]` on an unsorted frame). The leaderboard and `config_guidelines.md` correctly prefer CLOCS-ResNet18. Fix the helper or omit the flowchart.
- **`git_commit: null` in `config.json`.** Record the commit on the final paper run.
- **Index-based “RECOMMEND” text.** Fine in an appendix as decision-support output; not a scientific conclusion. Latency numbers are CPU ONNX server proxies, not edge hardware.
- **Memory 0.00 MB in the leaderboard.** Process RSS delta is not a useful memory claim; omit or replace with a measured encoder+probe footprint.

---

## 9. Phrase bank

- “We report a pre-registered secondary robustness index; the primary evidence is paired ΔAUROC, ΔECE, and ΔF1.”
- “CLOCS-ViT and CLOCS-ResNet18 share an objective and differ in backbone; their divergence under EM/MA is therefore a backbone–objective interaction.”
- “CKA and effective rank describe last-layer CLS geometry. They do not, by themselves, establish loss of task-relevant information.”
- “Sensitivity arms use one seed and are not compared as statistical peers of the three-seed primary paradigms.”
- “This is an engineering benchmark on a single-center German cohort. It is not a clinical validation study.”
