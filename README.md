# Wind Turbine Fault Detection — Corrected & Rigorous Version

Classifying mechanical faults in wind turbine drivetrains from 3-axis vibration
signals using 1D CNNs, RNNs, and CNN+RNN hybrids.

---

## What This Version Fixes

This is the methodologically corrected version of the original project.
The original work had several evaluation and interpretability gaps:

| Problem | Fix Applied |
|---|---|
| CNN used single-split; RNN used 5-run → not comparable | Unified 5-run protocol across all models |
| Test set used as validation (EarlyStopping leakage) | Proper 3-way stratified split throughout |
| Multi-scale CNN (V5) not developed deeply | Script 03 makes it the primary architecture |
| No Grad-CAM — model is a black box | Grad-CAM runs inline after every CNN training |
| Attention weights never visualized | Attention plot runs inline for attention models |
| Vanilla RNN failure not explained | Mechanistic explanation in script 04 |
| Data not interrogated before modelling | Script 00 runs before any model code |

---

## Project Structure

```
wind-turbine-fault-detection/
│
├── utils/
│   ├── data_utils.py     # Loading, normalization, splits, augmentation
│   └── eval_utils.py     # Evaluation, Grad-CAM, attention viz, plots
│
├── 00_dataset_analysis/
│   └── 00_dataset_analysis.py    # Data interrogation before any modelling
│
├── 01_CNN_baseline/
│   └── 01_CNN_baseline.py        # Baseline — limitations preserved and documented
│
├── 02_CNN_improved/
│   └── 02_CNN_improved.py        # + Normalization, BatchNorm, Aug, EarlyStopping, 5-run
│
├── 03_CNN_multiscale_final/
│   └── 03_CNN_multiscale_final.py # Inception-style kernels [3,5,7] + Grad-CAM inline
│
├── 04_RNN_benchmark/
│   └── 04_RNN_benchmark.py        # 9 RNN variants, 5-run, attention viz inline
│
├── 05_CNN_RNN_hybrid/
│   └── 05_CNN_RNN_hybrid.py       # CNN+GRU, CNN+LSTM, CNN+BiGRU, 5-run
│
└── 06_model_comparison/
    └── 06_model_comparison.py     # Final apples-to-apples comparison of all models
```

---

## Dataset

**File:** `Turbine.mat` — place in the project root directory.

| Property | Value |
|---|---|
| Samples | 14,000 |
| Input shape | (3, 1000) — 3 axes × 1000 time points |
| Sampling rate | 100,000 Hz |
| Window duration | 10 ms |
| Classes | 7 (Normal + 6 fault types) |
| Label format | One-hot encoded |

---

## Evaluation Standard (Applied Consistently)

Every model in scripts 02–05 uses the same protocol so comparisons are valid:

```
Data split:   70% train | 15% val | 15% test  (stratified)
Augmentation: 4× training data (noise + time shift + random gain)
EarlyStopping: monitors val_loss only — test set NEVER seen in training
Runs:         5 runs with different random seeds → mean ± std reported
Test set:     fixed across all runs and all models
```

**CNN Baseline (script 01) is flagged as NOT directly comparable** — it uses
a 2-way split where the test set is exposed during training. Its accuracy
is optimistic and cannot be compared against 5-run results.

---

## Inline Evaluation Pipeline

Every model, immediately after training, runs this pipeline in sequence:

```
model.fit() completes
    → plot_training_history()          # training + validation curves
    → evaluate_model()                 # accuracy, confusion matrix, report CSV
    → run_gradcam_suite()              # [CNN models only] one plot per fault class
    → run_attention_suite()            # [attention models only] 3 sample plots
    → summary_results.append(...)      # add to running table
    → [next model]
...
print_summary_table()                  # printed once at end of script
```

There is no second plotting loop at the end. Everything runs inline.

---

## Interpretability

### Grad-CAM (CNN models — script 03)
Shows which time windows in the raw vibration signal drove each prediction.
- Red regions = time segments the model relied on most
- Green regions = time segments largely ignored
- One plot per fault class, saved to `03_CNN_multiscale_final/outputs/`

### Attention Weights (RNN models — script 04)
Shows the learned attention distribution over 1000 timesteps.
- Peaks in the attention plot = timesteps the model focused on
- Scientific check: do peaks align with expected fault-signal regions?
- 3 sample plots saved to `04_RNN_benchmark/outputs/`

---

## Running Order

```bash
pip install tensorflow numpy scipy scikit-learn matplotlib seaborn pandas

# Place Turbine.mat in project root, then:
python 00_dataset_analysis/00_dataset_analysis.py   # understand data first
python 01_CNN_baseline/01_CNN_baseline.py
python 02_CNN_improved/02_CNN_improved.py
python 03_CNN_multiscale_final/03_CNN_multiscale_final.py
python 04_RNN_benchmark/04_RNN_benchmark.py
python 05_CNN_RNN_hybrid/05_CNN_RNN_hybrid.py
python 06_model_comparison/06_model_comparison.py   # aggregates all results
```

Each script saves all outputs (plots, CSVs, models) to its own `outputs/` folder.
Script 06 reads those CSVs and produces the final unified comparison.

---

## Key Expected Findings

1. **CNN outperforms all RNN variants** — vibration fault classification is a
   pattern-matching problem. CNNs detect local frequency-domain signatures;
   RNNs are optimized for sequential reasoning.

2. **Multi-scale kernels improve CNN** — fault signatures appear at different
   frequency scales. Parallel kernels [3, 5, 7] capture all scales simultaneously.

3. **Attention improves RNNs** — GRU+Attention outperforms plain GRU because
   attention can focus on fault-relevant time windows rather than compressing
   the entire 1000-step sequence into one vector.

4. **CNN+RNN hybrids outperform pure RNNs** — CNN frontend compresses the
   1000-step raw sequence to ~499 steps, making the RNN task tractable.

5. **Vanilla RNN fails on 1000-step sequences** — vanishing gradients.
   This is an architectural failure mode, not a hyperparameter problem.
   Documented and explained in script 04.

6. **Variance matters** — single-run numbers can be misleading. The 5-run
   protocol shows which models are consistently good vs. which got lucky.

---

## References

- Szegedy et al. (2015). Going Deeper with Convolutions (Inception). *CVPR*.
- Bahdanau et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
- Lei et al. (2020). Applications of machine learning to machine fault diagnosis. *Mechanical Systems and Signal Processing*.
- Tautz-Weinert & Watson (2017). Using SCADA data for wind turbine condition monitoring. *IET Renewable Power Generation*.
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
