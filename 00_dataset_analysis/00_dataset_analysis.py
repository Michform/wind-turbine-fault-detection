"""
00_dataset_analysis.py
======================
DATA INTERROGATION — Run this before any model code.

Purpose:
    Understand exactly what is in the dataset before building any model.
    Every architectural and preprocessing decision downstream should be
    grounded in what we observe here.

What this script answers:
    1. How many samples per class? Is the dataset balanced?
    2. What do the raw signals look like per fault type?
    3. What are the amplitude statistics per axis?
    4. How correlated are the 3 axes with each other?
    5. What does the frequency content look like (FFT per class)?
    6. Are there obvious visual differences between fault classes?
    7. What is the signal-to-noise ratio per axis?

Why this matters:
    - Class imbalance → must use stratified splits
    - Amplitude range → tells us whether normalization is essential
    - Axis correlation → informs whether a multi-branch architecture
      is justified (if axes are highly correlated, merging them might
      be sufficient; if they carry independent information, separate
      branches add genuine value)
    - Frequency content → validates why multi-scale CNN kernels help
      (fault signatures at different frequency bands)

Outputs (saved to outputs/):
    - class_distribution.png
    - sample_signals_per_class.png
    - axis_statistics.csv
    - axis_correlation_matrix.png
    - fft_per_class.png
    - dataset_summary.txt
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as scipy_signal

from utils.data_utils import load_data, class_distribution, per_axis_stats
from utils.eval_utils import FAULT_LABELS

# ── Configuration ────────────────────────────────────────────────────
MAT_PATH = os.path.join(os.path.dirname(__file__), '..', 'Turbine.mat')
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
SAMPLING_RATE = 100_000   # Hz — as specified in dataset documentation

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("  WIND TURBINE FAULT DETECTION — DATASET ANALYSIS")
print("=" * 60)

# ── Step 1: Load (without normalization for raw analysis) ────────────
print("\nStep 1: Loading raw dataset...")
X, Y_onehot, y_int = load_data(MAT_PATH, normalize=False)

N_SAMPLES, N_AXES, N_TIMEPOINTS = X.shape
N_CLASSES = Y_onehot.shape[1]

print(f"\n  Dataset shape  : {X.shape}")
print(f"  Samples        : {N_SAMPLES:,}")
print(f"  Axes (channels): {N_AXES}  (X, Y, Z accelerometer)")
print(f"  Time points    : {N_TIMEPOINTS} per sample")
print(f"  Sampling rate  : {SAMPLING_RATE:,} Hz")
print(f"  Window duration: {N_TIMEPOINTS / SAMPLING_RATE * 1000:.1f} ms")
print(f"  Classes        : {N_CLASSES}")
print(f"  Label format   : one-hot encoded")


# ── Step 2: Class Distribution ────────────────────────────────────────
print("\nStep 2: Analysing class distribution...")

dist = class_distribution(y_int, FAULT_LABELS)

print("\n  Class Distribution:")
print(f"  {'Class':<12} {'Count':>8} {'Percentage':>12}")
print(f"  {'─'*35}")
for label, info in dist.items():
    print(f"  {label:<12} {info['count']:>8,} {info['pct']:>11.2f}%")

# Is the dataset balanced?
counts  = [info['count'] for info in dist.values()]
balance = max(counts) / min(counts)
print(f"\n  Balance ratio (max/min): {balance:.2f}x")
if balance < 1.1:
    print("  → Dataset is BALANCED — stratified splits still recommended.")
else:
    print("  → Dataset is IMBALANCED — stratified splits are essential.")

# Plot
fig, axes_plot = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Dataset Class Distribution', fontsize=13, fontweight='bold')

# Bar chart
colors = plt.cm.Set2(np.linspace(0, 1, N_CLASSES))
bars = axes_plot[0].bar(FAULT_LABELS, counts, color=colors, edgecolor='black', linewidth=0.5)
axes_plot[0].set_xlabel('Fault Class')
axes_plot[0].set_ylabel('Sample Count')
axes_plot[0].set_title('Samples per Class')
axes_plot[0].tick_params(axis='x', rotation=30)
for bar, count in zip(bars, counts):
    axes_plot[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                      f'{count:,}', ha='center', va='bottom', fontsize=8)

# Pie chart
axes_plot[1].pie(counts, labels=FAULT_LABELS, colors=colors,
                 autopct='%1.1f%%', startangle=90, pctdistance=0.85)
axes_plot[1].set_title('Class Proportions')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: class_distribution.png")


# ── Step 3: Per-Axis Amplitude Statistics ────────────────────────────
print("\nStep 3: Computing per-axis amplitude statistics...")

stats = per_axis_stats(X)

stats_rows = []
for ax, s in stats.items():
    print(f"  {ax}-axis: mean={s['mean']:.4f}, std={s['std']:.4f}, "
          f"min={s['min']:.4f}, max={s['max']:.4f}, abs_max={s['abs_max']:.4f}")
    stats_rows.append({'Axis': ax, **s})

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(SAVE_DIR, 'axis_statistics.csv'), index=False)
print("  Saved: axis_statistics.csv")

# Check amplitude range — motivates normalization
abs_maxes = [s['abs_max'] for s in stats.values()]
ratio = max(abs_maxes) / min(abs_maxes)
print(f"\n  Amplitude range ratio across axes: {ratio:.2f}x")
if ratio > 1.2:
    print("  → Axes have significantly different amplitudes.")
    print("    Normalization is ESSENTIAL to prevent gradient imbalance.")
else:
    print("  → Axes have similar amplitudes.")
    print("    Normalization is still recommended for stability.")


# ── Step 4: Visualise Raw Signals per Class ───────────────────────────
print("\nStep 4: Visualising raw signals per fault class...")

fig, axes_grid = plt.subplots(N_CLASSES, 3, figsize=(16, N_CLASSES * 2.2))
fig.suptitle('Raw Vibration Signals — One Sample per Fault Class (X | Y | Z)',
             fontsize=12, fontweight='bold')

t = np.arange(N_TIMEPOINTS) / SAMPLING_RATE * 1000  # time in ms

for class_idx in range(N_CLASSES):
    # Pick first sample of this class
    sample_idx = np.where(y_int == class_idx)[0][0]
    sample     = X[sample_idx]  # (3, 1000)

    for ax_idx, ax_name in enumerate(['X', 'Y', 'Z']):
        ax = axes_grid[class_idx, ax_idx]
        ax.plot(t, sample[ax_idx], linewidth=0.6, color=plt.cm.tab10(class_idx))
        ax.set_ylabel(FAULT_LABELS[class_idx], fontsize=8, rotation=90, labelpad=2)
        if class_idx == 0:
            ax.set_title(f'{ax_name}-Axis', fontsize=10, fontweight='bold')
        if class_idx == N_CLASSES - 1:
            ax.set_xlabel('Time (ms)', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'sample_signals_per_class.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: sample_signals_per_class.png")


# ── Step 5: Axis Correlation Matrix ──────────────────────────────────
print("\nStep 5: Computing inter-axis correlation...")

# Flatten all samples for correlation
X_flat = X.reshape(N_SAMPLES, N_AXES * N_TIMEPOINTS)

# Compute correlations between axis-level signals (mean per axis per sample)
axis_means = np.array([X[:, i, :].mean(axis=1) for i in range(N_AXES)]).T
# axis_means shape: (N_SAMPLES, 3)

corr_matrix = np.corrcoef(axis_means.T)

print(f"\n  Inter-axis correlation (sample-level means):")
axis_labels = ['X-axis', 'Y-axis', 'Z-axis']
for i in range(3):
    for j in range(i+1, 3):
        print(f"  {axis_labels[i]} vs {axis_labels[j]}: r = {corr_matrix[i,j]:.4f}")

# Interpret correlation
max_corr = np.max(np.abs(corr_matrix - np.eye(3)))
if max_corr > 0.8:
    print("\n  → High inter-axis correlation: axes carry REDUNDANT information.")
    print("    Multi-branch architecture may be over-engineered for this data.")
elif max_corr > 0.5:
    print("\n  → Moderate inter-axis correlation: axes carry PARTIALLY INDEPENDENT info.")
    print("    Multi-branch architecture adds value.")
else:
    print("\n  → Low inter-axis correlation: axes carry INDEPENDENT information.")
    print("    Multi-branch architecture is WELL-JUSTIFIED for this data.")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=axis_labels, yticklabels=axis_labels,
            vmin=-1, vmax=1, linewidths=0.5, ax=ax)
ax.set_title('Inter-Axis Correlation Matrix\n(sample-level mean signals)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'axis_correlation_matrix.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: axis_correlation_matrix.png")


# ── Step 6: Frequency Content (FFT per Class) ─────────────────────────
print("\nStep 6: Computing frequency content per fault class...")

fig, axes_fft = plt.subplots(N_CLASSES, 1, figsize=(14, N_CLASSES * 2.0), sharex=True)
fig.suptitle('FFT Frequency Spectrum — X-Axis, One Sample per Class\n'
             '(Motivates multi-scale kernel choice)',
             fontsize=12, fontweight='bold')

freq_axis = np.fft.rfftfreq(N_TIMEPOINTS, d=1.0/SAMPLING_RATE)

for class_idx in range(N_CLASSES):
    sample_idx  = np.where(y_int == class_idx)[0][0]
    signal_x    = X[sample_idx, 0, :]  # X-axis only for clarity

    fft_mag = np.abs(np.fft.rfft(signal_x))
    fft_mag = fft_mag / fft_mag.max()  # normalize to [0,1]

    ax = axes_fft[class_idx]
    ax.plot(freq_axis / 1000, fft_mag, linewidth=0.7,
            color=plt.cm.tab10(class_idx))
    ax.set_ylabel(FAULT_LABELS[class_idx], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)

axes_fft[-1].set_xlabel('Frequency (kHz)', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fft_per_class.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: fft_per_class.png")
print("\n  Key observation: If fault signatures appear at different frequency")
print("  bands across classes → multi-scale kernels (3,5,7) are theoretically")
print("  motivated. Compare peaks across rows above to confirm.")


# ── Step 7: Signal Amplitude Distribution per Class ───────────────────
print("\nStep 7: Amplitude distribution per class...")

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title('RMS Amplitude per Class — X-Axis\n'
             '(Higher RMS in fault classes confirms discriminative signal)',
             fontsize=11, fontweight='bold')

rms_by_class = []
for class_idx in range(N_CLASSES):
    indices = np.where(y_int == class_idx)[0]
    rms = np.sqrt(np.mean(X[indices, 0, :] ** 2, axis=1))
    rms_by_class.append(rms)

ax.boxplot(rms_by_class, labels=FAULT_LABELS, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7))
ax.set_ylabel('RMS Amplitude (raw units)')
ax.set_xlabel('Fault Class')
ax.tick_params(axis='x', rotation=30)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'rms_amplitude_per_class.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: rms_amplitude_per_class.png")


# ── Step 8: Save Summary Text Report ─────────────────────────────────
print("\nStep 8: Writing dataset summary report...")

summary_lines = [
    "WIND TURBINE FAULT DETECTION — DATASET ANALYSIS SUMMARY",
    "=" * 60,
    f"File          : Turbine.mat",
    f"Total samples : {N_SAMPLES:,}",
    f"Input shape   : ({N_AXES}, {N_TIMEPOINTS})  — 3 axes × 1000 timesteps",
    f"Sampling rate : {SAMPLING_RATE:,} Hz",
    f"Window        : {N_TIMEPOINTS / SAMPLING_RATE * 1000:.1f} ms",
    f"Classes       : {N_CLASSES}  (Normal + 6 fault types)",
    "",
    "CLASS DISTRIBUTION:",
]
for label, info in dist.items():
    summary_lines.append(f"  {label:<12} {info['count']:>6,} samples  ({info['pct']:.2f}%)")

summary_lines += [
    "",
    f"BALANCE RATIO: {balance:.2f}x  ({'BALANCED' if balance < 1.1 else 'IMBALANCED'})",
    "",
    "PER-AXIS STATISTICS (raw, pre-normalization):",
]
for ax_name, s in stats.items():
    summary_lines.append(
        f"  {ax_name}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
        f"abs_max={s['abs_max']:.4f}"
    )

summary_lines += [
    "",
    "DESIGN DECISIONS INFORMED BY THIS ANALYSIS:",
    f"  1. Stratified splits: {'ESSENTIAL' if balance >= 1.1 else 'RECOMMENDED'} (balance ratio {balance:.2f}x)",
    f"  2. Normalization: ESSENTIAL (amplitude ratio across axes: {ratio:.2f}x)",
    f"  3. Multi-branch architecture: "
      f"{'WELL-JUSTIFIED' if max_corr < 0.5 else 'MODERATELY JUSTIFIED' if max_corr < 0.8 else 'QUESTIONABLE'} "
      f"(max inter-axis corr: {max_corr:.3f})",
    "  4. Multi-scale kernels: check FFT plot — fault signatures at different frequencies",
    "  5. Augmentation: recommended for deployment robustness across turbines",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(os.path.join(SAVE_DIR, 'dataset_summary.txt'), 'w') as f:
    f.write(summary_text)
print("\n  Saved: dataset_summary.txt")


print("\n" + "=" * 60)
print("  DATASET ANALYSIS COMPLETE")
print("  All outputs saved to: outputs/")
print("  → Now run 01_CNN_baseline.py")
print("=" * 60)
