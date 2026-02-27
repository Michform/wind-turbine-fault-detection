"""
01_CNN_baseline.py
==================
BASELINE: Multi-Branch 1D CNN — no augmentation, no normalization.

Purpose:
    Establish whether the core multi-branch architecture works at all.
    This is the starting point — not the answer. Every improvement in
    scripts 02 and 03 is measured against this baseline.

Documented limitations (deliberately preserved here):
    ✗ No input normalization → raw amplitude differences across axes
    ✗ No BatchNormalization → less stable training
    ✗ No data augmentation → model sees only 11,200 samples
    ✗ 2-way split: test set also serves as validation (information leakage)
      → EarlyStopping monitors the test set, so it's technically seen
        during training. The reported accuracy is therefore optimistic.
    ✗ Single run — no variance estimate across different random splits

Why document the limitations?
    The accuracy number from this script CANNOT be fairly compared to
    5-run results from later scripts. The gap between this and script 03
    includes both architectural improvements AND methodological corrections.
    Separating these two contributions is important for honest reporting.

Architecture:
    3 parallel CNN branches, one per vibration axis (X, Y, Z)
    Each branch: 4× [Conv1D(64, k=14) → MaxPool(2)]
    All branches concatenated → Dense(64) → Dense(7, softmax)

Inline evaluation (applied immediately after training):
    → Training curves
    → Test accuracy + classification report
    → Confusion matrix
    → Results appended to summary table
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D,
                                     Dropout, Dense, Flatten, Concatenate)
from tensorflow.keras.models import Model

from utils.data_utils import load_data, split_data_2way, to_channel_inputs
from utils.eval_utils import (evaluate_model, plot_training_history,
                              print_summary_table, save_summary_csv, save_model)

# ── Configuration ────────────────────────────────────────────────────
MAT_PATH     = os.path.join(os.path.dirname(__file__), '..', 'Turbine.mat')
SAVE_DIR     = os.path.join(os.path.dirname(__file__), 'outputs')
EPOCHS       = 50
BATCH_SIZE   = 64
RANDOM_STATE = 42

os.makedirs(SAVE_DIR, exist_ok=True)
summary_results = []   # populated after each model run; printed at end

print("=" * 60)
print("  01 — CNN BASELINE")
print("=" * 60)
print("\n  Limitations preserved intentionally:")
print("  - No normalization")
print("  - No augmentation")
print("  - 2-way split (test set leaks into EarlyStopping)")
print("  - Single run (no variance estimate)")
print("  These are fixed progressively in scripts 02 and 03.\n")

# ── Step 1: Load Data (no normalization — baseline condition) ─────────
print("Step 1: Loading data (normalize=False for baseline)...")
X, Y_onehot, y_int = load_data(MAT_PATH, normalize=False)
print(f"  X shape: {X.shape}  |  Y shape: {Y_onehot.shape}")

# ── Step 2: 2-Way Split (documented flaw) ────────────────────────────
print("\nStep 2: 2-way train/test split (known limitation)...")
X_train, X_test, y_train, y_test, yi_train, yi_test = split_data_2way(
    X, Y_onehot, y_int, test_size=0.2, random_state=RANDOM_STATE
)

print(f"\n  NOTE: In this script, X_test is used as validation during training.")
print(f"  EarlyStopping (if used) would monitor test set — information leakage.")
print(f"  This is the standard 2-way split used in the original notebooks.")
print(f"  Script 03 corrects this with a proper 3-way stratified split.\n")

X_train_ch = to_channel_inputs(X_train)
X_test_ch  = to_channel_inputs(X_test)

# ── Step 3: Build Model ───────────────────────────────────────────────
print("Step 3: Building baseline multi-branch CNN...")

def build_branch(input_shape=(1000, 1), name=None):
    """
    Single CNN branch processing one vibration axis.
    4 blocks of Conv1D(64, k=14) → MaxPool(2).
    Fixed kernel size 14 — no multi-scale yet (that's script 03).
    """
    inp = Input(shape=input_shape, name=name)
    x   = Conv1D(64, 14, activation='relu')(inp)
    x   = MaxPooling1D(2)(x)
    x   = Conv1D(64, 14, activation='relu')(x)
    x   = MaxPooling1D(2)(x)
    x   = Conv1D(64, 14, activation='relu')(x)
    x   = MaxPooling1D(2)(x)
    x   = Conv1D(64, 14, activation='relu')(x)
    x   = MaxPooling1D(2)(x)
    x   = Dropout(0.5)(x)
    x   = Flatten()(x)
    return inp, x

INPUT_NAMES = ['input_layer', 'input_layer_1', 'input_layer_2']

branches, model_inputs = [], []
for name in INPUT_NAMES:
    inp, out = build_branch(name=name)
    model_inputs.append(inp)
    branches.append(out)

merged = Concatenate()(branches)
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.5)(merged)
output = Dense(Y_onehot.shape[1], activation='softmax')(merged)

model = Model(inputs=model_inputs, outputs=output)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Step 4: Train ─────────────────────────────────────────────────────
print("\nStep 4: Training (50 epochs, no EarlyStopping)...")
history = model.fit(
    X_train_ch,
    y_train,
    validation_data=(X_test_ch, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)
# ── Step 5: INLINE EVALUATION ─────────────────────────────────────────
# This block runs IMMEDIATELY after training — not at the bottom of the script.
print("\nStep 5: Inline evaluation...")

# 5a. Training curves
plot_training_history(history, model_name='CNN Baseline', save_dir=SAVE_DIR)

# 5b. Accuracy, confusion matrix, classification report
test_acc, y_pred_int = evaluate_model(
    model       = model,
    X_test      = X_test_ch,
    y_test_onehot = y_test,
    y_test_int  = yi_test,
    model_name  = 'CNN Baseline',
    save_dir    = SAVE_DIR,
    is_branch_input = True
)

# 5c. No Grad-CAM here — baseline has no normalization, accuracy is already
#     compromised; Grad-CAM is run on the final model (script 03) where the
#     model is actually trustworthy.
print("\n  Note: Grad-CAM deferred to script 03 (final model).")
print("  Running Grad-CAM on an untrusted baseline would produce misleading")
print("  interpretability outputs.")

# 5d. Append to summary
summary_results.append({
    'Model':     'CNN Baseline',
    'Best_Acc':  test_acc,
    'Mean_Acc':  test_acc,     # single run
    'Std_Acc':   0.0,
    'Runs':      1,
    'Notes':     '2-way split, no norm, no aug — LEAKY EVALUATION'
})

# ── Step 6: Save Model ────────────────────────────────────────────────
save_model(model, 'CNN_Baseline', save_dir=SAVE_DIR)

# ── Step 7: Summary Table ─────────────────────────────────────────────
print_summary_table(summary_results, title='CNN BASELINE — RESULTS SUMMARY')

save_summary_csv(summary_results, 'CNN_Baseline_results.csv', save_dir=SAVE_DIR)

print("\n" + "=" * 60)
print("  INTERPRETATION:")
print(f"  Baseline accuracy: {test_acc*100:.2f}%")
print("  This number is OPTIMISTIC due to:")
print("    1. No normalization (raw amplitude training)")
print("    2. 2-way split (test set used as validation = leakage)")
print("    3. Single run (could be lucky split)")
print("  Do not compare this directly against 5-run results from later scripts.")
print("\n  → Next: 02_CNN_improved.py")
print("=" * 60)
