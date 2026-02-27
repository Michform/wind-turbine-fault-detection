"""
02_CNN_improved.py
==================
IMPROVED CNN: All the methodological fixes applied one step at a time.

Changes from baseline (01_CNN_baseline.py):
    ✅ Input normalization    → X / max_abs → range [-1, 1]
    ✅ BatchNormalization     → after every Conv1D layer
    ✅ Data augmentation      → Gaussian noise + time shift + random gain (4×)
    ✅ EarlyStopping          → patience=30, restore_best_weights=True
    ✅ ReduceLROnPlateau      → halves LR when val_loss plateaus
    ✅ Proper 3-way split     → separate val set; test set NEVER seen in training
    ✅ 5-run evaluation       → same protocol as RNN benchmark for fair comparison

Why each change matters:
    Normalization:
        Raw accelerometer axes can have amplitude differences of 10–100×.
        Without normalization, gradients are dominated by the highest-amplitude
        axis. Dividing by global max_abs puts all axes in [-1, 1].

    BatchNormalization:
        Normalizes activations within each mini-batch. Reduces internal
        covariate shift, allows higher learning rates, and typically improves
        both speed and final accuracy. Added after every Conv1D layer.

    Augmentation (training set ONLY):
        Gaussian noise  → robustness to sensor measurement noise
        Time shift      → invariance to when within the window a fault occurs
        Random gain     → robustness to sensor calibration drift between turbines
        Each strategy adds a full copy → 4× training data.

    EarlyStopping:
        Stops training when val_loss stops improving. Restores best weights.
        Prevents overfitting after the model has converged.

    ReduceLROnPlateau:
        Halves learning rate when val_loss plateaus for 10 epochs.
        Helps squeeze final performance without manual LR tuning.

    3-way split + 5-run protocol:
        The 5-run protocol (different random seed each run) gives a variance
        estimate — more honest than a single lucky/unlucky split.
        Now comparable to RNN benchmark results.

Inline evaluation (applied immediately after EACH of the 5 runs):
    → Training curves
    → Accuracy, confusion matrix, classification report
    → Summary table at end
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                                     Dense, Flatten, Concatenate, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import gc

from utils.data_utils import (load_data, split_data, augment_training_data,
                              to_channel_inputs)
from utils.eval_utils import (evaluate_model, plot_training_history,
                              compare_models_boxplot, print_summary_table,
                              save_summary_csv, save_model)

# ── Configuration ────────────────────────────────────────────────────
MAT_PATH     = os.path.join(os.path.dirname(__file__), '..', 'Turbine.mat')
SAVE_DIR     = os.path.join(os.path.dirname(__file__), 'outputs')
EPOCHS       = 100
BATCH_SIZE   = 64
N_RUNS       = 5
RANDOM_STATE = 42

os.makedirs(SAVE_DIR, exist_ok=True)
summary_results = []
run_accuracies  = []

print("=" * 60)
print("  02 — CNN IMPROVED")
print("=" * 60)
print("\n  Improvements over baseline:")
print("  + Normalization, BatchNorm, Augmentation")
print("  + EarlyStopping, ReduceLROnPlateau")
print("  + Proper 3-way stratified split")
print(f"  + {N_RUNS}-run evaluation protocol\n")

# ── Step 1: Load & Normalize ──────────────────────────────────────────
print("Step 1: Loading and normalizing data...")
X, Y_onehot, y_int = load_data(MAT_PATH, normalize=True)
print(f"  X range after normalization: [{X.min():.3f}, {X.max():.3f}]")

# Fixed test set — held out across all 5 runs
(X_train_full, X_val_full, X_test,
 y_train_full, y_val_full, y_test,
 yi_train_full, yi_val_full, yi_test) = split_data(
    X, Y_onehot, y_int,
    test_size=0.15, val_size=0.15,
    random_state=RANDOM_STATE
)

X_test_ch = to_channel_inputs(X_test)
print(f"\n  Test set fixed: {len(X_test)} samples — never touched during training.")

# ── Step 2: Model Builder ─────────────────────────────────────────────
def build_improved_model(n_classes: int = 7):
    """
    Improved branch: Conv1D → BatchNorm → MaxPool (×4 per branch).
    BatchNorm stabilizes activations, allows learning rate of 1e-4 without
    diverging, and generally speeds up convergence.
    """
    def build_branch(input_shape=(1000, 1), name=None):
        inp = Input(shape=input_shape, name=name)
        x   = Conv1D(64, 14, activation='relu')(inp)
        x   = BatchNormalization()(x)
        x   = MaxPooling1D(2)(x)
        x   = Conv1D(64, 14, activation='relu')(x)
        x   = BatchNormalization()(x)
        x   = MaxPooling1D(2)(x)
        x   = Conv1D(64, 14, activation='relu')(x)
        x   = BatchNormalization()(x)
        x   = MaxPooling1D(2)(x)
        x   = Conv1D(64, 14, activation='relu')(x)
        x   = BatchNormalization()(x)
        x   = MaxPooling1D(2)(x)
        x   = Dropout(0.5)(x)
        x   = Flatten()(x)
        return inp, x

    input_names = ['input_layer', 'input_layer_1', 'input_layer_2']
    branches, model_inputs = [], []
    for name in input_names:
        inp, out = build_branch(name=name)
        model_inputs.append(inp)
        branches.append(out)

    merged = Concatenate()(branches)
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    output = Dense(n_classes, activation='softmax')(merged)

    model = Model(inputs=model_inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Step 3: 5-Run Training Loop ───────────────────────────────────────
print(f"\nStep 2: Running {N_RUNS}-run training protocol...")
print("  Each run uses a different random seed for the train/val resample.")
print("  Test set remains FIXED across all runs for honest evaluation.\n")

best_acc   = 0
best_model = None
best_hist  = None

for run in range(N_RUNS):
    print(f"\n{'─' * 55}")
    print(f"  Run {run + 1} / {N_RUNS}")
    print(f"{'─' * 55}")

    tf.keras.backend.clear_session()
    gc.collect()

    # Re-sample train/val with different seed each run
    X_tr, X_v, yi_tr, yi_v = train_test_split(
        np.vstack([X_train_full, X_val_full]),
        np.concatenate([yi_train_full, yi_val_full]),
        test_size=0.15 / 0.85,
        stratify=np.concatenate([yi_train_full, yi_val_full]),
        random_state=run
    )
    y_tr = to_categorical(yi_tr, num_classes=7)
    y_v  = to_categorical(yi_v,  num_classes=7)

    # Augment training data only
    X_tr_aug, y_tr_aug = augment_training_data(
        X_tr, y_tr, noise=True, shift=True, gain=True
    )

    X_tr_ch = to_channel_inputs(X_tr_aug)
    X_v_ch  = to_channel_inputs(X_v)

    model = build_improved_model(n_classes=7)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-6, verbose=0)
    ]

    history = model.fit(
        X_tr_ch, y_tr_aug,
        validation_data=(X_v_ch, y_v),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ── INLINE EVALUATION — immediately after this run's training ────
    print(f"\n  Inline evaluation — Run {run + 1}:")

    # Training curves for this run
    plot_training_history(
        history,
        model_name=f'CNN Improved — Run {run + 1}',
        save_dir=SAVE_DIR
    )

    # Accuracy, confusion matrix, classification report
    run_acc, run_preds = evaluate_model(
        model         = model,
        X_test        = X_test_ch,
        y_test_onehot = y_test,
        y_test_int    = yi_test,
        model_name    = f'CNN Improved — Run {run + 1}',
        save_dir      = SAVE_DIR,
        is_branch_input = True
    )

    run_accuracies.append(run_acc)
    print(f"  Run {run + 1} test accuracy: {run_acc * 100:.2f}%")

    # Track best model across runs
    if run_acc > best_acc:
        best_acc   = run_acc
        best_model = model
        best_hist  = history
        print(f"  ✅ New best model (run {run + 1}): {best_acc * 100:.2f}%")

# ── Step 4: Summary Across All Runs ──────────────────────────────────
print(f"\n{'=' * 55}")
print(f"  5-Run Summary — CNN Improved")
print(f"{'=' * 55}")
print(f"  Best accuracy : {best_acc * 100:.2f}%")
print(f"  Mean accuracy : {np.mean(run_accuracies) * 100:.2f}%")
print(f"  Std accuracy  : {np.std(run_accuracies) * 100:.2f}%")
print(f"  All runs      : {[f'{a*100:.2f}%' for a in run_accuracies]}")

# Variance boxplot
compare_models_boxplot(
    {'CNN Improved': run_accuracies},
    title='CNN Improved — Accuracy Across 5 Runs',
    save_dir=SAVE_DIR,
    filename='CNN_Improved_variance_boxplot.png'
)

summary_results.append({
    'Model':    'CNN Improved',
    'Best_Acc': best_acc,
    'Mean_Acc': np.mean(run_accuracies),
    'Std_Acc':  np.std(run_accuracies),
    'Runs':     N_RUNS,
    'Notes':    '3-way split, norm, BatchNorm, aug, EarlyStopping, 5-run'
})

# ── Step 5: Save Best Model ───────────────────────────────────────────
save_model(best_model, 'CNN_Improved_best', save_dir=SAVE_DIR)

# ── Step 6: Final Summary Table ───────────────────────────────────────
print_summary_table(summary_results, title='CNN IMPROVED — RESULTS SUMMARY')
save_summary_csv(summary_results, 'CNN_Improved_results.csv', save_dir=SAVE_DIR)

print("\n" + "=" * 60)
print("  INTERPRETATION:")
print(f"  Best accuracy (5-run): {best_acc * 100:.2f}%")
print(f"  Mean accuracy (5-run): {np.mean(run_accuracies) * 100:.2f}% "
      f"± {np.std(run_accuracies) * 100:.2f}%")
print("  This number is NOW COMPARABLE to RNN benchmark results.")
print("  Same protocol: 5 runs, 3-way split, fixed test set.")
print("\n  → Next: 03_CNN_multiscale_final.py")
print("    (adds Inception-style multi-scale kernels as primary architecture)")
print("=" * 60)
