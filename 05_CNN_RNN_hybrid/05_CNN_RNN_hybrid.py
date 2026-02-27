"""
05_CNN_RNN_hybrid.py
====================
CNN + RNN HYBRIDS: Three architectures combining spatial and temporal modelling.

Models:
    1. CNN + GRU       — CNN extracts local features, GRU models temporal dependencies
    2. CNN + LSTM      — same idea with LSTM instead of GRU
    3. CNN + BiGRU     — bidirectional GRU reads compressed features in both directions

Why hybrid architectures?
    Pure RNNs on raw 1000-step sequences face two problems:
        1. Long-range gradient flow across 1000 recurrent steps (even with gating)
        2. No explicit mechanism to detect LOCAL frequency-domain patterns

    The CNN frontend solves both:
        - Conv1D layers learn to detect local fault signatures (transients, oscillations)
        - MaxPooling1D(2) compresses 1000 steps → ~499 steps
        - The RNN then models temporal evolution of these compressed features
          instead of raw signal samples

    This is a principled design: CNNs for WHAT (pattern presence),
    RNNs for HOW (pattern dynamics over time within the window).

Architecture:
    Conv1D(64, k=3, relu) → MaxPool(2)   [1000 → 499 steps]
         ↓
    GRU(128) / LSTM(128) / BiGRU(128)
         ↓
    Dropout(0.3)
         ↓
    Dense(7, softmax)

Evaluation protocol:
    5 runs per model, different random seed each run.
    Same fixed test set as all other scripts → directly comparable results.

Inline evaluation (applied immediately after EACH model's training):
    → Training curves
    → Accuracy, confusion matrix, classification report CSV
    → Results appended to summary table
    → Final boxplot and summary table at end
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import gc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, GRU, LSTM,
                                     Bidirectional, Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from utils.data_utils import load_data, to_rnn_input
from utils.eval_utils import (evaluate_model, plot_training_history,
                              compare_models_boxplot, print_summary_table,
                              save_summary_csv)

# ── Configuration ────────────────────────────────────────────────────
MAT_PATH     = os.path.join(os.path.dirname(__file__), '..', 'Turbine.mat')
SAVE_DIR     = os.path.join(os.path.dirname(__file__), 'outputs')
N_RUNS       = 5
EPOCHS       = 50
BATCH_SIZE   = 128
RANDOM_STATE = 42

os.makedirs(SAVE_DIR, exist_ok=True)
summary_results    = []
all_run_accuracies = {}

print("=" * 60)
print("  05 — CNN + RNN HYBRIDS")
print("=" * 60)
print("\n  CNN frontend compresses 1000 → ~499 steps")
print("  RNN backend models temporal dynamics of compressed features")
print(f"  {N_RUNS}-run protocol, same test set as scripts 03 and 04\n")


# ── Step 1: Load Data ─────────────────────────────────────────────────
print("Step 1: Loading data...")
X_raw, Y_onehot, y_int = load_data(MAT_PATH, normalize=True)
X = to_rnn_input(X_raw)   # (14000, 1000, 3)

# Fixed test set
X_all, X_test, yi_all, yi_test = train_test_split(
    X, y_int,
    test_size=0.2, stratify=y_int, random_state=RANDOM_STATE
)
y_test_cat = to_categorical(yi_test, num_classes=7)
print(f"  Training pool: {len(X_all)}  |  Test set: {len(X_test)}")


# ── Step 2: Model Builders ────────────────────────────────────────────

def build_cnn_gru():
    """
    CNN + GRU.
    Conv1D extracts local spectral features.
    GRU then processes the compressed sequence of feature vectors.
    Efficient: the GRU only needs to model ~499 steps (post-pooling)
    rather than the raw 1000.
    """
    return Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(1000, 3)),
        MaxPooling1D(pool_size=2),
        GRU(128),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_cnn_lstm():
    """
    CNN + LSTM.
    Identical structure to CNN+GRU but with LSTM.
    LSTM has 4 gates vs GRU's 3 — more parameters, slower convergence,
    but can sometimes capture longer dependencies.
    At ~499 steps (post-pool), GRU and LSTM are typically comparable.
    """
    return Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(1000, 3)),
        MaxPooling1D(pool_size=2),
        LSTM(128),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_cnn_bigru():
    """
    CNN + Bidirectional GRU.
    BiGRU processes the CNN-compressed features both forward and backward.
    Output dimension: 2×128 = 256 (concatenated forward + backward states).
    Useful when fault patterns have context-dependent structure —
    e.g., a transient at the end of the window informs interpretation
    of what came before.
    """
    return Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(1000, 3)),
        MaxPooling1D(pool_size=2),
        Bidirectional(GRU(128)),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])


MODEL_REGISTRY = {
    'CNN + GRU':   build_cnn_gru,
    'CNN + LSTM':  build_cnn_lstm,
    'CNN + BiGRU': build_cnn_bigru,
}


# ── Step 3: Training & Inline Evaluation Loop ─────────────────────────
print(f"\nStep 2: Training {len(MODEL_REGISTRY)} hybrid architectures...\n")

for model_name, builder in MODEL_REGISTRY.items():
    print(f"\n{'═' * 60}")
    print(f"  MODEL: {model_name}  ({N_RUNS} runs)")
    print(f"{'═' * 60}")

    run_accs  = []
    best_acc  = 0
    best_model = None
    best_hist  = None

    for run in range(N_RUNS):
        print(f"\n  Run {run + 1} / {N_RUNS}")

        tf.keras.backend.clear_session()
        gc.collect()

        X_tr, _, yi_tr, _ = train_test_split(
            X_all, yi_all,
            test_size=0.2,
            stratify=yi_all,
            random_state=run
        )
        y_tr = to_categorical(yi_tr, num_classes=7)

        m = builder()
        m.compile(
            optimizer=Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        es = EarlyStopping(monitor='val_loss', patience=10,
                           restore_best_weights=True, verbose=0)
        hist = m.fit(
            X_tr, y_tr,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=1
        )

        _, acc = m.evaluate(X_test, y_test_cat, verbose=0)
        run_accs.append(acc)
        print(f"  Run {run + 1}: {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc   = acc
            best_model = m
            best_hist  = hist

    all_run_accuracies[model_name] = run_accs

    # ── INLINE EVALUATION — immediately after this model ─────────────
    print(f"\n  INLINE EVALUATION — {model_name} (best: {best_acc*100:.2f}%)")

    # Training curves
    plot_training_history(
        best_hist,
        model_name=model_name,
        save_dir=SAVE_DIR
    )

    # Accuracy, confusion matrix, classification report
    final_acc, _ = evaluate_model(
        model           = best_model,
        X_test          = X_test,
        y_test_onehot   = y_test_cat,
        y_test_int      = yi_test,
        model_name      = model_name,
        save_dir        = SAVE_DIR,
        is_branch_input = False
    )

    summary_results.append({
        'Model':    model_name,
        'Best_Acc': best_acc,
        'Mean_Acc': np.mean(run_accs),
        'Std_Acc':  np.std(run_accs),
        'Runs':     N_RUNS,
        'Notes':    'CNN frontend compresses 1000→499 steps before RNN'
    })

    print(f"\n  Summary: Best={best_acc*100:.2f}%  "
          f"Mean={np.mean(run_accs)*100:.2f}%  "
          f"±{np.std(run_accs)*100:.2f}%")


# ── Step 4: Comparison Boxplot ────────────────────────────────────────
compare_models_boxplot(
    all_run_accuracies,
    title='CNN+RNN Hybrids — Accuracy Distribution (5 Runs Each)',
    save_dir=SAVE_DIR,
    filename='CNN_RNN_hybrid_boxplot.png'
)


# ── Step 5: Final Summary Table ───────────────────────────────────────
print_summary_table(summary_results, title='CNN+RNN HYBRIDS — RESULTS SUMMARY')
save_summary_csv(summary_results, 'CNN_RNN_hybrid_results.csv', save_dir=SAVE_DIR)


print("\n" + "=" * 60)
print("  KEY FINDINGS:")
print("  Expected result: CNN+RNN hybrids outperform pure RNNs (04)")
print("  because the CNN pre-processes the 1000-step raw sequence into")
print("  a compact feature representation the RNN can handle better.")
print()
print("  However: the pure multi-branch CNN (03) is still expected to")
print("  outperform all hybrids — vibration fault classification is a")
print("  PATTERN DETECTION problem (CNNs excel), not a sequential")
print("  reasoning problem (RNNs excel).")
print()
print("  → Next: 06_model_comparison.py")
print("    (final side-by-side table of all models, all scripts)")
print("=" * 60)
