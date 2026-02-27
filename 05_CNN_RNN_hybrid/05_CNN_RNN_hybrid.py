"""
05_CNN_RNN_hybrid.py
====================
CNN + RNN HYBRIDS: CNN+GRU, CNN+LSTM, CNN+BiGRU.
Compatible with: TF 2.19.1 | Keras 3.13.2 | NumPy 2.0.2
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import gc
import keras
from keras.models import Sequential
from keras.layers import (Conv1D, MaxPooling1D, GRU, LSTM,
                          Bidirectional, Dense, Dropout)
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
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
print("  05 - CNN + RNN HYBRIDS")
print("=" * 60)

# ── Step 1: Load Data ─────────────────────────────────────────────────
print("Step 1: Loading data...")
X_raw, Y_onehot, y_int = load_data(MAT_PATH, normalize=True)
X = to_rnn_input(X_raw)   # (14000, 1000, 3)

X_all, X_test, yi_all, yi_test = train_test_split(
    X, y_int, test_size=0.2, stratify=y_int, random_state=RANDOM_STATE
)
y_test_cat = to_categorical(yi_test, num_classes=7)
print(f"  Training pool: {len(X_all)}  |  Test set: {len(X_test)}")


# ── Step 2: Model Builders ────────────────────────────────────────────

def build_cnn_gru():
    return Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(1000, 3)),
        MaxPooling1D(pool_size=2),
        GRU(128),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_cnn_lstm():
    return Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(1000, 3)),
        MaxPooling1D(pool_size=2),
        LSTM(128),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_cnn_bigru():
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

# ── Step 3: Training & Inline Evaluation ─────────────────────────────
print(f"\nStep 2: Training {len(MODEL_REGISTRY)} hybrid architectures...\n")

for model_name, builder in MODEL_REGISTRY.items():
    print(f"\n{'='*60}\n  MODEL: {model_name}  ({N_RUNS} runs)\n{'='*60}")

    run_accs, best_acc, best_model, best_hist = [], 0, None, None

    for run in range(N_RUNS):
        print(f"\n  Run {run+1}/{N_RUNS}")

        keras.backend.clear_session()
        gc.collect()

        X_tr, _, yi_tr, _ = train_test_split(
            X_all, yi_all, test_size=0.2, stratify=yi_all, random_state=run
        )
        y_tr = to_categorical(yi_tr, num_classes=7)

        m = builder()
        m.compile(optimizer=Adam(0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

        hist = m.fit(
            X_tr, y_tr,
            validation_split=0.2,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10,
                                     restore_best_weights=True, verbose=0)],
            verbose=1
        )

        _, acc = m.evaluate(X_test, y_test_cat, verbose=0)
        run_accs.append(acc)
        print(f"  Run {run+1}: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc, best_model, best_hist = acc, m, hist

    all_run_accuracies[model_name] = run_accs

    print(f"\n  INLINE EVALUATION - {model_name} (best: {best_acc*100:.2f}%)")
    plot_training_history(best_hist, model_name=model_name, save_dir=SAVE_DIR)

    final_acc, _ = evaluate_model(
        model=best_model, X_test=X_test,
        y_test_onehot=y_test_cat, y_test_int=yi_test,
        model_name=model_name, save_dir=SAVE_DIR, is_branch_input=False
    )

    summary_results.append({
        'Model': model_name, 'Best_Acc': best_acc,
        'Mean_Acc': np.mean(run_accs), 'Std_Acc': np.std(run_accs),
        'Runs': N_RUNS, 'Notes': 'CNN frontend 1000->499 steps before RNN'
    })
    print(f"\n  Summary: Best={best_acc*100:.2f}%  "
          f"Mean={np.mean(run_accs)*100:.2f}%  +/-{np.std(run_accs)*100:.2f}%")


# ── Step 4: Comparison & Summary ─────────────────────────────────────
compare_models_boxplot(all_run_accuracies,
                       title='CNN+RNN Hybrids - Accuracy (5 Runs Each)',
                       save_dir=SAVE_DIR, filename='CNN_RNN_hybrid_boxplot.png')

print_summary_table(summary_results, title='CNN+RNN HYBRIDS - RESULTS')
save_summary_csv(summary_results, 'CNN_RNN_hybrid_results.csv', save_dir=SAVE_DIR)

print("\n  -> Next: 06_model_comparison.py")
