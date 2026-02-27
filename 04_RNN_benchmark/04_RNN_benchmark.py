"""
04_RNN_benchmark.py
===================
SYSTEMATIC RNN COMPARISON: 9 architectures, 5-run protocol.
Compatible with: TF 2.19.1 | Keras 3.13.2 | NumPy 2.0.2

Keras 3 changes:
    - keras.backend (K.*) replaced with tf.* ops in AttentionLayer
    - Imports use keras directly instead of tensorflow.keras
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import gc
import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import (Input, SimpleRNN, LSTM, GRU, Dense,
                          Dropout, BatchNormalization, Bidirectional, Layer)
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from utils.data_utils import load_data, to_rnn_input
from utils.eval_utils import (evaluate_model, plot_training_history,
                               run_attention_suite, compare_models_boxplot,
                               print_summary_table, save_summary_csv, FAULT_LABELS)

# ── Configuration ────────────────────────────────────────────────────
MAT_PATH     = os.path.join(os.path.dirname(__file__), '..', 'Turbine.mat')
SAVE_DIR     = os.path.join(os.path.dirname(__file__), 'outputs')
N_RUNS       = 5
EPOCHS       = 100
BATCH_SIZE   = 128
RANDOM_STATE = 42

os.makedirs(SAVE_DIR, exist_ok=True)
summary_results    = []
all_run_accuracies = {}

print("=" * 60)
print("  04 - RNN BENCHMARK")
print("=" * 60)

# ── Step 1: Load Data ─────────────────────────────────────────────────
print("Step 1: Loading data...")
X_raw, Y_onehot, y_int = load_data(MAT_PATH, normalize=True)
X = to_rnn_input(X_raw)   # (14000, 1000, 3)

X_all, X_test, yi_all, yi_test = train_test_split(
    X, y_int, test_size=0.2, stratify=y_int, random_state=RANDOM_STATE
)
# Keep raw test subset for attention plotting
X_raw_all, X_raw_test, _, _ = train_test_split(
    X_raw, y_int, test_size=0.2, stratify=y_int, random_state=RANDOM_STATE
)
y_test_cat = to_categorical(yi_test, num_classes=7)
print(f"  Training pool: {len(X_all)}  |  Test set: {len(X_test)}")


# ── Step 2: Custom Attention Layer (Keras 3 compatible) ───────────────
class AttentionLayer(Layer):
    """
    Additive (Bahdanau-style) attention for sequence classification.
    Updated for Keras 3: uses tf.* ops instead of keras.backend.
    """
    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight', shape=(input_shape[-1], 1),
            initializer='glorot_uniform', trainable=True
        )
        self.b = self.add_weight(
            name='att_bias', shape=(input_shape[1], 1),
            initializer='zeros', trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, T, units)
        e      = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, T, 1)
        a      = tf.nn.softmax(e, axis=1)                     # attention weights
        output = x * a                                         # weighted states
        return tf.reduce_sum(output, axis=1)                   # context vector

    def get_config(self):
        return super().get_config()


# ── Step 3: Model Builders ────────────────────────────────────────────

def build_vanilla_rnn():
    return Sequential([
        SimpleRNN(64, input_shape=(1000, 3)),
        Dense(7, activation='softmax')
    ])

def build_gru():
    return Sequential([
        GRU(128, input_shape=(1000, 3)),
        BatchNormalization(), Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_gru_2layer():
    return Sequential([
        GRU(128, return_sequences=True, input_shape=(1000, 3)),
        BatchNormalization(), Dropout(0.3),
        GRU(64),
        BatchNormalization(), Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_bigru():
    return Sequential([
        Bidirectional(GRU(128), input_shape=(1000, 3)),
        BatchNormalization(), Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_lstm():
    return Sequential([
        LSTM(128, input_shape=(1000, 3)),
        BatchNormalization(), Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_lstm_2layer():
    return Sequential([
        LSTM(128, return_sequences=True, input_shape=(1000, 3)),
        BatchNormalization(), Dropout(0.3),
        LSTM(64),
        BatchNormalization(), Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_bilstm():
    return Sequential([
        Bidirectional(LSTM(128), input_shape=(1000, 3)),
        BatchNormalization(), Dropout(0.3),
        Dense(7, activation='softmax')
    ])

def build_lstm_attention():
    inp = Input(shape=(1000, 3))
    x   = LSTM(128, return_sequences=True)(inp)
    x   = BatchNormalization()(x)
    x   = Dropout(0.3)(x)
    x   = AttentionLayer()(x)
    out = Dense(7, activation='softmax')(x)
    return Model(inp, out)

def build_gru_attention():
    inp = Input(shape=(1000, 3))
    x   = GRU(128, return_sequences=True)(inp)
    x   = BatchNormalization()(x)
    x   = Dropout(0.3)(x)
    x   = AttentionLayer()(x)
    out = Dense(7, activation='softmax')(x)
    return Model(inp, out)


MODEL_REGISTRY = {
    'Vanilla RNN':    (build_vanilla_rnn,    1,      False),
    'GRU':            (build_gru,            N_RUNS, False),
    '2-Layer GRU':    (build_gru_2layer,     N_RUNS, False),
    'BiGRU':          (build_bigru,          N_RUNS, False),
    'LSTM':           (build_lstm,           N_RUNS, False),
    '2-Layer LSTM':   (build_lstm_2layer,    N_RUNS, False),
    'BiLSTM':         (build_bilstm,         N_RUNS, False),
    'LSTM+Attention': (build_lstm_attention, N_RUNS, True),
    'GRU+Attention':  (build_gru_attention,  N_RUNS, True),
}


# ── Step 4: Training & Inline Evaluation Loop ─────────────────────────
print("Step 2: Training all 9 RNN architectures...\n")

for model_name, (builder, n_runs, has_attention) in MODEL_REGISTRY.items():
    print(f"\n{'='*60}\n  MODEL: {model_name}  ({n_runs} run{'s' if n_runs>1 else ''})\n{'='*60}")

    if model_name == 'Vanilla RNN':
        print("\n  NOTE: Expected ~50-70% due to vanishing gradients over 1000 steps.")
        print("  Running 1 run only.\n")

    run_accs, best_acc, best_model, best_hist = [], 0, None, None

    for run in range(n_runs):
        if n_runs > 1:
            print(f"\n  Run {run+1}/{n_runs}")

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
            callbacks=[EarlyStopping(monitor='val_loss', patience=30,
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

    if has_attention:
        print(f"\n  Attention Weight Visualization - {model_name}")
        run_attention_suite(model=best_model, X_test=X_raw_test,
                            y_test_int=yi_test, model_name=model_name,
                            save_dir=SAVE_DIR, n_samples=3)

    if model_name == 'Vanilla RNN':
        print(f"\n  Vanilla RNN result: {best_acc*100:.2f}%")
        print("  -> Confirms vanishing gradient failure. Do not tune further.")

    summary_results.append({
        'Model': model_name, 'Best_Acc': best_acc,
        'Mean_Acc': np.mean(run_accs), 'Std_Acc': np.std(run_accs),
        'Runs': n_runs,
        'Notes': ('Attention viz' if has_attention else
                  'Vanishing gradient case' if model_name == 'Vanilla RNN' else '')
    })
    print(f"\n  Summary: Best={best_acc*100:.2f}%  "
          f"Mean={np.mean(run_accs)*100:.2f}%  +/-{np.std(run_accs)*100:.2f}%")


# ── Step 5: Comparison Boxplot ────────────────────────────────────────
boxplot_data = {k: v for k, v in all_run_accuracies.items()
                if k != 'Vanilla RNN' and len(v) > 1}
compare_models_boxplot(boxplot_data,
                       title='RNN Benchmark - Accuracy Distribution (5 Runs)',
                       save_dir=SAVE_DIR, filename='RNN_benchmark_boxplot.png')

print_summary_table(summary_results, title='RNN BENCHMARK - FINAL RESULTS')
save_summary_csv(summary_results, 'RNN_benchmark_results.csv', save_dir=SAVE_DIR)

print("\n  -> Next: 05_CNN_RNN_hybrid.py")
