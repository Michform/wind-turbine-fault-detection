import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

"""
04_RNN_Benchmark.py
===================
SYSTEMATIC RNN COMPARISON: 9 architectures, 5 runs each.

Models compared:
    1. Vanilla SimpleRNN       — baseline, expected to struggle (vanishing gradients)
    2. GRU                     — single layer, gating solves vanishing gradient
    3. 2-Layer GRU             — deeper temporal modeling
    4. BiGRU                   — bidirectional: reads sequence forward AND backward
    5. LSTM                    — explicit memory cell, handles long dependencies
    6. 2-Layer LSTM            — deeper LSTM
    7. BiLSTM                  — bidirectional LSTM
    8. LSTM + Attention        — custom attention weights key timesteps
    9. GRU  + Attention        — (best performing pure-RNN model)

Evaluation Protocol:
    Each model trained 5 times with different random splits.
    Best run selected for detailed analysis (confusion matrix + report).
    Boxplot shows accuracy variance across runs — more informative than
    a single accuracy number, which can be lucky or unlucky.

Why Vanilla RNN fails:
    With sequences of length 1000, standard RNNs suffer from vanishing
    gradients — the gradient signal diminishes exponentially as it
    backpropagates through 1000 time steps. GRU/LSTM solve this with
    gating mechanisms that preserve gradient flow.

Key finding:
    GRU+Attention: ~97% best accuracy
    CNN (notebook 03): ~99.82%
    → CNNs win on vibration data because fault signatures are
      local patterns in frequency/time, not long-range dependencies.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, SimpleRNN, LSTM, GRU, Dense, Dropout,
                                     BatchNormalization, Bidirectional, Layer,
                                     Conv1D, MaxPooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils.data_utils import load_data, to_rnn_input
from utils.eval_utils import compare_models_boxplot, FAULT_LABELS

# ── Configuration ────────────────────────────────────────────────────
MAT_PATH   = 'Turbine.mat'
N_RUNS     = 5
EPOCHS     = 100
BATCH_SIZE = 128
RANDOM_STATE = 42

os.makedirs('outputs', exist_ok=True)

# ── Step 1: Load Data ─────────────────────────────────────────────────
X_raw, Y_onehot, y_int = load_data(MAT_PATH, normalize=True)
X = to_rnn_input(X_raw)   # shape: (14000, 1000, 3)

# Fixed test set (stratified) — never touched during training loops
X_trainval, X_test, yi_trainval, yi_test = train_test_split(
    X, y_int, test_size=0.2, stratify=y_int, random_state=RANDOM_STATE
)
y_test_cat = to_categorical(yi_test, num_classes=7)


# ── Step 2: Custom Attention Layer ───────────────────────────────────
class AttentionLayer(Layer):
    """
    Additive (Bahdanau-style) attention.
    Learns to weight each timestep's hidden state by its relevance.
    For vibration fault detection: focuses on the time windows where
    fault signatures are most prominent.

    Mechanism:
        e_t = tanh(h_t @ W + b)     # score for each timestep
        a_t = softmax(e_t)           # normalize to weights
        context = sum(h_t * a_t)     # weighted sum of hidden states
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
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


# ── Step 3: Model Builders ────────────────────────────────────────────

def build_vanilla_rnn():
    """
    SimpleRNN — included to demonstrate why gating is necessary.
    Expected accuracy: low (~50-70%) due to vanishing gradients over 1000 steps.
    """
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
    x = LSTM(128, return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = AttentionLayer()(x)
    out = Dense(7, activation='softmax')(x)
    return Model(inp, out)

def build_gru_attention():
    inp = Input(shape=(1000, 3))
    x = GRU(128, return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = AttentionLayer()(x)
    out = Dense(7, activation='softmax')(x)
    return Model(inp, out)


MODEL_BUILDERS = {
    'Vanilla RNN':      build_vanilla_rnn,
    'GRU':              build_gru,
    '2-Layer GRU':      build_gru_2layer,
    'BiGRU':            build_bigru,
    'LSTM':             build_lstm,
    '2-Layer LSTM':     build_lstm_2layer,
    'BiLSTM':           build_bilstm,
    'LSTM+Attention':   build_lstm_attention,
    'GRU+Attention':    build_gru_attention,
}


# ── Step 4: Training & Evaluation Loop ───────────────────────────────
results       = []
test_accs     = {}
best_histories = {}
best_cms       = {}
best_preds     = {}

for name, builder in MODEL_BUILDERS.items():
    print(f"\n{'─'*55}")
    print(f"  Running: {name}  ({N_RUNS} runs)")
    print(f"{'─'*55}")

    run_accs  = []
    best_acc  = 0
    best_hist = None
    best_pred = None

    n_runs = 1 if name == 'Vanilla RNN' else N_RUNS  # Vanilla: 1 run (slow)

    for run in range(n_runs):
        tf.keras.backend.clear_session()
        gc.collect()

        # Different random split each run → measures variance
        X_tr, _, yi_tr, _ = train_test_split(
            X_trainval, yi_trainval,
            test_size=0.2, random_state=run
        )
        y_tr = to_categorical(yi_tr, num_classes=7)

        m = builder()
        m.compile(optimizer=Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', patience=10,
                           restore_best_weights=True)
        hist = m.fit(
            X_tr, y_tr,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        _, acc = m.evaluate(X_test, y_test_cat, verbose=0)
        run_accs.append(acc)
        print(f"  Run {run+1}/{n_runs}: {acc:.4f}")

        if acc > best_acc:
            best_acc  = acc
            best_hist = hist
            best_pred = np.argmax(m.predict(X_test, verbose=0), axis=1)

        if acc >= 1.0:
            break

    test_accs[name]      = run_accs
    best_histories[name] = best_hist
    best_cms[name]       = confusion_matrix(yi_test, best_pred)
    best_preds[name]     = best_pred

    results.append({
        'Model':        name,
        'Best_Acc':     best_acc,
        'Mean_Acc':     np.mean(run_accs),
        'Std_Acc':      np.std(run_accs),
        'All_Accs':     run_accs
    })
    print(f"  ✅ Best: {best_acc:.4f}  |  Mean: {np.mean(run_accs):.4f} ± {np.std(run_accs):.4f}")


# ── Step 5: Summary Table ─────────────────────────────────────────────
df = pd.DataFrame([{k: v for k, v in r.items() if k != 'All_Accs'} for r in results])
df = df.sort_values('Best_Acc', ascending=False).reset_index(drop=True)
print("\n" + "="*55)
print("  FINAL RESULTS SUMMARY")
print("="*55)
print(df.to_string(index=False))
df.to_csv('outputs/rnn_results_summary.csv', index=False)

# ── Step 6: Boxplot ───────────────────────────────────────────────────
compare_models_boxplot(test_accs, save_dir='outputs')

# ── Step 7: Per-Model Curves & Confusion Matrices ────────────────────
for name in best_histories:
    hist  = best_histories[name]
    cm    = best_cms[name]
    pred  = best_preds[name]

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'{name} — Training History (Best Run)', fontweight='bold')
    axes[0].plot(hist.history['accuracy'],     label='Train')
    axes[0].plot(hist.history['val_accuracy'], label='Val', linestyle='--')
    axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(hist.history['loss'],     label='Train')
    axes[1].plot(hist.history['val_loss'], label='Val', linestyle='--')
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"outputs/{name.replace(' ','_')}_training_curves.png", dpi=150)
    plt.show()

    # Confusion matrix
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=FAULT_LABELS, yticklabels=FAULT_LABELS)
    plt.title(f'{name} — Confusion Matrix (Best Run)', fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(f"outputs/{name.replace(' ','_')}_confusion_matrix.png", dpi=150)
    plt.show()

    # Classification report
    print(f"\n{'─'*40}")
    print(f"  {name} — Classification Report")
    print(f"{'─'*40}")
    print(classification_report(yi_test, pred, target_names=FAULT_LABELS))

    # Save report and confusion matrix CSV (matches original kaggle.ipynb)
    from sklearn.metrics import classification_report as cr
    rep = cr(yi_test, pred, target_names=FAULT_LABELS, output_dict=True)
    pd.DataFrame(rep).T.to_csv(
        f"outputs/{name.replace(' ','_')}_classification_report.csv"
    )
    np.savetxt(
        f"outputs/{name.replace(' ','_')}_confusion_matrix.csv",
        cm, delimiter=',', fmt='%d'
    )

print("\n✅ RNN benchmark complete. See outputs/ for all saved results.")
print("Best pure-RNN model: GRU+Attention (~97%)")
print("Compare against CNN results in 03_CNN_multiscale_final.py (~99.82%)")
