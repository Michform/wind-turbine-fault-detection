"""
01_CNN_baseline.py
==================
BASELINE: Multi-Branch 1D CNN — no augmentation, no normalization.
Stable version: TF 2.19 + tf.keras only
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, Flatten, Concatenate
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
summary_results = []

print("=" * 60)
print("  01 - CNN BASELINE")
print("=" * 60)

# ── Step 1: Load Data ─────────────────────────────────────────────────
X, Y_onehot, y_int = load_data(MAT_PATH, normalize=False)

# ── Step 2: Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, yi_train, yi_test = split_data_2way(
    X, Y_onehot, y_int, test_size=0.2, random_state=RANDOM_STATE
)

X_train_ch = to_channel_inputs(X_train)
X_test_ch  = to_channel_inputs(X_test)

# ── Step 3: Build Model ───────────────────────────────────────────────
def build_branch(input_shape=(1000, 1)):
    inp = Input(shape=input_shape)
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

branches, model_inputs = [], []
for _ in range(3):
    inp, out = build_branch()
    model_inputs.append(inp)
    branches.append(out)

merged = Concatenate()(branches)
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.5)(merged)
output = Dense(Y_onehot.shape[1], activation='softmax')(merged)

model = Model(inputs=model_inputs, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ── Step 4: Train ─────────────────────────────────────────────────────
history = model.fit(
    X_train_ch,
    y_train,
    validation_data=(X_test_ch, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ── Step 5: Evaluation ────────────────────────────────────────────────
plot_training_history(history, model_name='CNN Baseline', save_dir=SAVE_DIR)

test_acc, y_pred_int = evaluate_model(
    model=model,
    X_test=X_test_ch,
    y_test_onehot=y_test,
    y_test_int=yi_test,
    model_name='CNN Baseline',
    save_dir=SAVE_DIR,
    is_branch_input=True
)

summary_results.append({
    'Model': 'CNN Baseline',
    'Best_Acc': test_acc,
    'Mean_Acc': test_acc,
    'Std_Acc': 0.0,
    'Runs': 1,
    'Notes': '2-way split, no norm, no aug'
})

save_model(model, 'CNN_Baseline', save_dir=SAVE_DIR)
print_summary_table(summary_results, title='CNN BASELINE - RESULTS SUMMARY')
save_summary_csv(summary_results, 'CNN_Baseline_results.csv', save_dir=SAVE_DIR)

print(f"\nBaseline accuracy: {test_acc*100:.2f}%")
