"""
01_CNN_baseline.py
==================
BASELINE: Multi-Branch 1D CNN — no augmentation, no normalization.
Compatible with: TF 2.19.1 | Keras 3.13.2 | NumPy 2.0.2

Key Keras 3 change: model.fit() now uses make_dataset() to wrap
multi-branch inputs into tf.data.Dataset instead of passing lists directly.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import keras
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, Flatten, Concatenate
from keras.models import Model

from utils.data_utils import load_data, split_data_2way, to_channel_inputs, make_dataset
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
print("\n  Limitations preserved intentionally:")
print("  - No normalization")
print("  - No augmentation")
print("  - 2-way split (test set leaks into EarlyStopping)")
print("  - Single run (no variance estimate)")
print("  These are fixed progressively in scripts 02 and 03.\n")

# ── Step 1: Load Data ─────────────────────────────────────────────────
print("Step 1: Loading data (normalize=False for baseline)...")
X, Y_onehot, y_int = load_data(MAT_PATH, normalize=False)
print(f"  X shape: {X.shape}  |  Y shape: {Y_onehot.shape}")

# ── Step 2: Split ─────────────────────────────────────────────────────
print("\nStep 2: 2-way train/test split (known limitation)...")
X_train, X_test, y_train, y_test, yi_train, yi_test = split_data_2way(
    X, Y_onehot, y_int, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\n  NOTE: X_test is used as validation — information leakage.")
print(f"  Script 03 corrects this with a proper 3-way stratified split.\n")

X_train_ch = to_channel_inputs(X_train)
X_test_ch  = to_channel_inputs(X_test)

# Keras 3: wrap in tf.data.Dataset for model.fit()
train_ds = make_dataset(X_train_ch, y_train, batch_size=BATCH_SIZE, shuffle_data=True)
val_ds   = make_dataset(X_test_ch,  y_test,  batch_size=BATCH_SIZE, shuffle_data=False)

# For evaluation (model.evaluate / model.predict) use the dataset too
test_ds  = make_dataset(X_test_ch, y_test, batch_size=BATCH_SIZE, shuffle_data=False)

# ── Step 3: Build Model ───────────────────────────────────────────────
print("Step 3: Building baseline multi-branch CNN...")

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
model.summary()

# ── Step 4: Train ─────────────────────────────────────────────────────
print("\nStep 4: Training (50 epochs, no EarlyStopping)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# ── Step 5: Inline Evaluation ─────────────────────────────────────────
print("\nStep 5: Inline evaluation...")

plot_training_history(history, model_name='CNN Baseline', save_dir=SAVE_DIR)

test_acc, y_pred_int = evaluate_model(
    model=model,
    X_test=test_ds,
    y_test_onehot=y_test,
    y_test_int=yi_test,
    model_name='CNN Baseline',
    save_dir=SAVE_DIR,
    is_branch_input=True
)

print("\n  Note: Grad-CAM deferred to script 03 (final model).")

summary_results.append({
    'Model':    'CNN Baseline',
    'Best_Acc': test_acc,
    'Mean_Acc': test_acc,
    'Std_Acc':  0.0,
    'Runs':     1,
    'Notes':    '2-way split, no norm, no aug - LEAKY EVALUATION'
})

# ── Step 6: Save ──────────────────────────────────────────────────────
save_model(model, 'CNN_Baseline', save_dir=SAVE_DIR)
print_summary_table(summary_results, title='CNN BASELINE - RESULTS SUMMARY')
save_summary_csv(summary_results, 'CNN_Baseline_results.csv', save_dir=SAVE_DIR)

print("\n" + "=" * 60)
print(f"  Baseline accuracy: {test_acc*100:.2f}%  (OPTIMISTIC - leaky eval)")
print("  -> Next: 02_CNN_improved.py")
print("=" * 60)
