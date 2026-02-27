import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

"""
01_CNN_Baseline.py
==================
BASELINE: Multi-Branch 1D CNN — no augmentation, no normalization.

PURPOSE:
    Establish whether the core architecture works at all.
    This is the starting point — not the final answer.
    Every improvement in notebooks 02 and 03 is measured against this.

Architecture:
    3 parallel CNN branches, one per vibration axis (X, Y, Z)
    Each branch: 4x [Conv1D(64, kernel=14) → MaxPool(2)]
    Branches concatenated → Dense(64) → Dense(7, softmax)

Known limitations of this version (addressed in later notebooks):
    - No input normalization
    - No data augmentation
    - No EarlyStopping
    - 2-way split: test set used as validation (minor leakage)
    - Fixed kernel size (14) applied to all layers
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model

from utils.data_utils import load_data, to_channel_inputs
from utils.eval_utils import evaluate_model, plot_training_history

# ── Configuration ────────────────────────────────────────────────────
MAT_PATH     = 'Turbine.mat'   # update to your path
EPOCHS       = 50
BATCH_SIZE   = 64
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Step 1: Load Data ─────────────────────────────────────────────────
# normalize=False to match original baseline experiment
X, Y_onehot, y_int = load_data(MAT_PATH, normalize=False)

# ── Step 2: Train / Test Split (2-way for baseline) ───────────────────
X_train, X_test, y_train, y_test, yi_train, yi_test = (
    lambda a, b, c, d, e, f: (a, b, c, d, e, f)
)(*train_test_split(X, Y_onehot, y_int, test_size=TEST_SIZE,
                    random_state=RANDOM_STATE))

X_train_ch = to_channel_inputs(X_train)
X_test_ch  = to_channel_inputs(X_test)

# ── Step 3: Build Model ───────────────────────────────────────────────
def build_branch(input_shape=(1000, 1)):
    inp = Input(shape=input_shape)
    x = Conv1D(64, 14, activation='relu')(inp)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 14, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 14, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 14, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
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
history = model.fit(
    X_train_ch, y_train,
    validation_data=(X_test_ch, y_test),  # NOTE: 2-way split, see 03 for fix
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ── Step 5: Evaluate ──────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)
plot_training_history(history, model_name='CNN Baseline', save_dir='outputs')
evaluate_model(model, X_test_ch, y_test, yi_test,
               model_name='CNN Baseline',
               save_dir='outputs',
               is_branch_input=True)

# Save model
from utils.eval_utils import save_model
save_model(model, 'CNN_Baseline', save_dir='outputs')

print("\n→ Next: 02_CNN_improved.py  (normalization + EarlyStopping + augmentation)")
