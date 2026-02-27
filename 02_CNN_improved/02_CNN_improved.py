import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

"""
02_CNN_Improved.py
==================
IMPROVED CNN: Adds the critical fixes missing from the baseline.

Changes from baseline (01):
    ✅ Input normalization (X / max_abs → [-1, 1])
    ✅ EarlyStopping (patience=30) + ReduceLROnPlateau
    ✅ BatchNormalization after every Conv1D layer
    ✅ Data augmentation: Gaussian noise + time shift + random gain
    ✅ Still 2-way split (fixed fully in 03)

Why these matter:
    Normalization:     Prevents gradient blow-up; accelerometer axes can have
                       very different raw magnitudes.
    BatchNorm:         Stabilizes activations between layers; allows higher
                       learning rate without diverging.
    Augmentation:      3x more training data; noise teaches robustness to
                       sensor imperfections; shift teaches location-invariance.
    EarlyStopping:     Saves the best weights; prevents overfitting after
                       the model has converged.
    ReduceLROnPlateau: Decays LR when val_loss stops improving — helps squeeze
                       out the last bit of performance.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                                     Dense, Flatten, Concatenate, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils.data_utils import load_data, augment_training_data, to_channel_inputs
from utils.eval_utils import evaluate_model, plot_training_history

# ── Configuration ────────────────────────────────────────────────────
MAT_PATH     = 'Turbine.mat'
EPOCHS       = 100
BATCH_SIZE   = 64
RANDOM_STATE = 42

# ── Step 1: Load and Normalize ────────────────────────────────────────
X, Y_onehot, y_int = load_data(MAT_PATH, normalize=True)

# ── Step 2: Train / Test Split ────────────────────────────────────────
X_train, X_test, y_train, y_test, yi_train, yi_test = (
    lambda a, b, c, d, e, f: (a, b, c, d, e, f)
)(*train_test_split(X, Y_onehot, y_int, test_size=0.2, random_state=RANDOM_STATE))

# ── Step 3: Augment Training Data ─────────────────────────────────────
# Applied to train only — test set must remain clean for honest evaluation
X_train_aug, y_train_aug = augment_training_data(
    X_train, y_train, noise=True, shift=True, gain=True
)

X_train_ch = to_channel_inputs(X_train_aug)
X_test_ch  = to_channel_inputs(X_test)

# ── Step 4: Build Model with BatchNorm ───────────────────────────────
def build_branch(input_shape=(1000, 1)):
    """
    CNN branch with BatchNormalization after each convolution.
    BN normalizes activations within each mini-batch, reducing
    internal covariate shift and making training more stable.
    """
    inp = Input(shape=input_shape)
    x = Conv1D(64, 14, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(64, 14, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(64, 14, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(64, 14, activation='relu')(x)
    x = BatchNormalization()(x)
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
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Step 5: Callbacks ─────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss', patience=30,
    restore_best_weights=True, verbose=1
)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=10, min_lr=1e-6, verbose=1
)

# ── Step 6: Train ─────────────────────────────────────────────────────
history = model.fit(
    X_train_ch, y_train_aug,
    validation_data=(X_test_ch, y_test),  # still 2-way, fixed in 03
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ── Step 7: Evaluate ──────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)
plot_training_history(history, model_name='CNN Improved', save_dir='outputs')
evaluate_model(model, X_test_ch, y_test, yi_test,
               model_name='CNN Improved',
               save_dir='outputs',
               is_branch_input=True)

# Save model
from utils.eval_utils import save_model
save_model(model, 'CNN_Improved', save_dir='outputs')

print("\n→ Next: 03_CNN_multiscale_final.py  (multi-scale kernels + honest 3-way split)")
