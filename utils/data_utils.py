"""
utils/data_utils.py
-------------------
Shared data loading, preprocessing, and augmentation utilities
used across all notebooks in this project.

Dataset: Turbine.mat
  - X: (14000, 3000) → reshaped to (14000, 3, 1000)
    3 channels = X, Y, Z accelerometer axes
    1000 time points per sample @ 100kHz sampling rate
  - Y: (14000, 7) one-hot encoded fault class labels

Fault Classes (0–6):
  0: Normal operation
  1–6: Progressive fault stages / fault types in the drivetrain
"""

import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_data(mat_path: str, normalize: bool = True):
    """
    Load and preprocess the Turbine.mat dataset.

    Args:
        mat_path:  Path to the .mat file.
        normalize: If True, scale X to [-1, 1] using global max abs.
                   Always do this — raw vibration amplitudes vary widely.

    Returns:
        X: np.ndarray of shape (14000, 3, 1000)
        y_onehot: np.ndarray of shape (14000, 7)  — one-hot labels
        y_int:    np.ndarray of shape (14000,)    — integer class labels
    """
    mat = scipy.io.loadmat(mat_path)
    X = mat['X'].reshape(14000, 3, 1000).astype(np.float32)
    Y = mat['Y'].astype(np.float32)

    if normalize:
        X = X / np.max(np.abs(X))

    y_int = np.argmax(Y, axis=1)
    return X, Y, y_int


# ─────────────────────────────────────────────
# Train / Val / Test Split  (3-way, honest)
# ─────────────────────────────────────────────

def split_data(X, y_onehot, y_int,
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42):
    """
    Proper 3-way stratified split: train / validation / test.

    Why 3-way?
      Using a separate validation set for EarlyStopping means the held-out
      test set was NEVER seen during training — giving an honest accuracy number.
      A 2-way split where test data drives EarlyStopping leaks information.

    Split sizes (defaults): 70% train | 15% val | 15% test

    Returns:
        X_train, X_val, X_test,
        y_train, y_val, y_test,         (one-hot)
        y_train_int, y_val_int, y_test_int  (integer labels)
    """
    # First: carve out test set
    X_trainval, X_test, y_trainval, y_test, yi_trainval, yi_test = train_test_split(
        X, y_onehot, y_int,
        test_size=test_size,
        stratify=y_int,
        random_state=random_state
    )

    # Second: split remainder into train and val
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, yi_train, yi_val = train_test_split(
        X_trainval, y_trainval, yi_trainval,
        test_size=relative_val,
        stratify=yi_trainval,
        random_state=random_state
    )

    print(f"Split sizes — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            yi_train, yi_val, yi_test)


# ─────────────────────────────────────────────
# Data Augmentation  (training set only)
# ─────────────────────────────────────────────

def add_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Additive Gaussian noise.
    Simulates real-world sensor noise and prevents memorization of exact waveforms.
    """
    return data + np.random.normal(0, noise_level, data.shape).astype(np.float32)


def time_shift(sample: np.ndarray, max_shift: int = 50) -> np.ndarray:
    """
    Circular time-shift of a single sample (shape: 3 x 1000).
    Teaches the model that fault signatures are shift-invariant.
    """
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(sample, shift, axis=1)


def random_gain(sample: np.ndarray, gain_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """
    Random amplitude scaling.
    Accounts for sensor calibration drift between turbines.
    """
    gain = np.random.uniform(*gain_range)
    return (sample * gain).astype(np.float32)


def augment_training_data(X_train: np.ndarray, y_train: np.ndarray,
                          noise: bool = True,
                          shift: bool = True,
                          gain: bool = True) -> tuple:
    """
    Apply all augmentation strategies to training data.
    Each strategy adds a full copy of the dataset → up to 4x original size.

    Returns:
        X_aug, y_aug — augmented arrays (shuffled)
    """
    X_aug = [X_train]
    y_aug = [y_train]

    if noise:
        X_aug.append(add_noise(X_train))
        y_aug.append(y_train)

    if shift:
        shifted = np.array([time_shift(s) for s in X_train])
        X_aug.append(shifted)
        y_aug.append(y_train)

    if gain:
        gained = np.array([random_gain(s) for s in X_train])
        X_aug.append(gained)
        y_aug.append(y_train)

    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)

    X_out, y_out = shuffle(X_out, y_out, random_state=42)
    print(f"Augmented training size: {len(X_out)} samples ({len(X_aug)}x original)")
    return X_out, y_out


# ─────────────────────────────────────────────
# Input Preparation  (CNN branch format)
# ─────────────────────────────────────────────

def to_channel_inputs(X: np.ndarray) -> list:
    """
    Convert (N, 3, 1000) array into a list of 3 arrays each shaped (N, 1000, 1).
    This matches the multi-branch CNN input format where each branch
    receives one vibration axis independently.
    """
    return [X[:, i, :].reshape(-1, 1000, 1) for i in range(3)]


def to_rnn_input(X: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3, 1000) to (N, 1000, 3) for RNN/LSTM/GRU input.
    RNNs expect (batch, timesteps, features).
    """
    return X.transpose(0, 2, 1)
