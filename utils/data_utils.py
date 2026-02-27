"""
utils/data_utils.py
-------------------
Shared data loading, preprocessing, splitting, and augmentation utilities.
Compatible with: TF 2.19.1 | Keras 3.13.2 | NumPy 2.0.2 | scikit-learn 1.6.1

Dataset: Turbine.mat
    X : (14000, 3000) -> reshaped to (14000, 3, 1000)
        3 channels = X, Y, Z accelerometer axes
        1000 time points per sample
    Y : (14000, 7) - one-hot encoded fault class labels

Fault Classes (0-6):
    0 : Normal operation
    1-6 : Fault types 1-6
"""

import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle


# ─────────────────────────────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────────────────────────────

def load_data(mat_path: str, normalize: bool = True):
    """
    Load and preprocess the Turbine.mat dataset.

    Returns:
        X        : np.ndarray shape (14000, 3, 1000) - float32
        y_onehot : np.ndarray shape (14000, 7)
        y_int    : np.ndarray shape (14000,)
    """
    mat   = scipy.io.loadmat(mat_path)
    X     = mat['X'].reshape(14000, 3, 1000).astype(np.float32)
    Y     = mat['Y'].astype(np.float32)
    y_int = np.argmax(Y, axis=1)
    if normalize:
        X = X / np.max(np.abs(X))
    return X, Y, y_int


# ─────────────────────────────────────────────────────────────────────
# 2. Splits
# ─────────────────────────────────────────────────────────────────────

def split_data(X, y_onehot, y_int,
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42):
    """3-way stratified split: train / val / test (70/15/15 default)."""
    X_tv, X_test, y_tv, y_test, yi_tv, yi_test = train_test_split(
        X, y_onehot, y_int,
        test_size=test_size, stratify=y_int, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, yi_train, yi_val = train_test_split(
        X_tv, y_tv, yi_tv,
        test_size=relative_val, stratify=yi_tv, random_state=random_state
    )
    print(f"  Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Class balance check (test): {np.bincount(yi_test)}")
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            yi_train, yi_val, yi_test)


def split_data_2way(X, y_onehot, y_int,
                    test_size: float = 0.2,
                    random_state: int = 42):
    """2-way stratified split (baseline only - documented leakage flaw)."""
    X_train, X_test, y_train, y_test, yi_train, yi_test = train_test_split(
        X, y_onehot, y_int,
        test_size=test_size, stratify=y_int, random_state=random_state
    )
    print(f"  2-way split -> Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, yi_train, yi_test


# ─────────────────────────────────────────────────────────────────────
# 3. Augmentation (training set only)
# ─────────────────────────────────────────────────────────────────────

def add_gaussian_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    return (data + np.random.normal(0, noise_level, data.shape)).astype(np.float32)

def time_shift(sample: np.ndarray, max_shift: int = 50) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(sample, shift, axis=1)

def random_gain(sample: np.ndarray, gain_range: tuple = (0.9, 1.1)) -> np.ndarray:
    return (sample * np.random.uniform(*gain_range)).astype(np.float32)

def augment_training_data(X_train: np.ndarray, y_train: np.ndarray,
                          noise: bool = True, shift: bool = True,
                          gain: bool = True) -> tuple:
    """Apply augmentation - each strategy appends a full copy (up to 4x size)."""
    X_parts, y_parts = [X_train], [y_train]
    if noise:
        X_parts.append(add_gaussian_noise(X_train))
        y_parts.append(y_train)
    if shift:
        X_parts.append(np.array([time_shift(s) for s in X_train]))
        y_parts.append(y_train)
    if gain:
        X_parts.append(np.array([random_gain(s) for s in X_train]))
        y_parts.append(y_train)
    X_out = np.concatenate(X_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)
    X_out, y_out = sk_shuffle(X_out, y_out, random_state=42)
    print(f"  Augmentation: {len(X_train)} -> {len(X_out)} samples ({len(X_parts)}x original)")
    return X_out, y_out


# ─────────────────────────────────────────────────────────────────────
# 4. Input Format Converters
# ─────────────────────────────────────────────────────────────────────

def to_channel_inputs(X: np.ndarray) -> list:
    """
    Convert (N, 3, 1000) -> list of 3 float32 arrays each (N, 1000, 1).
    Pass the result to make_dataset() for Keras 3 model.fit() compatibility.
    """
    return [
        X[:, 0, :].reshape(-1, 1000, 1).astype(np.float32),
        X[:, 1, :].reshape(-1, 1000, 1).astype(np.float32),
        X[:, 2, :].reshape(-1, 1000, 1).astype(np.float32),
    ]


def make_dataset(X_ch: list, y: np.ndarray,
                 batch_size: int = 64,
                 shuffle_data: bool = False) -> tf.data.Dataset:
    """
    Wrap multi-branch inputs into a tf.data.Dataset.

    REQUIRED for Keras 3.x: passing a plain list to model.fit() for
    multi-input models raises InvalidArgumentError. tf.data.Dataset
    is the correct approach for Keras 3.

    Args:
        X_ch         : List of 3 arrays from to_channel_inputs().
        y            : Labels array, shape (N, 7).
        batch_size   : Batch size.
        shuffle_data : True for training set only.

    Returns:
        tf.data.Dataset yielding ((branch0, branch1, branch2), labels).
    """
    ds = tf.data.Dataset.from_tensor_slices(
        ((X_ch[0], X_ch[1], X_ch[2]), y)
    )
    if shuffle_data:
        ds = ds.shuffle(buffer_size=len(y), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def to_rnn_input(X: np.ndarray) -> np.ndarray:
    """Convert (N, 3, 1000) -> (N, 1000, 3) for RNN models."""
    return X.transpose(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────
# 5. Dataset Interrogation
# ─────────────────────────────────────────────────────────────────────

def class_distribution(y_int: np.ndarray, fault_labels: list) -> dict:
    counts = np.bincount(y_int)
    total  = len(y_int)
    return {
        fault_labels[i]: {'count': int(counts[i]),
                          'pct': round(counts[i] / total * 100, 2)}
        for i in range(len(counts))
    }

def per_axis_stats(X: np.ndarray) -> dict:
    stats = {}
    for i, ax in enumerate(['X', 'Y', 'Z']):
        data = X[:, i, :].flatten()
        stats[ax] = {
            'mean': float(np.mean(data)), 'std':     float(np.std(data)),
            'min':  float(np.min(data)),  'max':     float(np.max(data)),
            'abs_max': float(np.max(np.abs(data))),
        }
    return stats
